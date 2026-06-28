"""Meta-labeler — the secondary model that decides take/skip on a fired signal.

Given the leak-free decision rows from :mod:`decision_dataset_builder`, this
learns which of a strategy's signals are worth taking. Two design choices make it
a *trader* rather than a classifier:

1. **Objective = net P&L, not accuracy.** The decision threshold is tuned on the
   validation split to maximise summed ``label_fwd_ret_net`` of the taken set
   (subject to a minimum coverage), because a 99%-accurate model that only ever
   says "skip" makes no money.
2. **Cost-aware labels.** The target already bakes in costs (see
   ``decision_dataset_builder.make_label``), so a "take" must clear the hurdle.

Backend: scikit-learn's ``HistGradientBoostingClassifier`` when available
(handles the nonlinear feature interactions); otherwise a dependency-free,
L2-regularised logistic regression in pure numpy so it always runs on the
embedded interpreter. CPU only — never touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import schema

LOG = logging.getLogger("metalabel.meta_labeler")

NUMERIC_KEYS = schema.FEATURE_KEYS + schema.CONTEXT_NUMERIC_KEYS
CATEGORICAL_KEYS = list(schema.CATEGORICAL_KEYS)


# ── data loading ───────────────────────────────────────────────────────────────
def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _split(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r.get("split") == name]


# ── feature encoding ───────────────────────────────────────────────────────────
@dataclass
class Encoder:
    """Numeric passthrough + one-hot for categoricals, vocab fit on train only."""
    numeric: list[str]
    categorical: list[str]
    vocab: dict[str, list[str]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: list[dict]) -> "Encoder":
        for k in self.numeric:
            vals = np.array([float(r.get(k, 0.0) or 0.0) for r in rows], dtype=float)
            self.mean[k] = float(vals.mean()) if len(vals) else 0.0
            self.std[k] = float(vals.std()) or 1.0
        for k in self.categorical:
            seen = sorted({str(r.get(k, "")) for r in rows})
            self.vocab[k] = seen
        return self

    @property
    def columns(self) -> list[str]:
        cols = list(self.numeric)
        for k in self.categorical:
            cols += [f"{k}={v}" for v in self.vocab.get(k, [])]
        return cols

    def transform(self, rows: list[dict]) -> np.ndarray:
        out = np.zeros((len(rows), len(self.columns)), dtype=float)
        col_index = {c: i for i, c in enumerate(self.columns)}
        for i, r in enumerate(rows):
            for k in self.numeric:
                z = (float(r.get(k, 0.0) or 0.0) - self.mean[k]) / self.std[k]
                out[i, col_index[k]] = z
            for k in self.categorical:
                c = f"{k}={str(r.get(k, ''))}"
                j = col_index.get(c)
                if j is not None:
                    out[i, j] = 1.0
        return out


# ── numpy logistic-regression fallback ─────────────────────────────────────────
class _LogReg:
    def __init__(self, l2: float = 1.0, lr: float = 0.1, epochs: int = 400):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        n, d = X.shape
        self.w = np.zeros(d)
        w = sample_weight if sample_weight is not None else np.ones(n)
        w = w / w.mean()
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * w
            grad_w = X.T @ g / n + self.l2 * self.w / n
            grad_b = float(g.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _make_backend():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return ("sklearn", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0))
    except Exception:
        return ("logreg", _LogReg())


# ── threshold tuned for net P&L (the profit objective) ─────────────────────────
def tune_threshold(prob: np.ndarray, fwd_ret_net: np.ndarray,
                   min_coverage: float = 0.05) -> tuple[float, dict]:
    """Pick the probability cutoff that maximises summed net return of the taken
    set on the validation split, requiring at least ``min_coverage`` of signals
    be taken (so the model can't trivially win by skipping everything)."""
    best_thr, best_pnl, best_stats = 0.5, -1e18, {}
    n = len(prob)
    grid = np.unique(np.round(np.quantile(prob, np.linspace(0.0, 0.98, 50)), 4))
    for thr in grid:
        take = prob >= thr
        cov = take.mean() if n else 0.0
        if cov < min_coverage:
            continue
        pnl = float(fwd_ret_net[take].sum())
        if pnl > best_pnl:
            best_pnl = pnl
            best_thr = float(thr)
            taken = fwd_ret_net[take]
            best_stats = {"coverage": round(float(cov), 4),
                          "avg_ret_net": round(float(taken.mean()), 5),
                          "total_ret_net": round(pnl, 4),
                          "hit_rate": round(float((taken > 0).mean()), 4)}
    return best_thr, best_stats


# ── model ──────────────────────────────────────────────────────────────────────
@dataclass
class MetaLabeler:
    encoder: Encoder
    backend_kind: str
    model: object
    threshold: float = 0.5
    val_stats: dict = field(default_factory=dict)

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        X = self.encoder.transform(rows)
        if self.backend_kind == "sklearn":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)

    def decide(self, rows: list[dict]) -> list[dict]:
        p = self.predict_proba(rows)
        return [{"decision": "take" if pi >= self.threshold else "skip",
                 "p_take": round(float(pi), 4)} for pi in p]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "MetaLabeler":
        return pickle.loads(Path(path).read_bytes())


def fit_model(train_rows: list[dict], val_rows: list[dict] | None = None, *,
              min_coverage: float = 0.05) -> MetaLabeler:
    """Train on explicit row lists (used by both file-based ``train`` and the
    cross-validator, which builds folds in memory)."""
    if not train_rows:
        raise RuntimeError("no train rows")
    if not val_rows:                                  # tiny corpora: tune on train
        val_rows = train_rows

    enc = Encoder(NUMERIC_KEYS, CATEGORICAL_KEYS).fit(train_rows)
    Xtr = enc.transform(train_rows)
    ytr = np.array([1.0 if r["label_decision"] == "take" else 0.0 for r in train_rows])

    # Class-balance weighting so the model isn't swamped by the majority label.
    pos = max(ytr.sum(), 1.0)
    neg = max(len(ytr) - ytr.sum(), 1.0)
    sw = np.where(ytr == 1.0, len(ytr) / (2 * pos), len(ytr) / (2 * neg))

    kind, model = _make_backend()
    if kind == "sklearn":
        try:
            model.fit(Xtr, ytr, sample_weight=sw)
        except TypeError:
            model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr, sample_weight=sw)

    ml = MetaLabeler(encoder=enc, backend_kind=kind, model=model)
    pval = ml.predict_proba(val_rows)
    rval = np.array([float(r["label_fwd_ret_net"]) for r in val_rows])
    ml.threshold, ml.val_stats = tune_threshold(pval, rval, min_coverage=min_coverage)
    LOG.info("meta-labeler trained (%s); threshold=%.3f val=%s",
             kind, ml.threshold, ml.val_stats)
    return ml


def train(dataset_path: str | Path, *, min_coverage: float = 0.05) -> MetaLabeler:
    """File-based entry point: read the dataset, split by tag, fit."""
    rows = load_rows(dataset_path)
    return fit_model(_split(rows, "train"), _split(rows, "val"), min_coverage=min_coverage)


def main() -> int:
    ap = argparse.ArgumentParser(prog="meta_labeler")
    ap.add_argument("--dataset", required=True, help="path to decisions_tabular.jsonl")
    ap.add_argument("--out", required=True, help="where to save the trained model (.pkl)")
    ap.add_argument("--min-coverage", type=float, default=0.05)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ml = train(args.dataset, min_coverage=args.min_coverage)
    ml.save(args.out)
    print(f"saved {ml.backend_kind} meta-labeler -> {args.out}")
    print(f"threshold={ml.threshold:.3f}  val={ml.val_stats}")
    return 0


if __name__ == "__main__":
    # Re-enter through the real module name so trained models pickle as
    # ``meta_labeler.MetaLabeler`` (not ``__main__.MetaLabeler``) and reload
    # cleanly in other processes (e.g. eval_policy).
    import meta_labeler
    raise SystemExit(meta_labeler.main())
