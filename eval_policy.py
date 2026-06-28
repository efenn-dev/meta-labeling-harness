"""Evaluate a take/skip policy as a *trader*, on the held-out test split.

This is the metric that actually answers "will it make money." It does NOT report
classification accuracy — a meta-labeler can be very accurate and unprofitable.
Instead it simulates the policy: take the signals the model says to take, and
measure the net P&L, edge over taking every signal, per-trade Sharpe, and the
drawdown of the resulting equity curve, broken down by regime and archetype.

Decision sources (``--policy``):
* ``take_all``   — baseline: take every signal the strategy fired.
* ``metalabeler``— a trained model from :mod:`meta_labeler` (``--model-path``).
* ``random``     — coin-flip at ``--coverage`` (sanity floor).

Run the same test split through ``take_all`` and ``metalabeler``; if the model's
``avg_ret_net`` and total P&L beat ``take_all`` net of costs, it is adding alpha.

    python eval_policy.py --dataset decisions_tabular.jsonl \
        --policy metalabeler --model-path metalabeler.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LOG = logging.getLogger("metalabel.eval_policy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load(path: str, split: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("split") == split:
                rows.append(r)
    return rows


def _decisions(rows: list[dict], policy: str, *, model_path: str | None,
               coverage: float, seed: int) -> np.ndarray:
    """Boolean 'take' mask aligned to ``rows``."""
    if policy == "take_all":
        return np.ones(len(rows), dtype=bool)
    if policy == "random":
        rng = np.random.default_rng(seed)
        return rng.random(len(rows)) < coverage
    if policy == "metalabeler":
        import meta_labeler
        ml = meta_labeler.MetaLabeler.load(model_path)
        return np.array([d["decision"] == "take" for d in ml.decide(rows)], dtype=bool)
    raise SystemExit(f"unknown policy: {policy}")


def _metrics(rows: list[dict], take: np.ndarray) -> dict:
    rets = np.array([float(r["label_fwd_ret_net"]) for r in rows])
    n = len(rows)
    taken = rets[take]
    base_avg = float(rets.mean()) if n else 0.0
    out = {
        "signals": n,
        "coverage": round(float(take.mean()), 4) if n else 0.0,
        "n_taken": int(take.sum()),
        "hit_rate": round(float((taken > 0).mean()), 4) if len(taken) else 0.0,
        "avg_ret_net": round(float(taken.mean()), 5) if len(taken) else 0.0,
        "total_ret_net": round(float(taken.sum()), 4),
        "edge_vs_take_all": round((float(taken.mean()) - base_avg), 5) if len(taken) else 0.0,
        "take_all_avg_ret": round(base_avg, 5),
    }
    if len(taken) > 1 and taken.std() > 0:
        per_trade_sharpe = float(taken.mean() / taken.std())
        out["per_trade_sharpe"] = round(per_trade_sharpe, 4)
        # Rough annualisation by average holding period (bars≈trading days).
        hb = np.array([float(r.get("holding_bars", 1) or 1) for r in rows])[take]
        avg_hold = max(float(hb.mean()), 1.0)
        out["ann_sharpe_est"] = round(per_trade_sharpe * math.sqrt(252.0 / avg_hold), 3)
    # Equity curve of taken trades in entry-time order → max drawdown.
    idx = [i for i in range(n) if take[i]]
    idx.sort(key=lambda i: rows[i].get("entry_ts", ""))
    curve = np.cumsum([rets[i] for i in idx])
    if len(curve):
        peak = np.maximum.accumulate(curve)
        out["max_drawdown"] = round(float((curve - peak).min()), 4)
        out["final_equity_ret"] = round(float(curve[-1]), 4)
    return out


def _breakdown(rows: list[dict], take: np.ndarray, key: str) -> dict:
    groups: dict[str, list[float]] = {}
    for i, r in enumerate(rows):
        if take[i]:
            groups.setdefault(str(r.get(key, "?")), []).append(float(r["label_fwd_ret_net"]))
    return {g: {"n": len(v), "avg_ret_net": round(float(np.mean(v)), 5)}
            for g, v in sorted(groups.items())}


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval_policy")
    ap.add_argument("--dataset", required=True, help="path to decisions_tabular.jsonl")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--policy", default="metalabeler",
                    choices=["take_all", "metalabeler", "random"])
    ap.add_argument("--model-path", default=None, help="trained meta-labeler (.pkl)")
    ap.add_argument("--coverage", type=float, default=0.5, help="for --policy random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _load(args.dataset, args.split)
    if not rows:
        LOG.error("no %s-split rows in %s", args.split, args.dataset)
        return 2
    if args.policy == "metalabeler" and not args.model_path:
        LOG.error("--policy metalabeler requires --model-path")
        return 2

    take = _decisions(rows, args.policy, model_path=args.model_path,
                      coverage=args.coverage, seed=args.seed)
    base = np.ones(len(rows), dtype=bool)

    m = _metrics(rows, take)
    b = _metrics(rows, base)

    print("-" * 64)
    print(f"dataset   : {args.dataset}  (split={args.split}, {len(rows)} signals)")
    print(f"policy    : {args.policy}")
    print("-" * 64)
    print(f"{'metric':<22}{'policy':>16}{'take_all':>16}")
    for k in ("coverage", "n_taken", "hit_rate", "avg_ret_net", "total_ret_net",
              "per_trade_sharpe", "ann_sharpe_est", "max_drawdown", "final_equity_ret"):
        print(f"{k:<22}{str(m.get(k, '-')):>16}{str(b.get(k, '-')):>16}")
    print(f"{'edge_vs_take_all':<22}{str(m.get('edge_vs_take_all', '-')):>16}")
    print("-" * 64)
    avg = m.get("avg_ret_net", 0.0)
    base_avg = b.get("avg_ret_net", 0.0)
    if m.get("n_taken", 0) == 0:
        verdict = "NO TRADES (policy skipped everything)"
    elif avg > 0 and avg > base_avg:
        verdict = "PROFITABLE and beats take-all"
    elif avg > 0:
        verdict = "PROFITABLE (but no edge over take-all)"
    elif avg > base_avg:
        verdict = "REDUCES LOSS vs take-all, but still unprofitable net of costs"
    else:
        verdict = "NO EDGE vs take-all"
    print(f"verdict   : {verdict}")
    print("-" * 64)
    print("by regime   :", json.dumps(_breakdown(rows, take, "regime")))
    print("by archetype:", json.dumps(_breakdown(rows, take, "archetype")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
