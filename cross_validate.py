"""Purged walk-forward cross-validation for the decision meta-labeler.

A single train/test split can get lucky. This re-folds the decision dataset by
time into an expanding-window walk-forward: fold k trains on everything before
test block k and is scored on block k, with the same **purge + embargo** that the
dataset builder uses (a train trade whose holding window reaches into the test
block is dropped). It then asks the only questions that matter:

* Does the policy beat **take-all** net of costs — in how many folds?
* Does it beat **buy-and-hold** (beta) over the same windows — in how many folds?
* Is the per-fold edge *consistent*, or carried by one fold?

If the edge only shows up in one fold, it's noise. If it shows up in most folds
and clears beta, it's worth trusting (and worth GPU).

    python cross_validate.py --dataset decisions_tabular.jsonl --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import meta_labeler  # noqa: E402
import eval_policy  # noqa: E402 (same dir)

LOG = logging.getLogger("metalabel.cross_validate")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def _ts(x):
    import pandas as pd
    try:
        t = pd.Timestamp(x)
        return t.tz_localize(None) if t.tzinfo else t
    except Exception:
        return None


def _purge(train_rows: list[dict], test_start, embargo_days: int) -> list[dict]:
    import pandas as pd
    if test_start is None:
        return train_rows
    emb = pd.Timedelta(days=embargo_days)
    kept = []
    for r in train_rows:
        ex = _ts(r.get("exit_ts")) or _ts(r.get("entry_ts"))
        en = _ts(r.get("entry_ts"))
        if ex is None or en is None:
            continue
        if ex >= test_start or en >= (test_start - emb):
            continue                    # future reaches into test, or inside embargo
        kept.append(r)
    return kept


def cross_validate(dataset: str, *, folds: int = 5, embargo_days: int = 3,
                   min_coverage: float = 0.05, val_frac: float = 0.15) -> dict:
    rows = [json.loads(l) for l in Path(dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r.get("entry_ts", ""))
    n = len(rows)
    if n < (folds + 1) * 20:
        LOG.warning("only %d rows; folds may be thin", n)

    # (folds+1) contiguous time blocks; block 0 is train-only seed.
    edges = [int(n * k / (folds + 1)) for k in range(folds + 2)]
    fold_reports: list[dict] = []

    for k in range(1, folds + 1):
        test = rows[edges[k]:edges[k + 1]]
        train_all = rows[:edges[k]]
        if not test or not train_all:
            continue
        test_start = _ts(test[0]["entry_ts"])
        train_all = _purge(train_all, test_start, embargo_days)
        if len(train_all) < 30:
            continue
        cut = int(len(train_all) * (1 - val_frac))
        train, val = train_all[:cut], train_all[cut:]

        ml = meta_labeler.fit_model(train, val or None, min_coverage=min_coverage)
        take = np.array([d["decision"] == "take" for d in ml.decide(test)], dtype=bool)
        m = eval_policy._metrics(test, take)

        underlying = np.array([float(r.get("label_underlying_ret", 0.0)) for r in test])
        beta_all = float(underlying.mean()) if len(underlying) else 0.0
        beta_taken = float(underlying[take].mean()) if take.any() else 0.0
        fold_reports.append({
            "fold": k,
            "train": len(train), "test": len(test),
            "coverage": m.get("coverage", 0.0),
            "policy_avg": m.get("avg_ret_net", 0.0),
            "take_all_avg": m.get("take_all_avg_ret", 0.0),
            "beta_avg": round(beta_all, 5),
            "edge_vs_take_all": m.get("edge_vs_take_all", 0.0),
            "edge_vs_beta": round(m.get("avg_ret_net", 0.0) - beta_taken, 5),
            "hit_rate": m.get("hit_rate", 0.0),
            "ann_sharpe": m.get("ann_sharpe_est", 0.0),
            "max_dd": m.get("max_drawdown", 0.0),
        })

    if not fold_reports:
        return {"folds": [], "summary": {"note": "no usable folds"}}

    def _mean(key):
        return round(float(np.mean([f[key] for f in fold_reports])), 5)

    nf = len(fold_reports)
    summary = {
        "n_folds": nf,
        "mean_policy_avg": _mean("policy_avg"),
        "mean_take_all_avg": _mean("take_all_avg"),
        "mean_beta_avg": _mean("beta_avg"),
        "mean_edge_vs_take_all": _mean("edge_vs_take_all"),
        "mean_edge_vs_beta": _mean("edge_vs_beta"),
        "mean_ann_sharpe": _mean("ann_sharpe"),
        "folds_profitable": sum(1 for f in fold_reports if f["policy_avg"] > 0),
        "folds_beat_take_all": sum(1 for f in fold_reports if f["edge_vs_take_all"] > 0),
        "folds_beat_beta": sum(1 for f in fold_reports if f["edge_vs_beta"] > 0),
    }
    return {"folds": fold_reports, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(prog="cross_validate")
    ap.add_argument("--dataset", required=True, help="path to decisions_tabular.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    args = ap.parse_args()

    rep = cross_validate(args.dataset, folds=args.folds, embargo_days=args.embargo_days,
                         min_coverage=args.min_coverage)
    print("-" * 78)
    print(f"purged walk-forward CV  ({args.dataset})")
    print("-" * 78)
    hdr = f"{'fold':>4}{'train':>7}{'test':>6}{'cov':>7}{'policy':>9}{'takeAll':>9}{'beta':>9}{'vsBeta':>9}{'annSh':>8}{'maxDD':>8}"
    print(hdr)
    for f in rep["folds"]:
        print(f"{f['fold']:>4}{f['train']:>7}{f['test']:>6}{f['coverage']:>7}"
              f"{f['policy_avg']:>9}{f['take_all_avg']:>9}{f['beta_avg']:>9}"
              f"{f['edge_vs_beta']:>9}{f['ann_sharpe']:>8}{f['max_dd']:>8}")
    print("-" * 78)
    s = rep["summary"]
    if not rep["folds"]:
        print(s.get("note", "no folds"))
        return 1
    print(f"mean policy avg/trade : {s['mean_policy_avg']:+.5f}   "
          f"take-all: {s['mean_take_all_avg']:+.5f}   beta: {s['mean_beta_avg']:+.5f}")
    print(f"mean edge vs take-all : {s['mean_edge_vs_take_all']:+.5f}   "
          f"mean edge vs beta: {s['mean_edge_vs_beta']:+.5f}   mean ann Sharpe: {s['mean_ann_sharpe']}")
    print(f"consistency           : profitable {s['folds_profitable']}/{s['n_folds']}   "
          f"beat take-all {s['folds_beat_take_all']}/{s['n_folds']}   "
          f"beat beta {s['folds_beat_beta']}/{s['n_folds']}")
    print("-" * 78)
    # Honest verdict — require it to clear beta in a majority of folds.
    beat_beta = s["folds_beat_beta"]
    if beat_beta >= (s["n_folds"] + 1) // 2 + 1 and s["mean_edge_vs_beta"] > 0:
        verdict = "ROBUST: beats beta in a majority of folds"
    elif s["folds_beat_take_all"] >= (s["n_folds"] + 1) // 2 and s["mean_edge_vs_take_all"] > 0:
        verdict = "PROMISING: beats take-all but not clearly beta (beta-contaminated)"
    else:
        verdict = "NOT ROBUST: edge does not survive walk-forward CV"
    print(f"verdict : {verdict}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
