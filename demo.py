"""End-to-end demo of the evaluation harness on synthetic data — no engine, no keys.

Builds a synthetic decision corpus with a known edge, trains the meta-labeler
(threshold tuned for net P&L, not accuracy), scores it as a *trader* against the
take-all baseline on the held-out test split, then runs purged walk-forward CV.

    python demo.py            # signal present
    python demo.py --noise    # no edge — watch the harness refuse to fake one

The numbers here are illustrative (synthetic data). The methodology — and the
real result on five years of cached bars — is in RESEARCH_NOTE.md.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import cross_validate
import eval_policy
import meta_labeler
import synthetic


def _split(rows, name):
    return [r for r in rows if r["split"] == name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", action="store_true", help="generate a no-edge corpus")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    rows = synthetic.make_rows(n=args.n, seed=args.seed, signal=not args.noise)
    label = "NO-EDGE (noise)" if args.noise else "SIGNAL PRESENT"
    print(f"\nSynthetic corpus: {len(rows)} rows, {label}")

    ml = meta_labeler.fit_model(_split(rows, "train"), _split(rows, "val"))
    test = _split(rows, "test")
    take = np.array([d["decision"] == "take" for d in ml.decide(test)], dtype=bool)
    m = eval_policy._metrics(test, take)
    b = eval_policy._metrics(test, np.ones(len(test), dtype=bool))

    print("\nHeld-out test split — meta-labeler vs take-all")
    print(f"  {'metric':<18}{'meta-labeler':>14}{'take-all':>12}")
    for k in ("coverage", "n_taken", "hit_rate", "avg_ret_net", "per_trade_sharpe"):
        print(f"  {k:<18}{str(m.get(k, '-')):>14}{str(b.get(k, '-')):>12}")
    edge = m.get("avg_ret_net", 0.0) - b.get("avg_ret_net", 0.0)
    print(f"  {'edge/trade':<18}{edge:>+14.5f}")

    # Purged walk-forward CV — the honest test.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "decisions_tabular.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        rep = cross_validate.cross_validate(str(p), folds=args.folds)
    s = rep["summary"]
    print(f"\nPurged walk-forward CV ({s.get('n_folds', 0)} folds)")
    print(f"  mean edge vs take-all : {s.get('mean_edge_vs_take_all', 0):+.5f}")
    print(f"  mean edge vs beta     : {s.get('mean_edge_vs_beta', 0):+.5f}")
    print(f"  beats take-all        : {s.get('folds_beat_take_all', 0)}/{s.get('n_folds', 0)} folds")
    print(f"  beats beta            : {s.get('folds_beat_beta', 0)}/{s.get('n_folds', 0)} folds")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
