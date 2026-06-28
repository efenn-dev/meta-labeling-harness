# Decision-time meta-labeling: an honest take/skip evaluation harness

[![tests](https://github.com/efenn-dev/meta-labeling-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/efenn-dev/meta-labeling-harness/actions/workflows/ci.yml)

A small, dependency-light harness for the question most trading-ML code quietly
dodges: *does this signal actually have edge — net of costs, out of sample, and
over and above just being long the market?* It learns a **take/skip** decision on
the signals a strategy fires (meta-labeling), and — more importantly — it
**scores that decision as a trader, not a classifier**, with the leakage controls
and benchmarks that decide whether a result is real.

The headline finding from running it on five years of real cached bars: a model
that looks great on a single held-out split (annualized Sharpe ≈ 1.8) **fails
purged walk-forward cross-validation** — it beats a take-everything baseline but
not buy-and-hold beta. The harness says so, plainly. That honesty is the point.
Full write-up: **[RESEARCH_NOTE.md](RESEARCH_NOTE.md)**.

## Why it's different

- **Leak-free by construction.** Features are strictly pre-entry; the forward
  return is the *label*, never an input. The corpus uses a **purged + embargoed**
  walk-forward split so a trade's future can't leak across the train/test wall.
- **Profit objective, not accuracy.** The decision threshold is tuned to maximize
  net P&L of the taken set (subject to a coverage floor) — a 99%-accurate model
  that only ever says "skip" makes no money.
- **Benchmarked against beta.** Every result is compared to take-all *and* to
  buy-and-hold over the same windows, because a long-biased filter looks
  profitable just by being in a rising market.
- **A verdict that can say "no."** The eval distinguishes *profitable*,
  *reduces-loss-but-still-negative*, and *no-edge*, and the cross-validator reports
  how many folds actually beat beta.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py          # synthetic corpus with a real edge — harness rewards it
python demo.py --noise  # no edge — single split flatters, walk-forward refuses it
```

`demo.py --noise` reproduces the core lesson in ten seconds: the single split can
look profitable on pure noise, but purged walk-forward CV beats beta in 0/5 folds.

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## What's here

| Path | Role |
|---|---|
| `meta_labeler.py` | Take/skip model; threshold tuned for net P&L. sklearn `HistGradientBoosting` if available, else a dependency-free numpy logistic regression. CPU-only. |
| `eval_policy.py` | Scores a policy **as a trader**: coverage, hit rate, net return, per-trade/annualized Sharpe, drawdown, per-regime/archetype breakdown, honest verdict. |
| `cross_validate.py` | Purged walk-forward CV with a beta benchmark — the test that catches single-split luck. |
| `schema.py` | The feature/context contract between corpus and harness. |
| `demo.py`, `synthetic.py` | Self-contained synthetic-data demo and fixtures. |
| `tests/` | Standalone tests: leak-free schema, rewards real edge, refuses noise. |
| `reference/` | The leak-free corpus **builder** (`decision_dataset_builder.py`, `decision_features.py`). Read-only — it depends on a private backtest engine and is included to document the methodology. |

## Note on scope

This is extracted from a larger private trading project. The **evaluation harness**
(repo root) is fully self-contained and runnable. The **corpus builder**
(`reference/`) is included for review of the leak-free construction but needs the
private engine to run; the synthetic demo stands in for it end to end.

## License

MIT — see [LICENSE](LICENSE).
