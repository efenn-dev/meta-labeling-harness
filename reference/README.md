# `reference/` — corpus construction (engine-dependent, read-only)

These two modules are the **leak-free corpus builder**. They are included so you
can read exactly how the decision-time dataset is constructed, but they are
**not standalone-runnable**: they import a private backtest engine (`archetypes`,
`dataset_builder`, `market_regime`, `reverse_strategies`, `pipeline_common`) that
is not part of this repository. The runnable harness at the repo root
(`meta_labeler.py`, `eval_policy.py`, `cross_validate.py`) operates on the rows
these produce and needs none of that — see `demo.py`.

- **`decision_dataset_builder.py`** — for every signal a strategy fired, emits one
  record: pre-entry `context` features (built from `df[df.index < entry_ts]`, no
  forward field) + a cost-netted forward-return `label`. Implements the
  **purged + embargoed walk-forward split** that stops a trade's future from
  leaking across the train/test wall.
- **`decision_features.py`** — the strictly causal, scale-free feature vector
  (`FEATURE_KEYS`) and the news cutoff (`news_before`, cut at the signal bar, not
  the exit). Appending future bars never changes a past decision's features.
- **`test_decision_dataset_builder.py`** — asserts the leak-free guarantees
  directly (no forward field in any model-visible packet, causal features, news
  cutoff, purged split, net-of-cost labels). Requires the private engine to run;
  shown here to document what is tested.

See `../RESEARCH_NOTE.md` for the methodology and the real-data result.
