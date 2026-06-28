"""Feature + context schema: the contract between the corpus and the harness.

These key lists are the only thing the evaluation harness (``meta_labeler`` /
``eval_policy`` / ``cross_validate``) needs from the corpus builder. They are
inlined here so the harness runs standalone — without importing the private
backtest engine that produces the real corpus (see ``reference/``). A row in
``decisions_tabular.jsonl`` carries every key below plus the ``label_*`` fields.
"""
from __future__ import annotations

# Scale-free, strictly pre-entry price/vol features, then market-relative
# features (the ones that can separate skill from beta).
FEATURE_KEYS: list[str] = [
    # trailing returns to the signal-bar close
    "f_ret_1", "f_ret_5", "f_ret_10", "f_ret_21", "f_ret_63",
    # moving-average structure
    "f_sma20_gap", "f_sma50_gap", "f_sma200_gap", "f_sma_fast_slow", "f_sma20_slope",
    # momentum / oscillators (centred)
    "f_rsi_14", "f_macd_hist", "f_donchian_pos", "f_dist_high_20", "f_dist_low_20",
    # volatility
    "f_atr_pct", "f_vol_20", "f_vol_ratio", "f_range_20d",
    # microstructure
    "f_gap_open", "f_vol_z_20", "f_dvol_ratio",
    # market-relative
    "f_relret_5", "f_relret_21", "f_relret_63", "f_beta_63", "f_resid_ret_21",
    "f_xsec_rank_21",
]

# Numeric context columns (beyond the price feature vector).
CONTEXT_NUMERIC_KEYS: list[str] = [
    "is_reverse", "regime_confidence", "regime_score",
    "news_sentiment", "news_count", "market_regime_score",
]

# Categorical context columns the learner one-hot encodes.
CATEGORICAL_KEYS: list[str] = [
    "archetype", "base_archetype", "side", "style", "regime", "market_regime",
]
