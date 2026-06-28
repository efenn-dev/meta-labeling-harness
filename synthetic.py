"""Synthetic decision-row generator for the standalone demo + tests.

Emits rows in the same schema as the real corpus (see ``schema.py``) WITHOUT the
trading engine: features ~ N(0, 1), a *known* linear edge in a few features, a
bull-ish "underlying" return so take-all behaves like beta, and a per-trade cost
baked into the label. This lets the evaluation harness be exercised end to end
with zero external data, models, or API keys.

It is a test fixture, not a market simulator: the point is to show the harness
*correctly rewards a real edge and refuses to manufacture one from noise*, which
is the same property the leak-free pipeline guarantees on real data.
"""
from __future__ import annotations

import numpy as np

import schema

_ARCHETYPES = ["sma_crossover", "rsi_mean_reversion", "bollinger_breakout",
               "macd_signal", "donchian_breakout", "ts_momentum"]
_REGIMES = ["bullish", "bearish", "chop"]


def make_rows(n: int = 1500, seed: int = 0, signal: bool = True,
              cost: float = 0.001) -> list[dict]:
    """Return ``n`` decision rows with a time-ordered train/val/test split.

    ``signal=True`` plants a learnable linear edge in three features so the
    meta-labeler can separate good signals from bad. ``signal=False`` makes the
    forward return pure noise minus cost — a no-edge world the harness should
    refuse to call profitable.
    """
    rng = np.random.default_rng(seed)
    feats = schema.FEATURE_KEYS
    w = np.zeros(len(feats))
    if signal:
        for i in (3, 10, 22):              # f_ret_21, f_rsi_14, f_relret_5
            w[i] = rng.choice([-1.0, 1.0]) * 0.013

    start = np.datetime64("2021-01-04")
    rows: list[dict] = []
    for t in range(n):
        x = rng.normal(0, 1, len(feats))
        edge = float(x @ w)
        underlying = float(rng.normal(0.004, 0.05))    # bull-ish beta
        net = edge + 0.3 * underlying + float(rng.normal(0, 0.03)) - cost

        row = {k: float(x[i]) for i, k in enumerate(feats)}
        for k in schema.CONTEXT_NUMERIC_KEYS:
            row[k] = float(rng.integers(0, 2)) if k == "is_reverse" else float(rng.normal(0, 1))
        arch = _ARCHETYPES[rng.integers(0, len(_ARCHETYPES))]
        regime = _REGIMES[rng.integers(0, len(_REGIMES))]
        entry = start + np.timedelta64(t, "D")
        hold = int(rng.integers(2, 12))
        row.update({
            "archetype": arch, "base_archetype": arch, "side": "long",
            "style": "trend", "regime": regime, "market_regime": regime,
            "label_decision": "take" if net > 0 else "skip",
            "label_fwd_ret_net": round(net, 5),
            "label_underlying_ret": round(underlying, 5),
            "entry_ts": str(entry),
            "exit_ts": str(entry + np.timedelta64(hold, "D")),
            "holding_bars": hold,
            "id": f"syn-{t}",
        })
        rows.append(row)

    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    for i, r in enumerate(rows):
        r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return rows
