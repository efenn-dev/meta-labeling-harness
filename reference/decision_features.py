"""Decision-time features — everything knowable AT the entry decision, nothing after.

This is the counterpart to the post-trade judge. The judge grades a *completed*
trade with hindsight (it sees exit price, P&L, MFE/MAE, the counterfactual). A
trader cannot. Every feature here is computed from bars **strictly before** the
fill, so a model trained on them is forced to learn a decision, not to read an
outcome that was handed to it.

The one rule
------------
The backtest fills a signal at the *next bar's open* (see
``archetypes._simulate_long_only``: "signal-on-bar-N → fill-at-bar-N+1-open").
So for a trade whose ``entry_ts`` is the fill bar, the decision was made at the
**close of the previous bar** (the signal bar). Therefore:

    pre = df[df.index < entry_ts]      # the signal bar is pre.iloc[-1]

is the entire visible world at decision time. ``feature_vector`` and
``signal_snapshot`` consume only ``pre``; ``news_before`` only reads headlines
dated at or before the signal bar. If a value needs a bar dated >= ``entry_ts``
to compute, it is a *label*, not a feature, and it does not live in this module.

All features are deliberately scale-free (returns, ratios, gaps, centred
oscillators) so the model generalises across symbols and price levels.
"""
from __future__ import annotations

import math
from datetime import timedelta

import market_regime

# The price/volatility feature vector. Stable, ordered — the tabular learner and
# the tests both key off this exact list, so additions go at the end.
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
    # market-relative — the features that can separate skill from beta. Need a
    # market reference (SPY) + the universe panel; default-neutral when absent.
    "f_relret_5", "f_relret_21", "f_relret_63", "f_beta_63", "f_resid_ret_21",
    "f_xsec_rank_21",
]


def _safe(x: float) -> float:
    """NaN/inf → 0.0 (a neutral value for these centred, scale-free features)."""
    if x is None or math.isnan(x) or math.isinf(x):
        return 0.0
    return float(x)


def feature_vector(pre) -> dict[str, float]:
    """Scale-free features computed from ``pre`` = bars strictly before the fill.

    ``pre.iloc[-1]`` is the signal bar (decision time). Returns every key in
    :data:`FEATURE_KEYS`; missing/short-history values fall back to 0.0.
    """
    out = {k: 0.0 for k in FEATURE_KEYS}
    if pre is None or len(pre) < 2:
        return out

    close = pre["close"].astype(float)
    high = pre["high"].astype(float)
    low = pre["low"].astype(float)
    openp = pre["open"].astype(float) if "open" in pre.columns else close
    vol = pre["volume"].astype(float) if "volume" in pre.columns else None

    c = float(close.iloc[-1])
    if c <= 0:
        return out

    # ── trailing returns ──────────────────────────────────────────────────
    def ret(h: int) -> float:
        if len(close) > h and float(close.iloc[-1 - h]) > 0:
            return c / float(close.iloc[-1 - h]) - 1.0
        return 0.0
    out["f_ret_1"] = _safe(ret(1))
    out["f_ret_5"] = _safe(ret(5))
    out["f_ret_10"] = _safe(ret(10))
    out["f_ret_21"] = _safe(ret(21))
    out["f_ret_63"] = _safe(ret(63))

    # ── moving averages ───────────────────────────────────────────────────
    def sma(p: int) -> float:
        return float(close.iloc[-p:].mean()) if len(close) >= p else math.nan
    s20, s50, s200 = sma(20), sma(50), sma(200)
    if not math.isnan(s20) and s20 > 0:
        out["f_sma20_gap"] = _safe(c / s20 - 1.0)
    if not math.isnan(s50) and s50 > 0:
        out["f_sma50_gap"] = _safe(c / s50 - 1.0)
        if not math.isnan(s20):
            out["f_sma_fast_slow"] = _safe(s20 / s50 - 1.0)
    if not math.isnan(s200) and s200 > 0:
        out["f_sma200_gap"] = _safe(c / s200 - 1.0)
    if len(close) >= 26:
        s20_prev = float(close.iloc[-26:-6].mean())
        if s20_prev > 0:
            out["f_sma20_slope"] = _safe(s20 / s20_prev - 1.0)

    # ── oscillators (reuse the regime classifier's pure-Python indicators) ─
    closes_l = close.tolist()
    highs_l = high.tolist()
    lows_l = low.tolist()
    rsi = market_regime._rsi(closes_l, 14)[-1]
    if not math.isnan(rsi):
        out["f_rsi_14"] = _safe((rsi - 50.0) / 50.0)        # centre to ~[-1, 1]
    atr = market_regime._atr(highs_l, lows_l, closes_l, 14)[-1]
    if not math.isnan(atr):
        out["f_atr_pct"] = _safe(atr / c)

    # MACD histogram, normalised by price.
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=9, adjust=False).mean()
    out["f_macd_hist"] = _safe(float(macd.iloc[-1] - sig.iloc[-1]) / c)

    # ── Donchian position in the trailing 20-bar range ────────────────────
    if len(close) >= 20:
        hi20 = float(high.iloc[-20:].max())
        lo20 = float(low.iloc[-20:].min())
        span = hi20 - lo20
        if span > 0:
            out["f_donchian_pos"] = _safe((c - lo20) / span)   # 0 = at low, 1 = at high
        if hi20 > 0:
            out["f_dist_high_20"] = _safe(c / hi20 - 1.0)
        if lo20 > 0:
            out["f_dist_low_20"] = _safe(c / lo20 - 1.0)

    # ── volatility ────────────────────────────────────────────────────────
    rets = close.pct_change()
    if len(rets.dropna()) >= 20:
        v20 = float(rets.iloc[-20:].std())
        out["f_vol_20"] = _safe(v20 * math.sqrt(252.0))
        if len(rets.dropna()) >= 5 and v20 > 0:
            v5 = float(rets.iloc[-5:].std())
            out["f_vol_ratio"] = _safe(v5 / v20)
    if len(close) >= 20:
        lo20c = float(close.iloc[-20:].min())
        if lo20c > 0:
            out["f_range_20d"] = _safe((float(close.iloc[-20:].max()) - lo20c) / lo20c)

    # ── microstructure ────────────────────────────────────────────────────
    if len(close) >= 2 and float(close.iloc[-2]) > 0:
        out["f_gap_open"] = _safe(float(openp.iloc[-1]) / float(close.iloc[-2]) - 1.0)
    if vol is not None and len(vol) >= 20:
        m = float(vol.iloc[-20:].mean())
        s = float(vol.iloc[-20:].std())
        if s > 0:
            out["f_vol_z_20"] = _safe((float(vol.iloc[-1]) - m) / s)
        if m > 0:
            out["f_dvol_ratio"] = _safe(float(vol.iloc[-1]) / m)

    return out


def market_features(pre, market_pre) -> dict[str, float]:
    """Market-relative features — the ones that can tell *skill* from *beta*.

    All causal (only bars strictly before the fill, on both the symbol and the
    market reference):
    * ``f_relret_{5,21,63}`` — the symbol's trailing return minus the market's
      over the same horizon (excess return).
    * ``f_beta_63`` — rolling beta of the symbol to the market over the last 63
      daily returns.
    * ``f_resid_ret_21`` — the 21-bar return left over after removing
      ``beta * market_return`` (idiosyncratic / alpha-ish move).

    Defaults to neutral (excess 0, beta 1) when there is no market reference —
    so a single-symbol build still works, it just carries no market signal.
    """
    import pandas as pd
    out = {"f_relret_5": 0.0, "f_relret_21": 0.0, "f_relret_63": 0.0,
           "f_beta_63": 1.0, "f_resid_ret_21": 0.0}
    if pre is None or market_pre is None or len(pre) < 22 or len(market_pre) < 22:
        return out
    sc = pre["close"].astype(float)
    mc = market_pre["close"].astype(float)

    def ret(s, h: int) -> float:
        return float(s.iloc[-1] / s.iloc[-1 - h] - 1.0) if len(s) > h and float(s.iloc[-1 - h]) > 0 else 0.0

    for h in (5, 21, 63):
        out[f"f_relret_{h}"] = _safe(ret(sc, h) - ret(mc, h))

    # Rolling beta over the last ~63 overlapping daily returns.
    pair = pd.concat([sc.pct_change(), mc.pct_change()], axis=1, keys=["s", "m"]).dropna().iloc[-63:]
    if len(pair) >= 20 and float(pair["m"].var()) > 0:
        beta = float(pair["s"].cov(pair["m"]) / pair["m"].var())
        out["f_beta_63"] = _safe(beta)
        out["f_resid_ret_21"] = _safe(ret(sc, 21) - beta * ret(mc, 21))
    return out


def signal_snapshot(pre, archetype: str) -> dict:
    """Human-readable values of the rule that fired, for the LLM evidence packet.

    Pre-entry only; purely descriptive (the tabular learner ignores it). Keeps
    the packet grounded so a fine-tuned judge can cite real numbers.
    """
    if pre is None or len(pre) < 2:
        return {}
    close = pre["close"].astype(float)
    c = float(close.iloc[-1])
    snap: dict = {"signal_close": round(c, 4)}
    base = (archetype or "").removeprefix("reverse_")
    try:
        if base in ("sma_crossover", "macd_signal"):
            s20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else math.nan
            s50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else math.nan
            snap["sma20"] = round(s20, 4)
            snap["sma50"] = round(s50, 4)
            if not (math.isnan(s20) or math.isnan(s50)) and s50:
                snap["fast_slow_gap_pct"] = round((s20 / s50 - 1.0) * 100, 3)
        elif base == "rsi_mean_reversion":
            snap["rsi14"] = round(market_regime._rsi(close.tolist(), 14)[-1], 2)
        elif base in ("donchian_breakout", "bollinger_breakout"):
            if len(close) >= 20:
                hi = float(pre["high"].astype(float).iloc[-20:].max())
                snap["dist_to_20d_high_pct"] = round((c / hi - 1.0) * 100, 3) if hi else None
        elif base == "ts_momentum":
            snap["ret_63d_pct"] = round((c / float(close.iloc[-64]) - 1.0) * 100, 3) \
                if len(close) >= 64 and float(close.iloc[-64]) > 0 else None
    except Exception:
        pass
    return snap


def news_before(symbol: str, decision_ts: str, cache: dict[str, list],
                lookback_days: int = 5, reader=None) -> tuple[list, float, int]:
    """Compact headlines in ``[decision_ts - lookback_days, decision_ts]`` and net
    signed sentiment. Strictly at/before the decision — never reads into the
    holding window (the leak the post-trade builder has).

    ``reader`` defaults to ``pipeline_common.read_news_items`` but is injectable
    for tests. Returns ``(items, net_sentiment, count)``.
    """
    import pandas as pd
    if reader is None:
        from pipeline_common import read_news_items as reader  # noqa: N806
    items = cache.get(symbol)
    if items is None:
        try:
            items = reader(symbol, months=6)
        except Exception:
            items = []
        cache[symbol] = items
    if not items:
        return [], 0.0, 0

    try:
        hi = pd.Timestamp(decision_ts)
        hi = hi.tz_localize(None) if hi.tzinfo else hi
        lo = hi - timedelta(days=lookback_days)
    except Exception:
        return [], 0.0, 0

    compact, sentiments = [], []
    for it in items:
        ts = it.get("ts")
        if not ts:
            continue
        try:
            t = pd.Timestamp(ts)
            t = t.tz_localize(None) if t.tzinfo else t
        except Exception:
            continue
        if not (lo <= t <= hi):          # the cutoff is the decision, not the exit
            continue
        label = it.get("label") or {}
        sent = str(label.get("sentiment") or "neutral").lower()
        conf = float(label.get("confidence") or 0.0)
        signed = conf if sent == "bullish" else (-conf if sent == "bearish" else 0.0)
        sentiments.append(signed)
        if len(compact) < 6:
            compact.append({"headline": (it.get("headline") or "")[:160],
                            "sentiment": sent, "confidence": round(conf, 2)})
    net = sum(sentiments) / len(sentiments) if sentiments else 0.0
    return compact, net, len(sentiments)
