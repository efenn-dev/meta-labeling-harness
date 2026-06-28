"""Decision-time dataset builder — the meta-labeling corpus.

This is the honest sibling of ``dataset_builder.py``. That one builds a
*post-trade judge*: it hands the model the realized outcome (exit price, P&L,
MFE/MAE, the counterfactual) and asks it to explain the grade — which trains a
narrator, not a trader, and makes the eval look great because the answer is in
the inputs.

This builder instead asks the **decision-time** question, framed as
*meta-labeling* (López de Prado): the archetype is a primary model that fires a
signal; the learner is a secondary model that decides **take / skip** on that
signal using only what is knowable at the moment of entry.

For every signal the backtest fired we emit one record:

* ``context`` — pre-entry features only (regime, indicators, news-so-far, the
  signal snapshot, the proposed fill). Built from ``df[df.index < entry_ts]``.
  Contains **no** forward field. This is what the model sees.
* ``label``   — the realized forward outcome of taking the signal, **net of
  costs**: ``fwd_ret_net``, the take/skip decision, a 3-class rating, and the
  forward adverse excursion. Built from ``df[df.index >= entry_ts]`` (it is the
  backtest's realized trade). This is the answer; it never enters ``context``.

Leakage controls
----------------
* Temporal wall at ``entry_ts`` (the fill bar); the signal bar is the last
  visible bar. ``decision_features`` enforces this on the feature side.
* News cutoff at the signal bar, not the exit (``decision_features.news_before``).
* **Purged + embargoed** walk-forward split: a train trade whose holding window
  spills past a split boundary is dropped (purge), plus an embargo gap around
  each boundary kills boundary serial-correlation leakage.

Outputs (under ``<userData>/datasets/``)
----------------------------------------
* ``decisions_tabular.jsonl`` — one flat row per signal: features + context +
  ``label_*`` + meta. Feeds :mod:`meta_labeler` (a CPU gradient-boosted model).
* ``decisions_sft.jsonl``     — chat view (decision prompt → take/skip verdict)
  for fine-tuning a local judge on the same task.
* ``decision_manifest.json``  — counts, date ranges, costs, split sizes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import archetypes
import dataset_builder as db          # reuse _run_variant / _window / _excursion
import decision_features as feats
import market_regime
import reverse_strategies
from pipeline_common import LOG, load_cached_bars, user_data_dir

SCHEMA_VERSION = 1

DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META"]
STOCK_ARCHETYPES = ["sma_crossover", "rsi_mean_reversion", "bollinger_breakout",
                    "macd_signal", "donchian_breakout", "ts_momentum"]

INITIAL_EQUITY = 10000.0
COMMISSION_PER_SHARE = 0.0
# Slippage the simulator already applies per side (10 bps round trip).
SLIPPAGE_PCT = 0.0005
# Extra round-trip haircut applied in the LABEL only — fees, market impact, and
# the fact that you cannot always fill at the modelled price. Deliberately
# conservative: a signal must clear this to be labelled "take". Tune per venue.
DEFAULT_EXTRA_COST_PCT = 0.0010

# Categorical context columns the learner one-hot encodes.
CATEGORICAL_KEYS = ["archetype", "base_archetype", "side", "style", "regime", "market_regime"]
# Numeric context columns (beyond the price feature vector) the learner uses.
CONTEXT_NUMERIC_KEYS = ["is_reverse", "regime_confidence", "regime_score",
                        "news_sentiment", "news_count", "market_regime_score"]
# Anything the learner must NEVER read as a feature.
LABEL_PREFIX = "label_"
META_KEYS = ["id", "symbol", "split", "entry_ts", "exit_ts", "holding_bars", "exit_reason"]

DECISION_SYSTEM = (
    "You are TradeStar's entry decision-maker. A strategy has fired a signal and "
    "proposes a trade. You receive ONLY what is knowable right now, at the moment "
    "of entry — the strategy logic, the market regime, recent indicators, news so "
    "far, and the proposed fill. You do NOT know what happens next.\n\n"
    "Decide whether to TAKE or SKIP this signal. A good trader does not take every "
    "signal — only the ones whose context (regime fit, momentum, risk, news) gives "
    "an edge net of costs. Return only JSON:\n"
    '{"decision":"take"|"skip","confidence":0.0,'
    '"direction":"long"|"short","reasoning":"one or two sentences grounded in the packet"}'
)


# ── label ────────────────────────────────────────────────────────────────────
def make_label(gross_pnl_pct: float, fwd_mae_pct: float, *,
               extra_cost_pct: float, underlying_ret: float = 0.0,
               target_mode: str = "absolute",
               take_band: float = 0.0, rating_band: float = 0.005) -> dict:
    """Forward outcome → meta-label. Built only from post-entry bars.

    ``gross_pnl_pct`` is the backtest's realized return (already net of the
    simulator's slippage); we subtract ``extra_cost_pct`` for the costs the bar
    sim cannot see. ``underlying_ret`` is the buy-and-hold return of the symbol
    over the same window — the **beta benchmark**.

    ``target_mode`` chooses what the learner is taught to chase:
    * ``absolute`` — take when net return clears the hurdle (rides beta).
    * ``alpha``    — take only when the trade BEATS buy-and-hold over the same
      window (``net - underlying``). This optimises *skill over beta*, which is
      what survives a regime change. ``label_fwd_ret_net`` (real P&L) and
      ``label_underlying_ret`` are kept intact for honest accounting either way.
    """
    net = gross_pnl_pct - extra_cost_pct
    alpha = net - underlying_ret
    score = alpha if target_mode == "alpha" else net
    decision = "take" if score > take_band else "skip"
    if net > rating_band:
        rating = "good"
    elif net < -rating_band:
        rating = "bad"
    else:
        rating = "neutral"
    return {
        "label_decision": decision,
        "label_fwd_ret_net": round(net, 5),
        "label_fwd_ret_gross": round(gross_pnl_pct, 5),
        "label_fwd_mae_pct": round(fwd_mae_pct, 5),
        "label_underlying_ret": round(underlying_ret, 5),
        "label_alpha": round(alpha, 5),
        "label_rating": rating,
    }


# ── pre-entry context ─────────────────────────────────────────────────────────
def _open_at(df, ts: str) -> float | None:
    w = db._window(df, ts, ts)
    if w.empty:
        return None
    return float(w["open"].iloc[0])


def _pre_bars(df, entry_ts: str):
    """Bars strictly before the fill — the entire visible world at decision time."""
    import pandas as pd
    try:
        t0 = pd.Timestamp(entry_ts)
    except Exception:
        return df.iloc[0:0]
    idx = df.index
    if getattr(idx, "tz", None) is not None and t0.tzinfo is None:
        t0 = t0.tz_localize(idx.tz)
    elif getattr(idx, "tz", None) is None and t0.tzinfo is not None:
        t0 = t0.tz_localize(None)
    return df[idx < t0]


def _lookup_xsec(frame, ts, symbol: str) -> float:
    """Cross-sectional return percentile for ``symbol`` at the decision date.
    Causal: the rank at a date depends only on that date's values."""
    if frame is None:
        return 0.5
    try:
        import pandas as pd
        v = frame.at[ts, symbol]
        return 0.5 if pd.isna(v) else float(v)
    except Exception:
        return 0.5


def build_context(t, *, symbol, archetype, base_archetype, is_reverse, df,
                  news_cache, reader=None, market_df=None, xsec_frame=None) -> dict | None:
    """Assemble the pre-entry context dict for one fired signal. Returns None if
    there is not enough pre-history to decide.

    ``market_df`` is the market reference (SPY) for market-relative features;
    ``xsec_frame`` is the per-date cross-sectional return-percentile panel. Both
    are optional — absent → neutral market signal.
    """
    pre = _pre_bars(df, t.entry_ts)
    if len(pre) < market_regime.MIN_BARS:
        return None
    decision_ts = str(pre.index[-1])                      # the signal bar = decision time

    regime = market_regime.classify(pre.tail(250)).to_dict()
    fvec = feats.feature_vector(pre)
    snap = feats.signal_snapshot(pre, archetype)
    news, sent, n_news = feats.news_before(symbol, decision_ts, news_cache, reader=reader)
    fill = _open_at(df, t.entry_ts)

    # Market-relative context (causal: market bars strictly before the fill).
    market_pre = _pre_bars(market_df, t.entry_ts) if market_df is not None else None
    mfeat = feats.market_features(pre, market_pre)
    if market_pre is not None and len(market_pre) >= market_regime.MIN_BARS:
        mreg = market_regime.classify(market_pre.tail(250)).to_dict()
        market_regime_label = str(mreg.get("regime", "unknown"))
        market_regime_score = float(mreg.get("score", 0)) / 100.0
    else:
        market_regime_label, market_regime_score = "unknown", 0.0

    ctx = {
        "id": f"{symbol}-{archetype}-{t.entry_ts}",
        "symbol": symbol,
        "archetype": archetype,
        "base_archetype": base_archetype,
        "is_reverse": 1 if is_reverse else 0,
        "side": t.side,
        "style": archetypes_style(archetype),
        "decision_ts": decision_ts,
        "entry_ts": t.entry_ts,
        "proposed_fill_px": round(fill, 4) if fill else round(float(t.entry_px), 4),
        "regime": str(regime.get("regime", "unknown")),
        "regime_confidence": float(regime.get("confidence", 0.0)),
        "regime_score": float(regime.get("score", 0)) / 100.0,
        "market_regime": market_regime_label,
        "market_regime_score": market_regime_score,
        "news_sentiment": round(float(sent), 3),
        "news_count": int(n_news),
        "signal": snap,
        "news_items": news,
        "regime_detail": regime,
    }
    ctx.update(fvec)
    ctx.update(mfeat)
    ctx["f_xsec_rank_21"] = _lookup_xsec(xsec_frame, pre.index[-1], symbol)
    return ctx


def archetypes_style(archetype: str) -> str:
    """Trend / mean_reversion / vol — reuse the judge's mapping."""
    import decision_layer
    return decision_layer._style_of(archetype)


# ── record views ──────────────────────────────────────────────────────────────
def _to_packet(ctx: dict) -> dict:
    """The pre-entry evidence packet shown to the LLM. No forward fields."""
    return {
        "strategy": {
            "archetype": ctx["archetype"],
            "is_reverse": bool(ctx["is_reverse"]),
            "style": ctx["style"],
            "proposes_side": ctx["side"],
        },
        "proposed_trade": {
            "symbol": ctx["symbol"],
            "side": ctx["side"],
            "decision_ts": ctx["decision_ts"],
            "proposed_fill_px": ctx["proposed_fill_px"],
        },
        "signal": ctx.get("signal", {}),
        "market": ctx.get("regime_detail", {"regime": ctx["regime"]}),
        "market_context": {"spy_regime": ctx.get("market_regime", "unknown"),
                           "spy_regime_score": ctx.get("market_regime_score", 0.0)},
        "features": {k: ctx[k] for k in feats.FEATURE_KEYS},
        "news": ctx.get("news_items", [])[:6],
        "news_sentiment": ctx["news_sentiment"],
    }


def _sft_record(ctx: dict, label: dict, split: str) -> dict:
    user = ("A strategy fired a signal. Decide TAKE or SKIP using only this "
            "pre-entry evidence:\n" + json.dumps(_to_packet(ctx), default=str))
    assistant = {
        "decision": label["label_decision"],
        "direction": ctx["side"],
        "confidence": min(1.0, abs(ctx["regime_confidence"])),
        "reasoning": _decision_rationale(ctx, label),
    }
    return {
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, default=str)},
        ],
        "meta": _meta(ctx, label, split),
    }


def _tabular_record(ctx: dict, label: dict, split: str) -> dict:
    """Flat row for the tabular learner: features + numeric/categorical context +
    label_* + meta. Keys are partitioned so the learner can trivially exclude
    everything that is not a feature."""
    row: dict = {}
    for k in feats.FEATURE_KEYS:
        row[k] = ctx[k]
    for k in CONTEXT_NUMERIC_KEYS:
        row[k] = ctx[k]
    for k in CATEGORICAL_KEYS:
        row[k] = ctx[k]
    row.update(label)
    row.update(_meta(ctx, label, split))
    return row


def _decision_rationale(ctx: dict, label: dict) -> str:
    """A short, pre-entry-grounded justification for the SFT target. References
    only regime fit + momentum + news — never the outcome."""
    style = ctx["style"]
    regime = ctx["regime"]
    trend = ctx.get("f_ret_21", 0.0)
    bits = [f"{style} setup in a {regime} regime"]
    if trend > 0.02:
        bits.append("recent trend is up")
    elif trend < -0.02:
        bits.append("recent trend is down")
    if abs(ctx["news_sentiment"]) >= 0.15:
        bits.append(f"news leans {'bullish' if ctx['news_sentiment'] > 0 else 'bearish'}")
    verb = "Take" if label["label_decision"] == "take" else "Skip"
    return f"{verb}: {', '.join(bits)}."


def _meta(ctx: dict, label: dict, split: str) -> dict:
    return {
        "id": ctx["id"],
        "symbol": ctx["symbol"],
        "split": split,
        "entry_ts": ctx["entry_ts"],
        "exit_ts": ctx.get("_exit_ts", ""),
        "holding_bars": ctx.get("_holding_bars", 0),
        "exit_reason": ctx.get("_exit_reason", ""),
    }


# ── purged + embargoed walk-forward split ─────────────────────────────────────
def purged_splits(entries: list[str], exits: list[str], *,
                  train: float = 0.70, val: float = 0.15,
                  embargo_days: int = 3) -> list[str]:
    """Temporal split with purge + embargo.

    Oldest entries → train, newest → test. A train/val trade whose **label
    window** (``[entry, exit]``) overlaps the next split's start is *purged*
    (label "excluded"), and an ``embargo_days`` gap before each boundary is also
    excluded. This is what stops a trade's future from leaking across the
    train→test wall — the silent re-leak that makes a naive temporal split look
    honest while it isn't.
    """
    import pandas as pd

    def _ts(x):
        try:
            t = pd.Timestamp(x)
            return t.tz_localize(None) if t.tzinfo else t
        except Exception:
            return None

    n = len(entries)
    order = sorted(range(n), key=lambda i: entries[i] or "")
    t_end = int(n * train)
    v_end = int(n * (train + val))

    val_start_ts = _ts(entries[order[t_end]]) if t_end < n else None
    test_start_ts = _ts(entries[order[v_end]]) if v_end < n else None
    embargo = pd.Timedelta(days=embargo_days)

    splits = ["train"] * n
    for rank, i in enumerate(order):
        if rank >= v_end:
            splits[i] = "test"
        elif rank >= t_end:
            splits[i] = "val"

    # Purge + embargo: drop any train/val row whose future reaches into the next
    # split, or that sits inside the embargo window before a boundary.
    for i in range(n):
        if splits[i] == "test":
            continue
        ex = _ts(exits[i]) or _ts(entries[i])
        en = _ts(entries[i])
        boundary = test_start_ts if splits[i] == "val" else val_start_ts
        if boundary is None or ex is None or en is None:
            continue
        if ex >= boundary or en >= (boundary - embargo):
            splits[i] = "excluded"
    return splits


# ── build ──────────────────────────────────────────────────────────────────────
def build_decision_dataset(*, universe: list[str] | None = None,
                           archetype_names: list[str] | None = None,
                           timeframe: str = "1Day",
                           extra_cost_pct: float = DEFAULT_EXTRA_COST_PCT,
                           target_mode: str = "absolute",
                           take_band: float = 0.0,
                           embargo_days: int = 3,
                           max_per_variant: int = 200,
                           max_per_archetype: int = 400,
                           out_dir: Path | None = None,
                           loader: Callable | None = None,
                           news_reader: Callable | None = None,
                           progress: Callable[[int, int], None] | None = None) -> dict:
    """Build the decision-time meta-labeling corpus. Returns the manifest dict.

    ``loader`` / ``news_reader`` are injectable (tests pass synthetic data);
    they default to the real bar cache and news store.
    """
    universe = universe or DEFAULT_UNIVERSE
    bases = archetype_names or list(STOCK_ARCHETYPES)
    out_dir = out_dir or (user_data_dir() / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    load = loader or load_cached_bars
    news_cache: dict[str, list] = {}

    # Preload the universe panel once so market-relative and cross-sectional
    # features (each name vs SPY and vs its peers on the same day) can be built.
    panel: dict = {}
    for symbol in universe:
        d = load(symbol, timeframe)
        if d is None or len(d) < 80:
            LOG.info("decision-ds: no/short bars for %s (%s); skipping", symbol, timeframe)
            continue
        panel[symbol] = d.sort_index()
    if not panel:
        raise RuntimeError(
            "no usable bars in the universe — cache historical bars first "
            "(run stage 01_data_fetch for the universe).")

    import pandas as pd
    market_symbol = "SPY" if "SPY" in panel else next(iter(panel))
    market_df = panel[market_symbol]
    # Cross-sectional 21-bar return percentile per date — skill-vs-peers signal.
    if len(panel) > 1:
        r21 = {s: panel[s]["close"].astype(float).pct_change(21) for s in panel}
        xsec_frame = pd.DataFrame(r21).rank(axis=1, pct=True)
    else:
        xsec_frame = None

    contexts: list[dict] = []
    labels: list[dict] = []

    for symbol, df in panel.items():
        for base in bases:
            for is_reverse in (False, True):
                spec = reverse_strategies.reverse_spec(base) if is_reverse else None
                if is_reverse and (spec is None or spec["kind"] != "stock_short"):
                    continue                       # v1: stock archetypes + their shorts
                try:
                    trades, _eq, _elim = db._run_variant(base, spec, symbol, df)
                except Exception as e:
                    LOG.warning("decision-ds: %s %s%s failed: %s", symbol, base,
                                " (rev)" if is_reverse else "", e)
                    continue
                if max_per_variant and len(trades) > max_per_variant:
                    step = len(trades) / max_per_variant
                    trades = [trades[int(k * step)] for k in range(max_per_variant)]
                archetype = (reverse_strategies.REVERSE_PREFIX + base) if is_reverse else base
                for t in trades:
                    ctx = build_context(t, symbol=symbol, archetype=archetype,
                                        base_archetype=base, is_reverse=is_reverse,
                                        df=df, news_cache=news_cache, reader=news_reader,
                                        market_df=market_df, xsec_frame=xsec_frame)
                    if ctx is None:
                        continue
                    win = db._window(df, t.entry_ts, t.exit_ts)
                    _mfe, mae = db._excursion(df, float(t.entry_px), t.side, t.entry_ts, t.exit_ts)
                    und = 0.0
                    if len(win) >= 2 and float(win["close"].iloc[0]) > 0:
                        und = float(win["close"].iloc[-1]) / float(win["close"].iloc[0]) - 1.0
                    ctx["_exit_ts"] = t.exit_ts
                    ctx["_exit_reason"] = t.reason
                    ctx["_holding_bars"] = len(win)
                    label = make_label(float(t.pnl_pct), float(mae),
                                       extra_cost_pct=extra_cost_pct, underlying_ret=und,
                                       target_mode=target_mode, take_band=take_band)
                    contexts.append(ctx)
                    labels.append(label)

    if not contexts:
        raise RuntimeError(
            "no signals collected — cache historical bars first "
            "(run stage 01_data_fetch for the universe).")

    # Global per-archetype cap so one prolific variant (e.g. a vol-targeted
    # momentum short that fires hundreds of micro-trades across every symbol)
    # can't dominate the corpus and bias the learner. Even temporal thinning
    # keeps a spread across the window rather than just the earliest trades.
    if max_per_archetype:
        by_arch: dict[str, list[int]] = {}
        for i, c in enumerate(contexts):
            by_arch.setdefault(c["archetype"], []).append(i)
        keep: set[int] = set()
        for idxs in by_arch.values():
            if len(idxs) <= max_per_archetype:
                keep.update(idxs)
                continue
            idxs.sort(key=lambda i: contexts[i]["entry_ts"] or "")
            step = len(idxs) / max_per_archetype
            keep.update(idxs[int(k * step)] for k in range(max_per_archetype))
        contexts = [contexts[i] for i in sorted(keep)]
        labels = [labels[i] for i in sorted(keep)]

    splits = purged_splits([c["entry_ts"] for c in contexts],
                           [c["_exit_ts"] for c in contexts],
                           embargo_days=embargo_days)

    tab_path = out_dir / "decisions_tabular.jsonl"
    sft_path = out_dir / "decisions_sft.jsonl"
    counts = {"by_decision": {"take": 0, "skip": 0},
              "by_rating": {"good": 0, "neutral": 0, "bad": 0},
              "by_split": {"train": 0, "val": 0, "test": 0, "excluded": 0},
              "by_archetype": {}}
    date_lo = date_hi = None
    written = 0

    with tab_path.open("w", encoding="utf-8") as tf, sft_path.open("w", encoding="utf-8") as sf:
        for i, (ctx, label, split) in enumerate(zip(contexts, labels, splits)):
            counts["by_split"][split] = counts["by_split"].get(split, 0) + 1
            if split == "excluded":
                continue                            # purged/embargoed — drop from both files
            tf.write(json.dumps(_tabular_record(ctx, label, split), default=str) + "\n")
            sf.write(json.dumps(_sft_record(ctx, label, split), default=str) + "\n")
            written += 1
            counts["by_decision"][label["label_decision"]] += 1
            counts["by_rating"][label["label_rating"]] += 1
            counts["by_archetype"][ctx["archetype"]] = counts["by_archetype"].get(ctx["archetype"], 0) + 1
            date_lo = ctx["entry_ts"] if date_lo is None else min(date_lo, ctx["entry_ts"])
            date_hi = ctx["_exit_ts"] if date_hi is None else max(date_hi, ctx["_exit_ts"] or ctx["entry_ts"])
            if progress:
                progress(i + 1, len(contexts))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "decision_time_meta_label",
        "n_records": written,
        "n_collected": len(contexts),
        "n_excluded": counts["by_split"].get("excluded", 0),
        "universe": universe,
        "archetypes": bases,
        "timeframe": timeframe,
        "costs": {"sim_slippage_pct": SLIPPAGE_PCT, "extra_cost_pct": extra_cost_pct,
                  "take_band": take_band},
        "embargo_days": embargo_days,
        "date_range": {"start": date_lo, "end": date_hi},
        "counts": counts,
        "feature_keys": feats.FEATURE_KEYS,
        "context_numeric_keys": CONTEXT_NUMERIC_KEYS,
        "categorical_keys": CATEGORICAL_KEYS,
        "files": {"tabular": tab_path.name, "sft": sft_path.name},
    }
    (out_dir / "decision_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    LOG.info("decision-ds: wrote %d records (%d excluded by purge/embargo) to %s",
             written, manifest["n_excluded"], out_dir)
    return manifest
