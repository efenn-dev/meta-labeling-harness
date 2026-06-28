"""Tests for the decision-time meta-labeling dataset.

The whole value of this corpus is that it does NOT leak the future into the
features. These tests assert that guarantee directly:

* no forward field (exit/pnl/mfe/mae/counterfactual) appears in any model-visible
  packet or feature column;
* the feature vector is causal — appending future bars never changes a past
  decision's features;
* news is cut off at the decision, never the exit;
* the walk-forward split is purged + embargoed (no train trade's future reaches
  into the test window);
* labels are the realized return net of the configured cost.

Run:
    cd engine && python -m unittest tests.test_decision_dataset_builder -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import decision_dataset_builder as ddb  # noqa: E402
import decision_features as feats  # noqa: E402

# Substrings that must never appear in anything the model is allowed to see.
FORBIDDEN = ["exit_px", "exit_ts", "pnl", "mfe", "mae",
             "counterfactual", "reverse_pnl", "fwd_ret", "label_"]

FUTURE_SENTINEL = "FUTURE_LEAK_SENTINEL_HEADLINE"


def _bars(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 360
    base = np.concatenate([np.linspace(100, 175, 150),
                           150 + 9 * np.sin(np.linspace(0, 11, 120)),
                           np.linspace(150, 110, 90)])
    close = base + rng.normal(0, 1.0, n)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    idx.name = "timestamp"
    return pd.DataFrame({"open": close, "high": close * 1.012, "low": close * 0.988,
                         "close": close, "volume": 1e6 + rng.normal(0, 1e5, n)}, index=idx)


def _fake_news(symbol: str, months: int = 6) -> list[dict]:
    # One headline far in the future (after every exit) and one at the very start.
    return [
        {"ts": "2025-06-01", "headline": FUTURE_SENTINEL,
         "label": {"sentiment": "bullish", "confidence": 0.9}},
        {"ts": "2022-01-04", "headline": "early benign update",
         "label": {"sentiment": "neutral", "confidence": 0.1}},
    ]


class TestDecisionDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bars = {"AAA": _bars(1), "BBB": _bars(2)}
        self.manifest = ddb.build_decision_dataset(
            universe=["AAA", "BBB"], timeframe="1Day",
            out_dir=self.tmp,
            loader=lambda sym, tf: self.bars.get(sym),
            news_reader=_fake_news,
        )
        self.tab = [json.loads(l) for l in
                    (self.tmp / "decisions_tabular.jsonl").read_text().splitlines() if l.strip()]
        self.sft = [json.loads(l) for l in
                    (self.tmp / "decisions_sft.jsonl").read_text().splitlines() if l.strip()]

    def test_built_records(self):
        self.assertGreater(self.manifest["n_records"], 10)
        self.assertEqual(len(self.tab), self.manifest["n_records"])
        self.assertEqual(len(self.sft), self.manifest["n_records"])

    def test_no_forward_fields_in_packet(self):
        """The system+user messages (what the model sees) carry no outcome."""
        for rec in self.sft:
            visible = " ".join(m["content"] for m in rec["messages"] if m["role"] != "assistant")
            low = visible.lower()
            for bad in FORBIDDEN:
                self.assertNotIn(bad, low, f"forward field '{bad}' leaked into packet")

    def test_feature_columns_have_no_label(self):
        feature_cols = set(feats.FEATURE_KEYS) | set(ddb.CONTEXT_NUMERIC_KEYS) | set(ddb.CATEGORICAL_KEYS)
        for k in feature_cols:
            self.assertFalse(k.startswith("label_"), k)
            for bad in ("exit", "pnl", "mfe", "mae", "fwd"):
                self.assertNotIn(bad, k.lower())

    def test_news_cutoff_no_future_leak(self):
        """A future-dated headline must never appear in any record."""
        blob = (self.tmp / "decisions_sft.jsonl").read_text()
        self.assertNotIn(FUTURE_SENTINEL, blob)

    def test_feature_vector_is_causal(self):
        """Appending bars AFTER the decision must not change the features."""
        df = self.bars["AAA"]
        entry_ts = str(df.index[120])
        pre = ddb._pre_bars(df, entry_ts)
        f1 = feats.feature_vector(pre)
        # Mutilate the future: scramble every bar at/after entry, recompute.
        df2 = df.copy()
        df2.iloc[120:] = df2.iloc[120:] * 3.0 + 50.0
        f2 = feats.feature_vector(ddb._pre_bars(df2, entry_ts))
        self.assertEqual(f1, f2)

    def test_market_features_are_causal(self):
        """Market-relative features must not change when future bars are altered."""
        df = self.bars["AAA"]
        mkt = self.bars["BBB"]
        entry_ts = str(df.index[140])
        m1 = feats.market_features(ddb._pre_bars(df, entry_ts), ddb._pre_bars(mkt, entry_ts))
        df2, mkt2 = df.copy(), mkt.copy()
        df2.iloc[140:] = df2.iloc[140:] * 2.5 + 30.0
        mkt2.iloc[140:] = mkt2.iloc[140:] * 0.3
        m2 = feats.market_features(ddb._pre_bars(df2, entry_ts), ddb._pre_bars(mkt2, entry_ts))
        self.assertEqual(m1, m2)

    def test_news_before_excludes_future(self):
        items, sent, n = feats.news_before("AAA", "2022-01-05", {},
                                           lookback_days=5, reader=_fake_news)
        heads = [it["headline"] for it in items]
        self.assertNotIn(FUTURE_SENTINEL, heads)

    def test_purged_split_no_lookahead(self):
        """No train trade's holding window may reach the test window's start."""
        train = [r for r in self.tab if r["split"] == "train"]
        test = [r for r in self.tab if r["split"] == "test"]
        if not test or not train:
            self.skipTest("degenerate split on synthetic data")
        test_start = min(r["entry_ts"] for r in test)
        for r in train:
            self.assertLess(r["exit_ts"], test_start,
                            "a train trade's exit reaches into the test window (leak)")

    def test_label_is_net_of_cost(self):
        cost = self.manifest["costs"]["extra_cost_pct"]
        for r in self.tab[:50]:
            net = round(r["label_fwd_ret_gross"] - cost, 5)
            self.assertAlmostEqual(r["label_fwd_ret_net"], net, places=5)
            expect = "take" if net > 0 else "skip"
            self.assertEqual(r["label_decision"], expect)

    def test_meta_labeler_trains_and_decides(self):
        import meta_labeler
        ml = meta_labeler.train(self.tmp / "decisions_tabular.jsonl", min_coverage=0.05)
        self.assertTrue(0.0 <= ml.threshold <= 1.0)
        decisions = ml.decide(self.tab[:20])
        self.assertEqual(len(decisions), 20)
        self.assertTrue(all(d["decision"] in ("take", "skip") for d in decisions))


if __name__ == "__main__":
    unittest.main()
