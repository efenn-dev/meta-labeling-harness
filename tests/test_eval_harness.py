"""Standalone tests for the evaluation harness — no trading engine, no data.

These run on synthetic decision rows and assert the two properties that make the
harness trustworthy:

1. It never reads a forward/label field as a feature (leak-free by construction).
2. It rewards a real edge but refuses to manufacture one from pure noise.
"""
from __future__ import annotations

import numpy as np

import eval_policy
import meta_labeler
import schema
import synthetic


def _split(rows, name):
    return [r for r in rows if r["split"] == name]


def test_feature_schema_is_leak_free():
    """No model-visible key may be a forward/outcome field."""
    for k in schema.FEATURE_KEYS + schema.CONTEXT_NUMERIC_KEYS + schema.CATEGORICAL_KEYS:
        assert not k.startswith("label_")
        for bad in ("exit", "pnl", "mfe", "mae", "fwd"):
            assert bad not in k.lower(), k


def test_decisions_are_valid_and_cover_partially():
    rows = synthetic.make_rows(n=1200, seed=0, signal=True)
    ml = meta_labeler.fit_model(_split(rows, "train"), _split(rows, "val"))
    test = _split(rows, "test")
    decisions = ml.decide(test)
    assert len(decisions) == len(test)
    assert all(d["decision"] in ("take", "skip") for d in decisions)
    cov = np.mean([d["decision"] == "take" for d in decisions])
    assert 0.0 < cov <= 1.0


def test_threshold_respects_min_coverage():
    rows = synthetic.make_rows(n=1200, seed=2, signal=True)
    ml = meta_labeler.fit_model(_split(rows, "train"), _split(rows, "val"),
                                min_coverage=0.10)
    assert ml.val_stats.get("coverage", 0.0) >= 0.10


def test_rewards_a_real_edge():
    """With a planted edge, the taken set must beat taking everything."""
    rows = synthetic.make_rows(n=1500, seed=0, signal=True)
    ml = meta_labeler.fit_model(_split(rows, "train"), _split(rows, "val"))
    test = _split(rows, "test")
    take = np.array([d["decision"] == "take" for d in ml.decide(test)], dtype=bool)
    m = eval_policy._metrics(test, take)
    b = eval_policy._metrics(test, np.ones(len(test), dtype=bool))
    assert m["avg_ret_net"] > b["avg_ret_net"]


def test_does_not_manufacture_edge_from_noise():
    """No real edge ⇒ the policy must not beat beta under purged walk-forward CV.

    A single split can flatter on noise (that's the whole lesson); the honest
    test is the cross-validation, which should refuse to call it robust.
    """
    import json
    import tempfile
    from pathlib import Path

    import cross_validate

    rows = synthetic.make_rows(n=1500, seed=1, signal=False)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "decisions_tabular.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        rep = cross_validate.cross_validate(str(p), folds=5)
    s = rep["summary"]
    assert s["folds_beat_beta"] <= 2          # not robust across folds
    assert s["mean_edge_vs_beta"] <= 0.01     # no fabricated alpha over beta
