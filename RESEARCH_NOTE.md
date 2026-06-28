# Decision-time meta-labeling: building the tool that can tell you your idea doesn't work

*A research note on learning take/skip decisions for a trading strategy without
fooling yourself — and on why the honest result here is "promising but not real
edge," not a Sharpe-1.8 win.*

## TL;DR

I built a leak-free pipeline that learns, for every signal a trading strategy
fires, whether to **take** or **skip** it — using only information available at the
moment of entry. On a single held-out test split it looks excellent (hit rate
32% → 63%, annualized Sharpe ≈ 1.8, max drawdown −88% → −16%). Under **purged,
embargoed walk-forward cross-validation** that result collapses: the model beats a
take-everything baseline in 4 of 5 folds but beats **buy-and-hold beta in only 1 of
5**, with a *negative* mean edge over beta. The honest conclusion is that its
apparent profit is mostly market exposure, not skill. The point of the note is the
apparatus: a measurement setup careful enough to catch its own self-deception, and
the discipline to report what it found.

## 1. Motivation: narrator vs. trader

There are two ways to build a "trading AI," and one of them is a trap.

The trap is the **post-trade judge**: hand a model the realized outcome of a trade
(exit price, P&L, max favorable/adverse excursion, the counterfactual) and ask it
to rate the trade good/bad with reasoning. This trains a *narrator*. The evaluation
looks great because the answer is already in the inputs — and the model can never
make money, because at decision time those inputs don't exist yet. It's a critic,
not a trader.

The honest version is the **decision-time** question, framed as *meta-labeling*
(López de Prado, *Advances in Financial Machine Learning*): a primary model fires a
signal; a secondary model decides **take/skip** using only what's knowable at
entry. This note is about building that secondary model so that the leak which makes
the narrator look smart cannot sneak back in — and then measuring it as a trader,
not a classifier.

## 2. Method

### 2.1 A strictly pre-entry corpus

For every signal the backtests fired, I emit one record split into two parts that
never touch:

- **context** — pre-entry features only: market regime, price/return/vol features,
  the signal snapshot, recent news, the proposed fill. Built from
  `df[df.index < entry_ts]`. Contains **no** forward field.
- **label** — the realized forward outcome of taking the signal, **net of costs**:
  the cost-netted forward return, the take/skip target, a 3-class rating, and the
  forward adverse excursion. This is the answer; it never enters `context`.

### 2.2 Leakage controls (all unit-tested)

The whole value of the corpus is the absence of look-ahead, so the guarantees are
asserted directly in `tests/test_decision_dataset_builder.py`:

- **Temporal wall** at the fill bar; the signal bar is the last visible bar.
- **Causal features** — appending or scrambling bars *after* the decision never
  changes a past decision's feature vector (tested by mutating the future and
  asserting the features are byte-identical).
- **News cutoff** at the signal bar, not the exit — a future-dated headline never
  appears in any record.
- **Purged + embargoed walk-forward split** — a train trade whose *holding window*
  reaches past a split boundary is dropped (purge), plus a 3-day embargo around each
  boundary kills boundary serial-correlation leakage. This is the silent re-leak
  that makes a naive temporal split *look* honest while it isn't.

### 2.3 Cost-aware labels, profit objective

- The label subtracts an extra round-trip cost (10 bps) on top of the simulator's
  slippage (5 bps/side); a signal must clear that hurdle to be labeled "take."
- The decision threshold is **not** tuned for accuracy. It's tuned on the validation
  split to maximize the summed net return of the taken set, subject to a minimum
  coverage — because a 99%-accurate model that only ever says "skip" makes no money.

### 2.4 Honest evaluation

`eval_policy.py` simulates the policy as a trader and reports coverage, hit rate,
average/total net return, per-trade and annualized Sharpe, and max drawdown — never
classification accuracy — against two benchmarks: **take-all** (take every signal)
and **beta** (buy-and-hold the underlying over the same window). `cross_validate.py`
re-folds the data into an expanding-window walk-forward with the same purge+embargo
and asks the only questions that matter: does it beat take-all, and does it beat
beta, and *in how many folds*? An edge that shows up in one fold is noise.

## 3. Data

| | |
|---|---|
| Universe | 30 liquid US symbols (mega-cap equities, sector ETFs, SPY/QQQ/IWM, TLT, GLD) |
| Strategies | 6 technical archetypes × {base, reverse} = 12 variants |
| Timeframe | Daily bars, 2021-03-31 → 2026-06-12 |
| Signals | 4,580 (65 excluded by purge/embargo) |
| Base rate | 1,437 take / 3,143 skip → only **31% of fired signals clear costs** |
| Outcome mix | 1,332 good / 598 neutral / 2,650 bad (net of cost) |

Most signals lose money net of costs — exactly the setting where a take/skip filter
should add value if it can.

## 4. Results

### 4.1 Single held-out test split (697 signals)

| metric | meta-labeler | take-all |
|---|---:|---:|
| coverage | 3.9% (27 trades) | 100% (697) |
| hit rate | **0.630** | 0.316 |
| avg net return / trade | **+1.35%** | +0.98% |
| total net return | +0.37 | +6.83 |
| per-trade Sharpe | **0.347** | 0.088 |
| annualized Sharpe (est.) | **1.77** | 0.48 |
| max drawdown | **−0.16** | −0.88 |
| edge vs take-all | +0.37%/trade | — |

Taken at face value: a sharp, selective filter that quadruples Sharpe and slashes
drawdown. **This is where an overfit project declares victory.** Two warning signs
say not to: it trades only 27 times (so the Sharpe rides on a tiny sample), and a
single split — even a properly purged one — can still be a lucky window.

### 4.2 Purged walk-forward CV (5 folds)

| fold | coverage | policy avg | take-all avg | beta avg | edge vs beta | ann Sharpe |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.203 | −0.0010 | −0.0031 | −0.0020 | +0.0041 | −0.18 |
| 2 | 0.083 | −0.0065 | −0.0018 | +0.0018 | −0.0000 | −1.16 |
| 3 | 0.067 | +0.0096 | +0.0030 | +0.0106 | −0.0026 | +1.05 |
| 4 | 0.406 | −0.0024 | −0.0032 | +0.0005 | −0.0014 | −0.37 |
| 5 | 0.039 | +0.0107 | +0.0087 | +0.0128 | −0.0057 | +1.46 |

**Summary:** mean policy +0.00208/trade vs take-all +0.00071 vs beta +0.00473.
Mean edge vs take-all **+0.0014** (positive); mean edge vs beta **−0.0011**
(negative). Profitable in **2/5** folds, beats take-all in **4/5**, beats beta in
**1/5**. Annualized Sharpe deflates from the single-split 1.77 to a fold mean of
**0.16**. The harness's own verdict:

> **PROMISING: beats take-all but not clearly beta (beta-contaminated).**

## 5. Interpretation

The model has learned *something* — it filters out bad signals well enough to beat
the take-everything baseline in 4 of 5 time periods. But the apparent profit is
mostly **beta**: 2021–2026 was largely a rising market, and a long-biased filter
looks profitable just by being exposed to it. Once you benchmark against
buy-and-hold over the *same windows*, the edge is gone (it clears beta in only one
fold, and the mean edge is negative). Coverage is also unstable across folds
(4%–41%), so the threshold is not finding a consistent operating point.

Conclusion: **weak, unstable signal-filtering with no demonstrated alpha over beta —
not deployable.** That is a negative result, and reporting it is the entire point.
The same pipeline that *could* have been tuned to print "Sharpe 1.8" instead tells
me the honest thing, because the benchmark and the walk-forward were built in from
the start rather than bolted on after a good-looking number appeared.

## 6. Limitations (and why each is also a next lever)

- **Linear model.** scikit-learn wasn't present in the run environment, so it fell
  back to the pure-numpy L2 logistic regression — the weaker of the two backends. A
  gradient-boosted model (`HistGradientBoostingClassifier`) can capture nonlinear
  feature interactions the linear model can't; rerunning with it is the cheapest way
  to learn whether "no alpha" is a capacity artifact or a genuine signal absence.
- **Low coverage / small samples.** Tuning for net P&L pushes the threshold high, so
  the taken set is small and high-variance. A coverage floor or a per-regime
  threshold would stabilize it.
- **Costs are a flat haircut.** No spread/impact model, no borrow cost on the
  shorts. The reverse archetypes especially deserve a stricter cost model.
- **One asset class, daily bars.** No intraday, no options (the actual product
  surface), no regime beyond the three-state classifier.

## 7. Reproduce

The methodology runs standalone on synthetic data — no market data, models, or
keys required:

```bash
pip install -r requirements.txt
python demo.py          # synthetic corpus with a real edge — the harness rewards it
python demo.py --noise  # no edge — the single split flatters, walk-forward refuses it
pip install -r requirements-dev.txt && pytest
```

The §4 figures come from real daily bars (2021-2026, 30 liquid US symbols) run
through the full pipeline:

```bash
# 1. Build the leak-free corpus -> decisions_tabular.jsonl.
#    The builder is in reference/ (decision_dataset_builder.py). It performs the
#    purged + embargoed walk-forward split but depends on a private backtest engine
#    (archetypes / dataset_builder / market_regime) and is included for review
#    rather than standalone execution.

# 2. Train the meta-labeler (threshold tuned for net P&L, not accuracy).
python meta_labeler.py --dataset decisions_tabular.jsonl --out metalabeler.pkl

# 3. Score it as a trader vs take-all, then purged walk-forward CV.
python eval_policy.py    --dataset decisions_tabular.jsonl --policy take_all
python eval_policy.py    --dataset decisions_tabular.jsonl --policy metalabeler --model-path metalabeler.pkl
python cross_validate.py --dataset decisions_tabular.jsonl --folds 5
```

> The §4 figures were produced with news features disabled (deterministic — no
> dependence on a news store; enabling them shifts the numbers only marginally) and
> with scikit-learn absent, so the model is the numpy logistic-regression fallback
> (see §6). The synthetic `demo.py` above exercises the identical evaluation path.

## 8. What this demonstrates

Not a profitable strategy — the honest answer is that this one isn't, yet. What it
demonstrates is the part that's hard to fake: an experimental setup that refuses to
leak the future into the features, benchmarks skill against beta rather than against
zero, validates across time instead of on one lucky split, and prints a verdict that
distinguishes *profitable*, *reduces-loss-but-still-negative*, and *no-edge*. The
willingness to build the tool that can tell you your idea doesn't work — and then to
believe it — is the whole job.

---

*Code: `decision_dataset_builder.py`, `decision_features.py`, `meta_labeler.py`,
`training/eval_policy.py`, `training/cross_validate.py`. Leak-free guarantees are
enforced in `tests/test_decision_dataset_builder.py` (10 tests).*
