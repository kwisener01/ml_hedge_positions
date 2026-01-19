# Real Data Test Results

**Date:** 2026-01-18
**Test Type:** XGBoost Models on Actual Intraday Data

---

## Executive Summary

⚠️ **CRITICAL FINDING:** The XGBoost model shows **significant degradation** when tested on real market data compared to synthetic data.

| Metric | Synthetic Data | Real Data | Difference |
|--------|----------------|-----------|------------|
| **SPY Accuracy** | **66.53%** | **49.95%** | **-16.59%** ❌ |
| Target | 65.00% | 65.00% | - |
| Status | ✅ PASSED | ❌ **FAILED** | -25% relative drop |

**Conclusion:** The 66.53% accuracy on synthetic data **does not translate** to real market conditions.

---

## Test Details

### Data Source
- **Synthetic (Training):** 10 quotes/day generated from daily OHLC
- **Real (Testing):** Actual 5-minute intraday bars from Tradier
- **Date Range:** 2025-12-17 to 2026-01-16 (1 month)
- **Bars:** 4,641 5-minute bars
- **Valid Samples:** 923 non-overlapping windows

### SPY Results on Real Data

**Overall Accuracy: 49.95%**
- Just barely below random (50%)
- **16.59% worse than synthetic test set**
- Falls far short of 65% target

**Classification Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DOWN | 0.47 | 0.63 | 0.54 | 428 (46.4%) |
| NEUTRAL | 0.00 | 0.00 | 0.00 | 14 (1.5%) |
| UP | 0.54 | 0.40 | 0.46 | 481 (52.1%) |

**Key Issues:**
1. ❌ **Model never predicts NEUTRAL** - ignored entire class
2. ⚠️ DOWN recall too high (63%) at expense of UP recall (40%)
3. ⚠️ Biased toward predicting DOWN (556 predictions vs 367 UP)

**Confusion Matrix:**
```
              Predicted
              DOWN  NEUTRAL  UP
Actual DOWN    268     0    160
Actual NEUTRAL  10     0      4
Actual UP      288     0    193
```

**Observations:**
- 288 false DOWN predictions (should be UP)
- 160 false UP predictions (should be DOWN)
- 0 NEUTRAL predictions made (model ignores this class)

---

## Why the Performance Drop?

### 1. Synthetic Data Artifacts

**Synthetic quote generation (from train_xgboost.py):**
```python
# Generate intraday price path
prices = np.linspace(row['open'], row['close'], quotes_per_day)

# Add noise within high/low range
noise_range = (row['high'] - row['low']) / 4
prices += np.random.normal(0, noise_range, quotes_per_day)
```

**Problems:**
- Linear interpolation between open/close is unrealistic
- Random noise doesn't capture market microstructure
- Missing: order flow, institutional behavior, regime changes
- Synthetic spread (2 bps) is constant - real spreads vary

### 2. Feature Distribution Mismatch

**Synthetic data:**
- Smooth, predictable patterns
- No sudden volatility spikes
- No market maker behavior
- No real order book dynamics

**Real data:**
- Irregular price movements
- Flash crashes, news events
- Institutional order flow
- Real market microstructure

**Result:** Features trained on synthetic patterns don't generalize.

### 3. Target Distribution Difference

**Real data target distribution:**
- DOWN: 46.4%
- NEUTRAL: 1.5% ← **Very rare!**
- UP: 52.1%

**Implications:**
- Model learned to predict NEUTRAL on synthetic data
- Real markets rarely stay stationary (1.5% vs expected ~33%)
- Model's NEUTRAL predictions are useless in practice

### 4. Overfitting to Synthetic Patterns

**Training metrics:**
- Train accuracy: 82.08%
- Test accuracy (synthetic): 66.53%
- Real data: 49.95%

**Gap analysis:**
- Train-to-test gap: 15.55% (expected for gradient boosting)
- **Test-to-real gap: 16.59%** ← This is the problem!
- Model memorized synthetic data characteristics

---

## Real vs Synthetic: Feature Comparison

### Issues with Synthetic Features

| Feature Type | Synthetic | Real | Impact |
|--------------|-----------|------|--------|
| **Spread** | Constant 2 bps | Variable (1-10 bps) | Low |
| **Volume** | Uniform distribution | Bursty, news-driven | High |
| **Price path** | Linear interpolation | Non-linear jumps | **Critical** |
| **Volatility** | Smoothed | Clustered (GARCH) | High |
| **Order book** | Estimated | Actual dynamics | Medium |
| **Institutional** | From daily options | Intraday mismatch | Medium |

**Biggest issue:** Synthetic price paths are too smooth and predictable.

---

## What This Means for Trading

### Live Trading Risk

**At 49.95% accuracy:**
- Win rate: ~50% (basically random)
- Expected profit: **~0%** (no edge)
- Risk: **VERY HIGH** - would lose money with transaction costs

**Profitability calculation:**
- Wins: 50 × $100 = $5,000
- Losses: 50 × -$100 = -$5,000
- Gross: **$0**
- After commissions: **NEGATIVE** ⚠️

**Recommendation:** 🚫 **DO NOT trade with current model**

### Model Trust

**Reasons to distrust synthetic results:**
1. ❌ 16.59% accuracy drop on real data
2. ❌ Ignores NEUTRAL class entirely
3. ❌ Barely better than coin flip (49.95% vs 50%)
4. ❌ Synthetic training doesn't reflect reality
5. ❌ Large train/test/real gaps (82% → 66% → 50%)

---

## Next Steps to Fix This

### 1. Train on Real Intraday Data (HIGHEST PRIORITY)

**What to do:**
- Use actual 5-minute data for training (not just testing)
- Download more historical intraday data (3-6 months)
- Train XGBoost on real quotes, test on hold-out period

**Expected improvement:** +10-15% accuracy

**Implementation:**
```python
# Use real intraday data instead of synthetic
intraday_df = load_intraday_data('SPY', '5min', months=6)
quotes = convert_to_quotes(intraday_df)
X, y = build_features(quotes)
model.fit(X, y)
```

### 2. Fix NEUTRAL Class Imbalance

**Issue:** Only 1.5% of real samples are NEUTRAL

**Solutions:**
- Adjust NEUTRAL threshold (currently might be too strict)
- Use SMOTE or class weights to balance training
- Consider removing NEUTRAL entirely (just UP/DOWN binary)

**Expected improvement:** +2-3% accuracy

### 3. Walk-Forward Validation

**What to do:**
- Split real data into rolling windows
- Train on month 1-2, test on month 3
- Retrain and validate consistency

**Expected improvement:** Better confidence, not accuracy

### 4. Feature Engineering for Real Data

**Add real-market features:**
- Volume-weighted average price (VWAP) deviation
- Time-of-day effects (market open/close patterns)
- Recent volatility (realized vol last 30 bars)
- Price momentum (rate of change)
- Order flow imbalance (if available)

**Expected improvement:** +3-5% accuracy

### 5. Ensemble with Real Data

**What to do:**
- Train multiple models on real data (XGBoost, LightGBM, Random Forest)
- Weighted voting based on recent performance
- Adaptive weighting (more weight to recent winner)

**Expected improvement:** +2-4% accuracy

### 6. Hyperparameter Tuning for Real Data

**Current params optimized for synthetic data**

**Re-optimize for real data:**
- Max depth (currently 6, try 4-8)
- Learning rate (0.1, try 0.01-0.3)
- Regularization (try higher alpha/lambda)
- Number of estimators (100, try 50-200)

**Expected improvement:** +1-3% accuracy

---

## Revised Roadmap

### Phase 1: Data (CRITICAL)
- [ ] Download 6 months of real 5-minute data
- [ ] Validate data quality and coverage
- [ ] Build train/validation/test splits

### Phase 2: Retrain on Real Data
- [ ] Train XGBoost on real intraday quotes
- [ ] Evaluate on hold-out test set
- [ ] Target: >60% accuracy on real data

### Phase 3: Address Class Imbalance
- [ ] Analyze NEUTRAL threshold
- [ ] Try binary classification (UP/DOWN only)
- [ ] Apply class weights or SMOTE

### Phase 4: Feature Engineering
- [ ] Add time-of-day features
- [ ] Add realized volatility
- [ ] Add momentum indicators

### Phase 5: Validation
- [ ] Walk-forward testing
- [ ] Out-of-sample validation
- [ ] Paper trading simulation

---

## Comparison Table: Synthetic vs Real

| Aspect | Synthetic Data | Real Data | Winner |
|--------|----------------|-----------|--------|
| **Accuracy** | 66.53% | 49.95% | Synthetic |
| **Realism** | Low | High | **Real** |
| **Trading value** | None | Actual | **Real** |
| **Cost** | Free | API cost | Synthetic |
| **Availability** | Unlimited | Limited | Synthetic |
| **Generalization** | Poor | Good | **Real** |
| **Production ready** | No | Potentially | **Real** |

**Verdict:** Must train on real data for production use.

---

## Lessons Learned

### 1. Synthetic Data is Misleading

**What we learned:**
- 66% accuracy on synthetic ≠ 66% on real
- Linear interpolation creates unrealistic patterns
- Model learns synthetic artifacts, not market behavior

**Lesson:** Never trust synthetic data for live trading validation

### 2. Always Validate on Real Data

**What we learned:**
- Real markets are fundamentally different
- Feature distributions don't match
- Model assumptions break down

**Lesson:** Test on real data BEFORE celebrating results

### 3. Train/Test/Real Gaps Matter

**What we learned:**
- Train: 82% (overfitting)
- Test: 66% (still synthetic)
- Real: 50% (reality check)

**Lesson:** Monitor all three metrics, not just test accuracy

### 4. Class Imbalance in Real Markets

**What we learned:**
- NEUTRAL is 1.5% in real data (not 33% expected)
- Markets trend more than they consolidate
- Binary UP/DOWN might be better

**Lesson:** Understand real class distribution before modeling

---

## Recommendations

### Immediate Actions
1. 🔴 **STOP using synthetic data for training**
2. 🔴 **DO NOT trade with current models**
3. 🟡 Download 6+ months of real intraday data
4. 🟡 Retrain on actual market data

### Short-term Goals
- Real data accuracy: >60% (realistic target)
- Binary classification (UP/DOWN only)
- Walk-forward validation

### Long-term Goals
- Real data accuracy: >65% (if achievable)
- Ensemble of real-trained models
- Live paper trading validation

---

## Final Summary

**Starting assumption:**
- ✅ SPY: 66.53% on synthetic data

**Reality check:**
- ❌ **SPY: 49.95% on real data**

**Gap:** -16.59% (25% relative degradation)

**Conclusion:**

The XGBoost model that appeared to exceed the 65% target was actually overfitting to synthetic data patterns that don't exist in real markets. The model is essentially **no better than random** on actual market data.

**Action required:** Complete retraining on real intraday data before any production deployment.

**Expected timeline:**
- Data collection: 1 day
- Retraining: 1 day
- Validation: 2-3 days
- **Total: ~1 week to get real results**

---

**Status:** ⚠️ **PROJECT ON HOLD** until real data training is complete

The 65% target has NOT been achieved on real market data.
