# Real Data Training Results

**Date:** 2026-01-18
**Training:** XGBoost on Actual 1-Minute Intraday Data

---

## Executive Summary

⚠️ **CRITICAL ISSUE: SEVERE OVERFITTING**

Training on real 1-minute data revealed a **worse** problem than synthetic data:

| Symbol | Train Accuracy | Test Accuracy | Gap | vs Target |
|--------|----------------|---------------|-----|-----------|
| **SPY** | **99.06%** | **50.16%** | **48.90%** | **-14.84%** ❌ |
| **QQQ** | **98.98%** | **49.53%** | **49.45%** | **-15.47%** ❌ |
| Target | - | 65.00% | - | - |

**Status:** ❌ **COMPLETE FAILURE** - Models cannot generalize to held-out data

---

## What Happened?

### The Overfitting Problem

**SPY Results:**
- Training set: 99.06% (nearly perfect!)
- Test set: 50.16% (random guessing)
- **Gap: 48.90%** ← Model completely memorized training data

**QQQ Results:**
- Training set: 98.98%
- Test set: 49.53%
- **Gap: 49.45%**

### Why This Happened

**1. Insufficient Data**
- Only 1,595 samples from 10 trading days
- 1,276 training samples
- 319 test samples
- **Too small for 47 features!**

**Rule of thumb:** Need ~10-20 samples per feature
- 47 features × 20 = 940 minimum samples
- We have 1,276 training samples (barely enough)
- XGBoost with 100 trees can memorize this easily

**2. XGBoost Too Powerful for Small Dataset**
- 100 trees × max_depth 6 = very high capacity
- Can memorize 1,276 samples perfectly
- No generalization to unseen data

**3. Time-Based Split Exposes Regime Changes**
- Training: Days 1-8 (Jan 7-14)
- Test: Days 9-10 (Jan 15-16)
- If market behavior changed (news, volatility shift), model fails
- This is more realistic than random split but harder to pass

---

## Detailed Results

### SPY on Real 1-Minute Data

**Dataset:**
- Total samples: 1,595
- Training: 1,276 (80%)
- Test: 319 (20%)
- Features: 47
- Data: Jan 7-16, 2026 (10 trading days)

**Performance:**
```
Train Accuracy: 99.06%
Test Accuracy:  50.16%
Gap:            48.90%
```

**Test Set Confusion Matrix:**
```
              Predicted
              DOWN  NEUTRAL  UP
Actual DOWN     82      0     68
Actual NEUTRAL   4      0      4
Actual UP       83      0     78
```

**Classification Report:**
```
           Precision  Recall  F1-Score  Support
DOWN          0.49     0.55    0.51      150
NEUTRAL       0.00     0.00    0.00        8
UP            0.52     0.48    0.50      161
```

**Key Issues:**
- Never predicts NEUTRAL (again!)
- DOWN/UP predictions essentially random (47-48% recall each)
- No edge whatsoever

**Top 5 Important Features:**
1. inst_hp_support_dist (6.38%)
2. V12_ask (5.59%)
3. inst_hp_resist_dist (5.14%)
4. V4_high_mid (4.69%)
5. V13_mid_price (4.67%)

### QQQ on Real 1-Minute Data

**Dataset:**
- Total samples: 1,595
- Training: 1,276 (80%)
- Test: 319 (20%)
- Features: 37
- Data: Jan 7-16, 2026 (10 trading days)

**Performance:**
```
Train Accuracy: 99.00%
Test Accuracy:  49.53%
Gap:            49.45%
```

**Same issues as SPY:**
- Severe overfitting
- No NEUTRAL predictions
- Random performance on test set

---

## Comparison: Synthetic vs Real Training

| Metric | Synthetic Training | Real Training |
|--------|-------------------|---------------|
| **SPY Train Acc** | 82.08% | 99.06% |
| **SPY Test Acc** | 66.53% | 50.16% |
| **SPY Gap** | 15.55% | 48.90% |
| **QQQ Train Acc** | 82.19% | 98.98% |
| **QQQ Test Acc** | 65.86% | 49.53% |
| **QQQ Gap** | 16.33% | 49.45% |
| **Training Samples** | 12,550 | 1,276 |
| **Test on Real Data** | 49.95% | 50.16% |

**Key Insight:**
- Synthetic training had MORE data (12,550 vs 1,595)
- But synthetic data was unrealistic
- Real data training shows the true problem: **not enough data**

---

## Root Cause Analysis

### Problem 1: Data Scarcity

**Current situation:**
- 10 days × ~160 minutes/day = 1,600 minutes
- 1,600 minutes ÷ 5 (window size) = 320 windows
- After cleaning: 1,595 samples
- 80% train = 1,276 samples

**What we need:**
- 47 features × 50 samples/feature = **2,350 minimum**
- Better: 47 × 100 = **4,700 samples**
- Ideal: 47 × 200 = **9,400 samples**

**How to get more data:**
- Download more days (but Tradier only has 30 days)
- Use 5-minute bars (less granular, but 5x more history)
- Use overlapping windows (increases samples but introduces autocorrelation)
- Get different data source (paid service with years of history)

### Problem 2: Model Complexity Too High

**Current XGBoost params:**
```python
'max_depth': 6,           # Trees up to 6 levels deep
'n_estimators': 100,      # 100 trees
'learning_rate': 0.1,     # Standard learning rate
'subsample': 0.8,         # Use 80% of samples per tree
'colsample_bytree': 0.8,  # Use 80% of features per tree
'reg_alpha': 0.1,         # L1 regularization (weak)
'reg_lambda': 1.0,        # L2 regularization (moderate)
```

**Model capacity:** Huge!
- 100 trees × 2^6 leaves = 6,400 decision paths
- Can easily memorize 1,276 samples

**Better params for small dataset:**
```python
'max_depth': 3,           # Shallower trees (reduce from 6)
'n_estimators': 20,       # Fewer trees (reduce from 100)
'learning_rate': 0.05,    # Slower learning
'subsample': 0.7,         # More randomness
'colsample_bytree': 0.7,  # More randomness
'reg_alpha': 1.0,         # Stronger L1 (10x increase)
'reg_lambda': 5.0,        # Stronger L2 (5x increase)
'min_child_weight': 5,    # NEW: Require more samples per leaf
```

### Problem 3: Too Many Features

**Current features: 47 for SPY**

**Feature categories:**
- Base features: 22
- Institutional: 10
- Greeks (gamma/vanna): 5
- LOB microstructure: 10

**With only 1,276 samples:**
- Each feature gets ~27 samples to learn from
- Many features likely spurious correlations
- Need feature selection

**Better approach:**
- Use only top 15-20 features
- Feature selection via importance scores
- Reduce to core predictive features

### Problem 4: NEUTRAL Class Too Rare

**Real data distribution:**
- DOWN: 46%
- NEUTRAL: **2%** ← Problem!
- UP: 52%

**With only 1.9% NEUTRAL:**
- Training: 31 samples (out of 1,276)
- Test: 8 samples (out of 319)
- Model learns to ignore this class

**Solutions:**
- Remove NEUTRAL entirely (binary classification)
- Adjust threshold (make NEUTRAL easier to trigger)
- SMOTE oversampling (synthetic minority samples)

---

## Solutions to Try (Prioritized)

### Solution 1: Binary Classification (HIGHEST PRIORITY)

**Remove NEUTRAL class entirely**

**Rationale:**
- NEUTRAL is only 2% of data
- Market rarely stays flat
- Binary UP/DOWN is cleaner prediction
- Reduces problem complexity

**Expected improvement:** +2-5% accuracy

**Implementation:**
```python
# In TargetCalculator
def calculate_binary_target(current_price, future_quotes):
    future_price = future_quotes[-1].mid_price
    return 'UP' if future_price > current_price else 'DOWN'
```

### Solution 2: Reduce Model Complexity

**Adjust XGBoost hyperparameters for small dataset**

**Changes:**
```python
params = {
    'max_depth': 3,          # ↓ from 6 (reduce overfitting)
    'n_estimators': 20,      # ↓ from 100 (simpler model)
    'learning_rate': 0.05,   # ↓ from 0.1 (slower learning)
    'min_child_weight': 5,   # NEW (require more samples per leaf)
    'reg_alpha': 1.0,        # ↑ from 0.1 (stronger L1)
    'reg_lambda': 5.0,       # ↑ from 1.0 (stronger L2)
    'subsample': 0.6,        # ↓ from 0.8 (more randomness)
    'colsample_bytree': 0.6, # ↓ from 0.8 (more randomness)
}
```

**Expected improvement:** +5-10% test accuracy

### Solution 3: Feature Selection

**Use only top 15-20 most important features**

**From current results, top features are:**
1. inst_hp_support_dist
2. V12_ask
3. inst_hp_resist_dist
4. V4_high_mid
5. V13_mid_price
6. V11_bid
7. V21_vwap
8. V22_arrival_rate
9. V5_low_mid
10. V14_spread
11. lob_micro_edge
12. V2_close_mid
13. inst_mhp_support_dist
14. V7_avg_spread
15. V20_volume

**Benefit:**
- 47 → 15 features
- Less overfitting risk
- Simpler model
- Easier to interpret

**Expected improvement:** +3-5% test accuracy

### Solution 4: Use 5-Minute Data Instead

**Switch from 1-minute to 5-minute bars**

**Current:**
- 1-min: ~8,000 bars → 1,595 samples
- Limited to 10 days of data

**With 5-minute:**
- 5-min: ~4,400 bars → 880 samples (worse!)

**Actually not better for sample count, but:**
- Can get more historical days
- Less noise
- More stable patterns

**Expected improvement:** Unclear, may help with robustness

### Solution 5: Overlapping Windows

**Create overlapping windows instead of non-overlapping**

**Current:**
- Window every 5 bars (non-overlapping)
- 8,000 bars → 1,600 windows

**Overlapping:**
- Window every 1 bar (sliding window)
- 8,000 bars → 7,990 windows (5x more!)

**Downside:**
- Samples are autocorrelated (not independent)
- Overstates true sample size
- May still overfit

**Expected improvement:** +5-10% but inflated (not real improvement)

### Solution 6: Get More Historical Data

**Use different data source**

**Options:**
1. Polygon.io (paid) - years of minute data
2. AlphaVantage (paid) - extensive history
3. IEX Cloud (paid) - high quality data
4. Yahoo Finance (free but limited intraday)

**With 3-6 months of data:**
- 60-120 trading days
- ~60,000-120,000 1-min bars
- ~12,000-24,000 samples
- **Enough to train properly!**

**Expected improvement:** +10-20% (most impactful)

---

## Recommended Immediate Actions

### Quick Fixes (Today)

1. **Binary classification** - Remove NEUTRAL class
2. **Reduce complexity** - Simpler XGBoost params
3. **Feature selection** - Use top 15 features only

**Expected combined impact:** +10-15% → ~60-65% accuracy

### Medium-term (This Week)

4. **Get more data** - Sign up for paid data source
5. **Download 3-6 months** - Build proper training set
6. **Retrain with sufficient samples**

**Expected impact:** +15-20% → 65-70% accuracy

### Long-term (This Month)

7. **Walk-forward validation** - Test on rolling windows
8. **Ensemble models** - Combine XGBoost, LightGBM, CatBoost
9. **Online learning** - Adapt to regime changes

**Expected impact:** +5-10% → 70-75% accuracy

---

## Updated Roadmap

### Phase 1: Quick Wins (1 day)
- [ ] Implement binary classification (UP/DOWN only)
- [ ] Reduce XGBoost complexity (max_depth=3, n_estimators=20)
- [ ] Feature selection (top 15 features)
- [ ] **Target: 60% accuracy**

### Phase 2: Data Acquisition (3 days)
- [ ] Sign up for Polygon.io or similar
- [ ] Download 6 months of 1-minute data
- [ ] Validate data quality
- [ ] **Increase samples from 1,595 to 24,000+**

### Phase 3: Proper Training (1 week)
- [ ] Train on 6 months of data
- [ ] Time-based validation (monthly rolling)
- [ ] Hyperparameter tuning with cross-validation
- [ ] **Target: 65% accuracy**

### Phase 4: Production Ready (2 weeks)
- [ ] Ensemble multiple models
- [ ] Walk-forward validation
- [ ] Paper trading simulation
- [ ] **Target: 65-70% accuracy with confidence**

---

## Key Learnings

### 1. Data Quantity Matters More Than Quality

**Comparison:**
- Synthetic: 12,550 samples → 66% test (on synthetic)
- Real: 1,595 samples → 50% test (on real)

**Lesson:** Need BOTH quality AND quantity
- Synthetic had quantity but not quality
- Real has quality but not quantity
- **Need real data with high sample count**

### 2. Overfitting is Easier Than Expected

**Surprising result:**
- Only 1,276 training samples
- XGBoost achieved 99% training accuracy
- Means model has way too much capacity

**Lesson:** Match model complexity to dataset size

### 3. Time-Based Splits Are Harder

**Random split:**
- Shuffles data
- Test set has samples from all time periods
- Easier to generalize (train and test are similar)

**Time-based split:**
- Test set is future data only
- Model must predict genuinely unseen market behavior
- Harder but more realistic
- **This is what live trading will face**

**Lesson:** Always use time-based validation for time series

### 4. NEUTRAL Class is Problematic

**Every test shows:**
- NEUTRAL is <2% of real data
- Model never predicts it
- Wastes model capacity

**Lesson:** Binary classification is more practical

---

## Final Summary

**Attempt 1: Synthetic Data**
- Result: 66% test (on synthetic), 50% test (on real)
- Issue: Unrealistic data doesn't transfer to reality

**Attempt 2: Real Data**
- Result: 99% train, 50% test (on real)
- Issue: Not enough data, severe overfitting

**Root Problem:**
- Need 10,000-20,000 samples of **real** data
- Currently have only 1,595 samples
- Must acquire more historical data or change approach

**Path Forward:**
1. Quick wins: Binary classification + simpler model → 60% (maybe)
2. Long-term: Get 6 months of real data → 65% (likely)
3. Ultimate: Ensemble + proper validation → 70% (possible)

**Current Status:** 🔴 **BLOCKED ON DATA**

Cannot achieve 65% target without more training data.

**Recommendation:**
1. Try quick wins today (binary + simpler model)
2. If that fails to reach 60%, must acquire more data
3. Budget for paid data service (~$100-300/month)

---

**Next Action:** Implement binary classification with reduced complexity and test again.
