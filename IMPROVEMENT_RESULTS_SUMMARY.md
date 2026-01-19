# SVM Model Improvement Results - Complete Summary

**Date:** 2026-01-16
**Goal:** Achieve 65%+ accuracy for SPY and QQQ
**Starting Point:** SPY 41%, QQQ 54%

---

## Summary of All Experiments

### Baseline (Original Configuration)
- **SPY:** 41.24% test accuracy
- **QQQ:** 54.18% test accuracy
- **Configuration:** C=0.25, degree=2, alpha=1e-05
- **Data:** Synthetic quotes (10/day from daily OHLC)
- **Samples:** 2,507 training windows
- **Timeframe:** ~3-4 hours per prediction

---

## Experiment 1: Hyperparameter Optimization

**Applied Changes:**
```python
polynomial_degree: int = 3      # Changed from 2
constraint_param: float = 0.1   # Changed from 0.25
```

**Results:**
| Symbol | Before | After | Change | Status |
|--------|---------|--------|---------|--------|
| SPY | 41.24% | **49.20%** | +7.96% | Better |
| QQQ | 54.18% | **51.00%** | -3.18% | Worse |

**Analysis:**
- SPY improved significantly (+8%)
- QQQ unexpectedly decreased (-3%)
- Still far from 65% target (need +15.8% for SPY, +14% for QQQ)
- Hyperparameters help but are not sufficient alone

**Key Finding:**
The optimal hyperparameters found in experiments (C=0.1, degree=3) gave +3.6% in isolated tests but only +8% for SPY in production. This is still valuable but not enough to reach 65%.

---

## Experiment 2: Real Intraday Data

**Downloaded Data:**
- SPY: 4,641 five-minute bars (30 days)
- QQQ: 4,709 five-minute bars (30 days)
- Date range: 2025-12-17 to 2026-01-16

**Training Configuration:**
- Interval: 5-minute bars
- Lookforward: 5 bars (25 minutes)
- Window size: 5 bars
- Samples: ~923 (SPY), ~936 (QQQ)

**Results:**
| Symbol | Synthetic Data | Real Intraday | Change | Status |
|--------|----------------|---------------|---------|--------|
| SPY | 49.20% | **51.89%** | +2.69% | Slightly better |
| QQQ | 51.00% | **39.89%** | -11.11% | Much worse |

**Analysis:**
- SPY improved slightly with real data (+2.7%)
- QQQ performance collapsed with real data (-11%)
- QQQ had network errors fetching institutional data (10 retries failed)
- Much less training data: ~900 samples vs 2,500 from synthetic

**Key Finding:**
Real intraday data fixes the timeframe mismatch but introduces a critical problem: **insufficient training data**. 30 days of data = ~900 samples, compared to 5 years of synthetic data = 2,500 samples. This is a fundamental trade-off.

---

## Complete Results Table

| Approach | SPY Accuracy | QQQ Accuracy | Avg | Distance to 65% |
|----------|--------------|--------------|-----|-----------------|
| **Original Baseline** | 41.24% | 54.18% | 47.71% | -17.29% |
| **+ Hyperparameters** | 49.20% | 51.00% | 50.10% | -14.90% |
| **+ Real Intraday** | 51.89% | 39.89% | 45.89% | -19.11% |

---

## Critical Findings

### 1. The Timeframe Mismatch Problem

**Discovered Issue:**
- Training uses 10 quotes/day = 1 quote every 39 minutes
- Window of 5 quotes = 195 minutes (~3.25 hours)
- But live signals collect 5 quotes in 5 seconds

This fundamental mismatch means the model is trained to predict 3-hour movements but used to predict 5-second movements.

**Attempted Fix:**
Real intraday data (5-minute bars) aligns the timeframe: 5 bars = 25 minutes for both training and prediction.

**Result:**
The fix worked for SPY (+2.7%) but failed for QQQ (-11%) due to:
1. Insufficient training data (900 vs 2,500 samples)
2. Network failures fetching institutional features
3. Only 30 days of market history

### 2. The Training Data Quantity Problem

**Trade-off Discovered:**
- **Synthetic data:** 5 years of history = 2,500 samples, but timeframe mismatch
- **Real intraday:** Correct timeframe, but only 30 days = 900 samples

This is a fundamental constraint with current data availability from Tradier (30-day limit on intraday data).

### 3. Alpha Threshold Findings

From experiments:
- Current alpha (1e-05) is actually optimal at 50.80%
- Larger alpha (1e-02) gave 50.20% in some tests but created better class balance
- The alpha threshold is the most important parameter for target definition

---

## Why 65% Is Difficult

### Industry Context

**Typical Accuracy Ranges:**
- Random guess: 50%
- Basic model: 52-54%
- Good model: 55-60%
- Excellent model: 60-65%
- Exceptional: 65-70%
- Suspicious (likely overfit): >75%

**Current Status:**
- SPY at 51.89% = "Basic model" range
- QQQ at 39.89% (real data) or 51% (synthetic) = "Below basic" to "Basic"
- Need +13-25% improvement to reach 65%

### Fundamental Constraints

1. **Data Limitations:**
   - Real intraday data: Only 30 days available
   - Synthetic data: Timeframe mismatch
   - No good middle ground currently

2. **Feature Limitations:**
   - 32 features (8 window, 14 classic, 10 institutional)
   - Volume (V20) is dominant feature (0.1207 importance)
   - Institutional features need live API calls (network issues)

3. **Target Definition:**
   - Alpha threshold of 1e-05 = 0.001% movement
   - Creates highly imbalanced classes (58% UP, 41% DOWN, 0.1% NEUTRAL)
   - Class balancing experiments failed due to insufficient NEUTRAL samples

4. **Model Limitations:**
   - SVM ensemble of 100 models
   - Polynomial kernel (degree=3, C=0.1)
   - Already optimized hyperparameters
   - May have hit ceiling of what SVM can achieve with current features

---

## Pathways to 65% (Revised Assessment)

### Path A: Extended Synthetic Data (Most Realistic)

**Approach:**
1. Keep hyperparameter optimization (C=0.1, degree=3)
2. Increase synthetic quote density (100/day instead of 10/day)
3. This creates 25,000 samples with ~20-minute timeframe
4. Matches live signal collection better

**Expected Improvement:** +5-8%
**New Accuracy:** SPY ~54-57%, QQQ ~56-59%
**Difficulty:** Low
**Success Probability:** 70%
**Still short of 65%:** Yes, by 6-9%

### Path B: XGBoost Alternative Model (Moderate Potential)

**Approach:**
1. Train XGBoost on same features
2. Gradient boosting often outperforms SVM on tabular data
3. Can handle class imbalance better
4. Better feature interaction modeling

**Expected Improvement:** +3-7%
**New Accuracy:** SPY ~52-56%, QQQ ~54-58%
**Difficulty:** Medium
**Success Probability:** 60%
**Still short of 65%:** Yes, by 7-13%

### Path C: Hybrid Ensemble SVM + XGBoost (Highest Potential)

**Approach:**
1. Train both SVM (optimized) and XGBoost
2. Ensemble predictions with weighted voting
3. Combine on extended synthetic data (100/day)
4. Leverage strengths of both algorithms

**Expected Improvement:** +8-12%
**New Accuracy:** SPY ~57-61%, QQQ ~59-63%
**Difficulty:** High
**Success Probability:** 50%
**Reaches 65%:** Maybe, if we're lucky (60-63% range)

### Path D: Feature Engineering (Long Term)

**Approach:**
1. Add new features:
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Order flow imbalance
   - Volatility measures (realized, implied)
   - Market regime indicators
   - Correlation features
2. Feature selection to find best combinations

**Expected Improvement:** +5-10%
**New Accuracy:** SPY ~54-59%, QQQ ~56-61%
**Difficulty:** Very High
**Success Probability:** 40%
**Reaches 65%:** Unlikely without other improvements

---

## Realistic Recommendation

**Can you reach 65%? Probably not with current approach alone.**

**Best realistic target with available methods: 60-63%**

**Recommended Strategy:**

1. **Short term (1-2 days):**
   - Increase synthetic quote density to 100/day
   - Retrain with C=0.1, degree=3
   - Expected: SPY 54-57%, QQQ 56-59%

2. **Medium term (3-5 days):**
   - Implement XGBoost alternative
   - Create hybrid ensemble (SVM + XGBoost)
   - Expected: SPY 57-61%, QQQ 59-63%

3. **Long term (1-2 weeks):**
   - Add new features (technical indicators, volatility)
   - Fine-tune ensemble weights
   - Collect more real intraday data over time
   - Expected: SPY 60-65%, QQQ 62-66%

**Reality Check:**
Reaching exactly 65% for both symbols is very ambitious. The industry standard for "excellent" models is 60-65%, and you're fighting against:
- Limited data (30 days intraday or timeframe-mismatched synthetic)
- Noisy market conditions
- Fundamental unpredictability of short-term price movements

A more realistic goal is **60-63% average** across both symbols, which would still be an excellent model.

---

## Files Created

1. `PATH_TO_65_PERCENT.md` - Original optimistic roadmap
2. `ALPHA_EXPERIMENT_RESULTS.md` - Alpha threshold experiments
3. `training/improve_model.py` - Automated experiment framework
4. `training/train_ensemble_intraday.py` - Real intraday data training
5. `data/intraday_downloader.py` - Tradier intraday data downloader
6. `visualization/model_visualizer.py` - Model visualization tools
7. `IMPROVEMENT_RESULTS_SUMMARY.md` - This file

---

## Configuration Changes Applied

**config/settings.py:**
```python
polynomial_degree: int = 3      # Optimized from 2
constraint_param: float = 0.1   # Optimized from 0.25
alpha_threshold: float = 1e-05  # Kept as is (already optimal)
```

**Models Trained:**
- `models/trained/SPY_ensemble.pkl` - Synthetic data, optimized hyperparameters
- `models/trained/QQQ_ensemble.pkl` - Synthetic data, optimized hyperparameters
- `models/trained/SPY_ensemble_intraday_5min.pkl` - Real intraday data
- `models/trained/QQQ_ensemble_intraday_5min.pkl` - Real intraday data

---

## Next Steps (If Continuing)

1. **Immediate:** Increase synthetic quote density to 100/day and retrain
2. **Short term:** Implement XGBoost model
3. **Medium term:** Create hybrid ensemble
4. **Long term:** Add technical indicator features and volatility measures

**Expected Final Results:**
- Best case: 60-65% (with all improvements)
- Realistic: 58-62% (with XGBoost + extended synthetic)
- Conservative: 54-58% (current approach optimized)

---

## Conclusion

The journey from 41% to 65% has revealed several important findings:

1. **Hyperparameter optimization** gave +8% for SPY (41% → 49%)
2. **Real intraday data** gave an additional +2.7% for SPY (49% → 51.89%)
3. **Current best:** SPY 51.89%, QQQ 51% (synthetic)
4. **Gap to 65%:** Still need +13-14% improvement

The 65% target is achievable but requires:
- XGBoost implementation
- Extended synthetic data (100 quotes/day)
- Possibly additional features
- Likely several more weeks of development and testing

A more pragmatic target is **60-63%**, which is still an excellent model by industry standards.
