# Model Improvement Guide

Complete guide to improving SVM ensemble performance, specifically targeting SPY (41% → 50%+).

## Current Performance

| Symbol | Test Accuracy | Status |
|--------|---------------|--------|
| **QQQ** | 54.18% | ✅ Production ready |
| **SPY** | 41.24% | ⚠️ Needs improvement |

## Quick Start: Run All Experiments

```bash
# Install additional dependencies
pip install imbalanced-learn

# Run complete improvement pipeline for SPY
python training/improve_model.py --symbol SPY --experiment all

# Or run specific experiments
python training/improve_model.py --symbol SPY --experiment alpha
python training/improve_model.py --symbol SPY --experiment hyperparams
python training/improve_model.py --symbol SPY --experiment features
```

Results saved to: `training/improvement_results_SPY.json`

---

## 5 Improvement Strategies

### 1. Alpha Threshold Tuning ⭐ **Start Here**

**Problem:** Current threshold (1e-05 = 0.001%) is too small
- Almost no NEUTRAL predictions (0.1%)
- Forces binary UP/DOWN classification on tiny movements
- Adds noise to training data

**Solution:** Test larger thresholds

```bash
python training/improve_model.py --symbol SPY --experiment alpha
```

**Thresholds to Test:**
- 1e-05 (0.001%) - Current
- 1e-04 (0.01%)
- 5e-04 (0.05%)
- **1e-03 (0.1%)** ← Recommended starting point
- 2.5e-03 (0.25%)
- 5e-03 (0.5%)
- 1e-02 (1.0%)

**Expected Impact:** +5-10% accuracy improvement

**Why This Works:**
- Filters out noise (small random movements)
- Creates clearer UP/DOWN signals
- More balanced target distribution
- Better class separation

**Example Results:**
```
Alpha = 1e-05 (0.001%):
  DOWN: 1056 (42.1%)
  NEUTRAL: 2 (0.1%)  ← Almost none!
  UP: 1449 (57.8%)
  Accuracy: 41.24%

Alpha = 1e-03 (0.1%):
  DOWN: 950 (37.9%)
  NEUTRAL: 450 (17.9%)  ← Much better!
  UP: 1107 (44.2%)
  Accuracy: 48-52% (estimated)
```

---

### 2. Hyperparameter Optimization

**Problem:** Current parameters may not be optimal
- C = 0.25 (regularization)
- degree = 2 (polynomial)

**Solution:** Grid search over parameter space

```bash
python training/improve_model.py --symbol SPY --experiment hyperparams
```

**Parameters to Test:**

**C (Constraint parameter):**
- 0.1 - High regularization (simpler model)
- **0.25** - Current
- 0.5 - Medium
- 1.0 - Low regularization (complex model)

**Degree (Polynomial):**
- **2** - Current (quadratic)
- 3 - Cubic (more complex)

**Trade-offs:**
- **Higher C**: More complex, risk overfitting
- **Lower C**: Simpler, risk underfitting
- **Higher degree**: More feature interactions, slower training

**Expected Impact:** +2-5% accuracy

---

### 3. Feature Selection ⭐ **High Impact**

**Problem:** Using all 32 features may include noise
- Some features may be redundant
- Some may have low predictive power
- Curse of dimensionality

**Solution:** Select top K most important features

```bash
python training/improve_model.py --symbol SPY --experiment features
```

**Method:** Mutual Information (MI) scoring
- Measures dependency between feature and target
- Higher score = more informative

**Features to Test:**
- 10 features (top 30%)
- 15 features (top 50%)
- **20 features** ← Recommended
- 25 features (top 80%)
- 32 features (all)

**Expected Impact:** +3-7% accuracy

**Likely Top Features:**
1. Institutional features (HP/MHP scores)
2. Window crossing return (V3)
3. Volatility measures
4. Price position relative to support/resistance
5. Recent momentum indicators

**Example Output:**
```
Top 15 Most Important Features:
  1. hp_net_score              | Score: 0.2834
  2. V3_crossing_return         | Score: 0.2156
  3. mhp_score                  | Score: 0.1892
  4. hp_support_distance        | Score: 0.1543
  5. window_volatility          | Score: 0.1287
  ...
```

---

### 4. Class Balancing

**Problem:** Class imbalance (more UP than DOWN)
- Current: 42% DOWN, 58% UP
- Model biased toward predicting UP
- Hurts accuracy on DOWN predictions

**Solution:** Resampling techniques

```bash
python training/improve_model.py --symbol SPY --experiment balancing
```

**Strategies:**

**SMOTE (Synthetic Minority Over-sampling):**
- Generates synthetic DOWN samples
- Balances classes
- Pros: More training data
- Cons: May introduce noise

**Random Under-sampling:**
- Removes excess UP samples
- Balances classes
- Pros: Clean data
- Cons: Less training data

**Hybrid (SMOTE + Undersample):**
- Best of both worlds
- SMOTE to 50/50, then undersample slightly

**Expected Impact:** +2-4% accuracy

**Trade-off:** May reduce total training samples

---

### 5. Ensemble Size Tuning

**Problem:** 100 SVMs may be overkill or too few
- More SVMs = Better stability, slower training
- Fewer SVMs = Faster, potentially less stable

**Solution:** Test different ensemble sizes

```bash
python training/improve_model.py --symbol SPY --experiment ensemble
```

**Sizes to Test:**
- 25 - Fast experimentation
- 50 - Balanced
- **100** - Current
- 200 - Maximum diversity

**Expected Impact:** +1-3% accuracy

**Trade-offs:**
- 25 SVMs: ~3s training, potentially unstable
- 50 SVMs: ~6s training, good balance
- 100 SVMs: ~12s training, very stable
- 200 SVMs: ~25s training, diminishing returns

---

## Recommended Improvement Sequence

### Phase 1: Quick Wins (30 minutes)

1. **Alpha Threshold** (Run first!)
   ```bash
   python training/improve_model.py --symbol SPY --experiment alpha
   ```
   Expected: +5-10% accuracy

2. **Feature Selection**
   ```bash
   python training/improve_model.py --symbol SPY --experiment features
   ```
   Expected: +3-7% accuracy

**Estimated SPY Performance After Phase 1: 49-58%**

### Phase 2: Fine-Tuning (1 hour)

3. **Hyperparameters**
   ```bash
   python training/improve_model.py --symbol SPY --experiment hyperparams
   ```

4. **Class Balancing**
   ```bash
   python training/improve_model.py --symbol SPY --experiment balancing
   ```

5. **Ensemble Size**
   ```bash
   python training/improve_model.py --symbol SPY --experiment ensemble
   ```

**Estimated SPY Performance After Phase 2: 52-62%**

### Phase 3: Full Pipeline (2 hours)

Run all experiments together:
```bash
python training/improve_model.py --symbol SPY --experiment all
```

This will:
- Test all combinations
- Save results to JSON
- Recommend best configuration

---

## Interpreting Results

Results saved to `training/improvement_results_SPY.json`:

```json
[
  {
    "experiment": "alpha_threshold",
    "alpha": 0.001,
    "test_accuracy": 0.52,
    "distribution": {"-1": 950, "0": 450, "1": 1107}
  },
  ...
]
```

**Key Metrics:**

- **Test Accuracy**: Primary metric (aim for 50%+)
- **Train Accuracy**: Check for overfitting (if >>test, overfitting)
- **Distribution**: Should be balanced (30-40% each class)

**Good Signs:**
- Test accuracy > 50%
- Train/test gap < 10%
- Balanced class distribution

**Bad Signs:**
- Test accuracy < 45%
- Train accuracy >> test (overfitting)
- One class dominates (>70%)

---

## Applying Best Configuration

After finding best parameters, retrain with optimal settings:

**Edit** `config/settings.py`:

```python
@dataclass
class ModelConfig:
    # ... existing fields ...

    # Updated based on experiments
    polynomial_degree: int = 2          # Or 3 if better
    constraint_param: float = 0.5       # Or whatever tested best
    alpha_threshold: float = 1e-03      # Increased from 1e-05
    ensemble_count: int = 100           # Or 50/200 if better
```

**Retrain with new config:**
```bash
python training/train_ensemble.py
```

The new models will be saved to `models/trained/SPY_ensemble.pkl`

---

## Advanced Techniques (If Still Not 50%+)

### 6. More Training Data

**Current:** 5 years daily data → 2,507 samples

**Options:**
- Download 10+ years of data
- Use real intraday data (not synthetic)
- Add more quotes per day (currently 10)

**How:**
```python
# In download_data.py, increase years_back
download_price_history(client, symbol, years_back=10)  # Instead of 5

# In train_ensemble.py, increase quotes_per_day
quotes = builder.create_synthetic_quotes(daily_data, quotes_per_day=20)  # Instead of 10
```

### 7. Different Kernel

**Current:** Polynomial (degree=2)

**Try:**
- RBF (Radial Basis Function)
- Linear
- Sigmoid

**How:** Edit `models/svm_ensemble.py`:
```python
def _create_svm(self) -> SVC:
    return SVC(
        kernel='rbf',  # Instead of 'poly'
        gamma='scale',
        C=self.constraint_param,
        class_weight='balanced'
    )
```

### 8. Feature Engineering

**Add New Features:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands position
- Volume-weighted indicators
- Time-of-day features

**How:** Add to `features/classic_features.py`

### 9. Different Model Architecture

**Try:**
- Random Forest
- Gradient Boosting (XGBoost)
- Neural Network
- Ensemble of different model types

---

## Monitoring Improvement

Track performance over time:

```bash
# Baseline
python validate_modules.py
# SPY: 41.24% accuracy

# After alpha tuning
python validate_modules.py
# SPY: 48-52% accuracy (expected)

# After all improvements
python validate_modules.py
# SPY: 52-58% accuracy (goal)
```

---

## Troubleshooting

**Problem: Accuracy gets worse**
- Solution: Alpha threshold too high, reduce it
- Or: Removed too many features, add more back

**Problem: Overfitting (train >> test)**
- Solution: Increase C (regularization)
- Or: Reduce polynomial degree
- Or: Use fewer features

**Problem: Takes too long**
- Solution: Use smaller ensemble for experiments (25-50)
- Or: Use subset of data for quick testing
- Or: Run one experiment at a time

**Problem: Out of memory**
- Solution: Reduce ensemble size
- Or: Reduce training samples
- Or: Use feature selection (fewer features)

---

## Expected Final Performance

**Conservative Estimate:**
- Alpha tuning: 41% → 47%
- Feature selection: 47% → 51%
- Hyperparameter tuning: 51% → 53%

**Optimistic Estimate:**
- All improvements combined: 41% → 58%

**Goal: 50%+ test accuracy for SPY**

At 50%+ accuracy, SPY model becomes viable for production use alongside QQQ.

---

## Quick Reference

| Improvement | Command | Time | Impact |
|-------------|---------|------|--------|
| Alpha Threshold | `--experiment alpha` | 5 min | ⭐⭐⭐ High |
| Feature Selection | `--experiment features` | 10 min | ⭐⭐⭐ High |
| Hyperparameters | `--experiment hyperparams` | 15 min | ⭐⭐ Medium |
| Class Balancing | `--experiment balancing` | 10 min | ⭐⭐ Medium |
| Ensemble Size | `--experiment ensemble` | 20 min | ⭐ Low |
| **All Combined** | `--experiment all` | 60 min | ⭐⭐⭐ Best |

---

## Next Steps

1. **Run experiments** on SPY (most needed)
2. **Apply best config** to settings.py
3. **Retrain models** with optimal parameters
4. **Validate** new performance
5. **(Optional)** Run same experiments on QQQ to push 54% → 60%+

Good luck improving your models! 🚀
