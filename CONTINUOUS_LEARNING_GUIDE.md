# Continuous Learning Setup Guide

Your ML hedge positions app now includes **automatic continuous learning** that collects live data and retrains models periodically.

## How It Works

### 1. **Data Collection** (Automated)
- Every prediction is logged to CSV files in `data_collected/`
- Logs include: features, predictions, confidence scores, and signal firings
- Data persists across restarts using Railway volumes

### 2. **Scheduled Retraining** (Automated)
- Runs daily at 2:00 AM (configurable)
- Checks if enough new data collected (default: 100 samples)
- Retrains model with combined live + historical data
- Replaces old model with improved version

### 3. **Model Improvement** (Automatic)
- New model automatically used after successful retraining
- System continues learning from every market interaction
- No manual intervention required

---

## Railway Setup for Continuous Learning

### Step 1: Add Persistent Volume

Railway volumes ensure your collected data persists across deployments:

1. Go to your Railway service dashboard
2. Click **"Settings"** → **"Volumes"**
3. Click **"New Volume"**
4. Configure:
   - **Mount Path**: `/app/data_collected`
   - **Size**: 1GB (free tier)
5. Click **"Add"**

6. Add another volume for models:
   - **Mount Path**: `/app/models/trained`
   - **Size**: 1GB

### Step 2: Environment Variables (Optional Tuning)

Add these to Railway environment variables if you want to customize:

```bash
# Retraining Configuration
RETRAIN_HOUR=2                 # Hour of day (0-23) to run retraining
MIN_SAMPLES_FOR_RETRAIN=100    # Minimum samples needed
POLLING_INTERVAL=60            # Seconds between predictions (more data = faster learning)
```

### Step 3: Initial Model Upload

Upload your initial trained model (see RAILWAY_DEPLOYMENT.md):

```bash
railway run bash
cd models/trained
# Upload via wget/curl or copy from local machine
```

---

## Monitoring Continuous Learning

### Check Data Collection Status

Via Railway logs:
```bash
railway logs
```

Look for:
```
Data logger initialized: ../data_collected
Logged prediction: QQQ @ 515.23 = 1
```

### Check Retraining Status

Logs will show:
```
Retraining scheduler started
Starting model retraining...
Training samples: 150
Test Accuracy: 0.68
Model saved: /app/models/trained/QQQ_xgboost_optimized.pkl
Retraining completed successfully
```

### Access Data via Railway CLI

```bash
railway run bash

# Check collected data
ls -lh data_collected/
cat data_collected/predictions.csv | wc -l  # Count predictions

# Check model metadata
python -c "
import pickle
with open('models/trained/QQQ_xgboost_optimized.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f\"Training date: {data['training_date']}\")
    print(f\"Samples: {data['training_samples']}\")
    print(f\"Accuracy: {data['test_accuracy']:.3f}\")
"
```

---

## Data Files Structure

### `data_collected/predictions.csv`
Logs every prediction made:
- timestamp, symbol, price, prediction
- prob_up, confidence, strength_score
- level_type, level_source, signal_fired

### `data_collected/features.csv`
Complete feature vectors for each prediction:
- timestamp, symbol, features_json
- Stores both feature arrays and feature dictionaries

### `data_collected/outcomes.csv`
Records actual outcomes (future enhancement):
- Will track if predictions were correct
- Can be used for more sophisticated retraining

---

## Retraining Configuration

### Automatic (Default)
- Runs daily at 2 AM
- Requires 100+ collected samples
- Uses XGBoost model
- Combines live data with historical data

### Manual Trigger

Force retraining via Railway CLI:

```bash
railway run bash

# Run retraining manually
cd /app
python training/retrain_from_live_data.py --symbol QQQ --min-samples 50

# Or use the Python API
python -c "
from webapp.retraining_scheduler import get_scheduler
scheduler = get_scheduler()
success = scheduler.force_retrain()
print(f'Retraining: {'SUCCESS' if success else 'FAILED'}')
"
```

### Customize Schedule

Edit `webapp/app.py` line 65:

```python
retraining_scheduler = RetrainingScheduler(
    symbol='QQQ',
    retrain_hour=4,      # Change to 4 AM
    min_samples=200      # Require more data
)
```

---

## Storage Requirements

### Free Tier (2GB total volumes)
- **Models**: ~500MB per model × 2 models = 1GB
- **Collected Data**: ~10MB per 1000 predictions
- **Logs**: ~100MB over time

**Estimate**: Can run for 3-6 months before hitting limits

### When You Need More Storage

**Option 1**: Upgrade Railway plan ($5/month for 5GB)

**Option 2**: External storage (S3/Dropbox)
```python
# Periodic backup to S3
aws s3 sync data_collected/ s3://your-bucket/ml-data/
```

**Option 3**: Periodic cleanup
```bash
# Keep only last 30 days of data
find data_collected/ -name "*.csv" -mtime +30 -delete
```

---

## Performance Optimization

### Faster Learning
```bash
# More frequent predictions = more data
POLLING_INTERVAL=30  # Every 30 seconds instead of 60
```

### More Frequent Retraining
```python
# Edit webapp/app.py
retraining_scheduler = RetrainingScheduler(
    retrain_hour=2,
    check_interval=1800  # Check every 30 minutes instead of hourly
)
```

### Quality Over Quantity
```bash
# Require more samples for retraining (better model quality)
MIN_SAMPLES_FOR_RETRAIN=500
```

---

## Troubleshooting

### "Not enough data for retraining"
- **Cause**: Less than 100 predictions collected
- **Solution**: Wait longer or lower `min_samples`
- **Check**: `railway run bash` → `wc -l data_collected/predictions.csv`

### "Retraining failed"
- **Cause**: Usually missing dependencies or corrupt data
- **Check logs**: `railway logs | grep "retraining"`
- **Solution**: Check error message, may need to clear corrupted data

### "Model not found after retraining"
- **Cause**: Model file wasn't saved correctly
- **Solution**: Check volume is mounted at `/app/models/trained`
- **Verify**: `railway run bash` → `ls -lh models/trained/`

### Data Not Persisting
- **Cause**: Volume not configured
- **Solution**: Add volumes as described in Step 1
- **Verify**: Data should persist after Railway restart

---

## Advanced: Add Outcome Tracking

To improve retraining quality, track actual outcomes:

```python
# In monitor_service.py, after signal fires:
# Wait 10 bars (10 minutes), then:
entry_price = quote.mid_price
# ... wait 10 minutes ...
exit_price = new_quote.mid_price

self.data_logger.log_outcome(
    prediction_timestamp=signal_timestamp,
    symbol='QQQ',
    entry_price=entry_price,
    exit_price=exit_price,
    bars_held=10
)
```

Then update `retrain_from_live_data.py` to use actual outcomes as labels instead of model predictions.

---

## Monitoring Best Practices

1. **Check logs daily** for first week
2. **Monitor data collection rate** (should see ~1440 predictions/day with 60s polling)
3. **Track model accuracy** after each retraining
4. **Backup data periodically** if using free tier
5. **Monitor Railway storage usage** in dashboard

---

## Cost Breakdown

### Railway Free Tier
- **Cost**: $0 (with $5/month credit)
- **Storage**: 2GB volumes
- **Compute**: 500MB RAM
- **Duration**: ~3-6 months of continuous learning

### Railway Pro ($20/month)
- **Storage**: 10GB+ volumes
- **Compute**: 2GB+ RAM
- **Duration**: Unlimited continuous learning

### With External Storage (S3)
- **Railway**: $5-10/month (basic plan)
- **S3 Storage**: ~$0.50/month for 10GB
- **Total**: ~$10/month for unlimited duration

---

## Success Metrics

After 1 month of continuous learning, you should see:
- ✅ 40,000+ predictions collected
- ✅ 30+ model retraining cycles completed
- ✅ Improving test accuracy over time
- ✅ Better signal quality (higher confidence signals)
- ✅ Reduced false positives

---

## Next Steps

1. ✅ Deploy to Railway
2. ✅ Add volumes for persistence
3. ✅ Upload initial model
4. ⏳ Let it run for 1 week
5. ⏳ Check first retraining results
6. ⏳ Monitor and optimize

Your system is now **self-improving**! Every trade makes your model smarter. 🚀

---

## Questions?

- **Railway Issues**: https://railway.app/help
- **Model Performance**: Check logs for accuracy metrics after each retrain
- **Data Questions**: Use Railway CLI to inspect data files

Happy learning! 📈
