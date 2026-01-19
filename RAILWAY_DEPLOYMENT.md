# Railway.app Deployment Guide

This guide walks you through deploying your ML hedge positions app to Railway.app.

## Prerequisites

1. **Railway Account**: Sign up at https://railway.app (free tier available)
2. **GitHub Account**: Your code is already on GitHub
3. **Tradier API Key**: Get one from https://developer.tradier.com
4. **Model File**: You'll need to upload the trained model after deployment

## Step 1: Deploy to Railway

### Option A: Deploy from GitHub (Recommended)

1. Go to https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub account
5. Select repository: `kwisener01/ml_hedge_positions`
6. Railway will automatically detect the configuration and start building

### Option B: Deploy with Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Link to your GitHub repo
railway link

# Deploy
railway up
```

## Step 2: Configure Environment Variables

After deployment starts, add these environment variables in Railway dashboard:

1. Click on your service
2. Go to **"Variables"** tab
3. Add the following:

```bash
# Required
TRADIER_API_KEY=your_tradier_api_key_here

# Optional (defaults are fine for testing)
TRADIER_USE_SANDBOX=false
POLLING_INTERVAL=60
CACHE_TTL=300
MIN_CONFIDENCE=0.55
MIN_STRENGTH=15
MAX_STRENGTH=30
ALERT_COOLDOWN=300
LOG_LEVEL=INFO
MODEL_PATH=../models/trained/QQQ_xgboost_optimized.pkl
```

4. Click **"Add"** or **"Save"** for each variable

## Step 3: Add Persistent Storage (For Models)

Railway provides volumes for persistent storage:

1. In your service dashboard, go to **"Settings"**
2. Scroll to **"Volumes"**
3. Click **"New Volume"**
4. **Mount Path**: `/app/models/trained`
5. **Size**: 1GB (free tier allows up to 1GB)
6. Click **"Add"**

## Step 4: Upload Model Files

Since model files are too large for Git, you need to upload them to Railway:

### Method 1: Using Railway CLI

```bash
# Login to Railway
railway login

# Link to your project
railway link

# Shell into your container
railway run bash

# Now you're in the container - you can upload files or download them
# Exit when done
exit
```

### Method 2: Upload via URL (if models are hosted)

If you host your model files on Dropbox, Google Drive, or S3:

```bash
# Shell into Railway container
railway run bash

# Download model
cd models/trained
wget "YOUR_MODEL_URL" -O QQQ_xgboost_optimized.pkl

# Or use curl
curl -L "YOUR_MODEL_URL" -o QQQ_xgboost_optimized.pkl
```

### Method 3: Copy from Local Machine

```bash
# Using Railway CLI to copy files
railway run -- python -c "
import pickle
import base64
import sys

# Read base64 encoded model from stdin
model_data = base64.b64decode(sys.stdin.read())
with open('models/trained/QQQ_xgboost_optimized.pkl', 'wb') as f:
    f.write(model_data)
"
```

Then pipe your local model (base64 encoded):
```bash
base64 models/trained/QQQ_xgboost_optimized.pkl | railway run python -c "..."
```

## Step 5: Verify Deployment

1. **Check Deployment Status**:
   - In Railway dashboard, watch the deployment logs
   - Look for "Application startup complete"

2. **Get Your App URL**:
   - Railway will generate a URL like: `https://your-app.up.railway.app`
   - Click "Settings" → "Generate Domain" if no domain exists

3. **Test the Health Endpoint**:
```bash
curl https://your-app.up.railway.app/health
```

Should return:
```json
{
  "status": "healthy",
  "monitoring_active": false
}
```

4. **Open Dashboard**:
   - Visit `https://your-app.up.railway.app/`
   - You should see the QQQ Support Trading Monitor dashboard

## Step 6: Monitor and Troubleshoot

### View Logs
```bash
# Via CLI
railway logs

# Or in Railway dashboard: Click "Deployments" → "View Logs"
```

### Common Issues

#### 1. **Model Not Found Warning**
```
WARNING - Model not found: ../models/trained/QQQ_xgboost_optimized.pkl
```
**Solution**: Upload model file (see Step 4)

#### 2. **TRADIER_API_KEY Not Set**
```
ValueError: TRADIER_API_KEY not set
```
**Solution**: Add `TRADIER_API_KEY` in environment variables (see Step 2)

#### 3. **Port Binding Issues**
Railway automatically sets the `PORT` environment variable. The app uses `$PORT` from Railway.

#### 4. **Build Failures**
- Check build logs in Railway dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

## Railway Limitations (Free Tier)

- **$5 credit/month**: Usually enough for small apps
- **500MB RAM**: Should be sufficient for your app
- **1GB storage**: Enough for 1-2 model files
- **Sleeps after inactivity**: App may sleep if no requests for 30 minutes

If you need more resources, Railway has paid plans starting at $5/month.

## Continuous Learning & Data Storage

### Option 1: Manual Retraining (Simple)
1. Retrain models locally
2. Upload new model files to Railway (see Step 4)
3. Restart the app

### Option 2: Scheduled Retraining (Advanced)
Add a retraining script to your app:

```python
# training/retrain_job.py
import schedule
import time

def retrain_model():
    # Download latest data
    # Retrain model
    # Save updated model
    pass

schedule.every().day.at("02:00").do(retrain_model)

while True:
    schedule.run_pending()
    time.sleep(60)
```

Run as a separate Railway service.

### Option 3: External Storage
For larger datasets and models:
- **Railway + AWS S3**: Store models in S3, app downloads on startup
- **Railway + Dropbox API**: Automatically sync models
- **Railway + PostgreSQL**: Store trade signals and metrics

## Cost Optimization

1. **Use Sandbox API** during testing:
   ```
   TRADIER_USE_SANDBOX=true
   ```

2. **Increase polling interval** to reduce API calls:
   ```
   POLLING_INTERVAL=120  # 2 minutes instead of 60 seconds
   ```

3. **Monitor usage** in Railway dashboard

## Next Steps

After successful deployment:

1. **Test the monitoring**: Watch live QQQ data update
2. **Verify signals**: Check if alerts trigger correctly
3. **Log analysis**: Monitor for any errors
4. **API limits**: Ensure you're within Tradier rate limits
5. **Consider upgrades**: If needed, upgrade Railway plan

## Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Tradier API Docs**: https://documentation.tradier.com

## Security Notes

- **Never commit** your `TRADIER_API_KEY` to Git
- Use Railway's environment variables for secrets
- Keep `.env` files local only
- Consider using Railway's secret management features

---

## Quick Command Reference

```bash
# Deploy
railway up

# View logs
railway logs

# Open in browser
railway open

# Shell access
railway run bash

# Environment variables
railway variables

# Status
railway status
```

Good luck with your deployment! 🚀
