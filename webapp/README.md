# QQQ Support Trading Monitor - Web Dashboard

Real-time web monitoring dashboard for the QQQ support trading strategy (66.67% validated win rate).

## Features

- **Real-time price updates** (60-second polling)
- **Strength meter** (0-100 score with component breakdown)
- **Greek level visualization** (HP, MHP, HG levels)
- **Bayesian confidence scoring**
- **Visual alerts** when tradeable setups appear
- **Signal history** (last 24 hours)
- **Server-Sent Events** for real-time updates
- **Docker deployment** ready

## Quick Start

### 1. Setup Environment

```bash
cd webapp
cp .env.example .env
# Edit .env and add your TRADIER_API_KEY
```

### 2. Run with Docker (Recommended)

```bash
docker-compose up -d
```

Dashboard will be available at: **http://localhost:8000**

To view logs:
```bash
docker-compose logs -f
```

To stop:
```bash
docker-compose down
```

### 3. Run Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Dashboard will be available at: **http://localhost:8000**

## Configuration

Edit `.env` file to configure:

```bash
# Required
TRADIER_API_KEY=your_api_key_here

# Optional (defaults shown)
TRADIER_USE_SANDBOX=false
POLLING_INTERVAL=60          # Seconds between checks
CACHE_TTL=300                # 5-minute institutional cache
MIN_CONFIDENCE=0.55          # Minimum Bayesian confidence
MIN_STRENGTH=15              # Minimum strength score
MAX_STRENGTH=30              # Maximum strength score (moderate range)
ALERT_COOLDOWN=300           # 5 minutes between alerts
LOG_LEVEL=INFO
```

## Architecture

### Backend (FastAPI)

- **app.py** - Main FastAPI application
- **monitor_service.py** - Background monitoring loop (60-second polling)
- **strategy_engine.py** - Strategy logic wrapper
- **routes.py** - API endpoints and SSE streaming
- **state.py** - Shared state manager
- **config.py** - Configuration loader

### Frontend (Vanilla JS)

- **index.html** - Main dashboard UI
- **styles.css** - Custom styling (with Tailwind CSS)
- **sse.js** - Server-Sent Events client
- **main.js** - UI update functions

### API Endpoints

- `GET /` - Main dashboard
- `GET /api/status` - Health check
- `GET /api/current` - Current state (quote, greeks, strength)
- `GET /api/strength` - Strength score breakdown
- `GET /api/signals` - Recent signals
- `GET /api/events` - SSE stream (real-time updates)
- `GET /health` - Docker health check

### SSE Events

Real-time events pushed to dashboard:

- **initial_state** - Full state on connection
- **quote_update** - New QQQ price
- **greek_update** - Updated Greek levels
- **strength_update** - Updated strength/confidence
- **signal_alert** - Tradeable setup detected
- **monitoring_status** - Monitor active/inactive

## Entry Conditions

Signal alerts trigger when ALL conditions met:

1. **Moderate strength** (15-30 score)
2. **Support level** only (not resistance)
3. **High confidence** (> 0.55 Bayesian)
4. **Model predicts UP** (P(UP) > 0.5)

## API Rate Limits

Strategy respects Tradier API limits:

- **Quote calls**: 1/min = 60/hour (limit: 120/min)
- **Institutional data**: Cached 5 minutes = 12 refreshes/hour × 4 calls = 48/hour
- **Total**: 108 calls/hour (well within limits)

## Dashboard Sections

### 1. Current Price
- Live QQQ price
- Bid/ask spread
- Last update timestamp

### 2. Strength Meter
- Total score (0-100)
- Color-coded (red < 15, yellow 15-30, green 30+)
- Component breakdown:
  - Gamma (0-20 points)
  - Vanna (0-15 points)
  - HP (0-10 points)
  - MHP (0-20 points, weighted 2x)
  - HG (0-25 points, highest weight)
  - Overlap (0-10 points)
- Level type (support/resistance/none)
- Level source (HP/MHP/HG)

### 3. Greek Levels
- HP Support/Resistance
- MHP Support/Resistance
- HG Below/Above
- Gamma Flip level

### 4. Model Confidence
- Bayesian confidence (0-1)
- Model P(UP) probability
- Status indicator

### 5. Alert Banner
- Appears when signal detected
- Entry price
- Suggested stop (-1%)
- Suggested target (+1.5%)
- Confidence and strength
- Auto-dismisses after 30 seconds

### 6. Recent Signals
- Last 24 hours of signals
- Time, type, entry, confidence, strength, level

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Common issues:
# 1. TRADIER_API_KEY not set in .env
# 2. Model file not found
# 3. Port 8000 already in use
```

### No data updates
```bash
# Check if monitoring is active
curl http://localhost:8000/api/status

# Check logs
docker-compose logs -f

# Verify API key is valid
```

### Connection errors
- Check that port 8000 is accessible
- Verify firewall settings
- Check Docker network configuration

## Production Deployment

### Cloud Server (AWS/DigitalOcean)

1. Copy webapp directory to server
2. Configure .env with production API key
3. Run with docker-compose:
   ```bash
   docker-compose up -d
   ```
4. Set up reverse proxy (nginx) for HTTPS
5. Configure domain name

### Monitoring

- Check logs: `docker-compose logs -f`
- Health endpoint: `curl http://localhost:8000/health`
- API usage tracking in logs

## Development

### Run in dev mode with auto-reload:

```bash
cd webapp
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Test SSE connection:

```bash
curl -N http://localhost:8000/api/events
```

## Strategy Details

Based on validated backtest results:

- **Win rate**: 66.67% (36/54 trades)
- **Return**: 2.28% in 10 days
- **Profit factor**: 1.73
- **Strategy**: Support-only with moderate Greek strength
- **Time-based exits**: 10 bars (primary exit method)

## License

Part of the ml_arb_svm_spy_qqq trading system.
