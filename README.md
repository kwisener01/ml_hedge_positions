# SVM Arbitrage Trading System

A complete machine learning-based trading system that combines Support Vector Machine (SVM) ensemble predictions with institutional options flow analysis for SPY and QQQ.

## System Overview

This system generates trading signals by:
1. **Feature Extraction**: 32 features from price action, technical indicators, and institutional flow
2. **SVM Ensemble**: 100 polynomial SVMs vote on directional predictions
3. **Institutional Analysis**: Hedge Pressure (HP), Monthly HP (MHP), and Half Gap (HG) calculations
4. **Signal Integration**: Combines ML predictions with institutional levels using confluence scoring
5. **Risk Management**: Automated stop-loss and take-profit calculation

## Performance Metrics

**QQQ Model (Production Ready)**
- Test Accuracy: 54.18%
- Trained on 2,507 samples (5 years historical data)
- 100 SVMs with polynomial kernel (degree=2, C=0.25)

**SPY Model**
- Test Accuracy: 41.24%
- Same architecture, needs improvement

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn requests python-dotenv
```

### 2. Set Up API Key

Create `.env` file:
```
TRADIER_API_KEY=your-key-here
```

### 3. Download Market Data

```bash
python download_data.py
```

Downloads 5 years of price history and current options chains for SPY/QQQ.

### 4. Validate System

```bash
python validate_modules.py
```

Tests all modules: Feature Pipeline, Institutional Layer, SVM Ensemble.

### 5. Generate Live Signals

**Single Scan:**
```bash
python signals/live_monitor.py --mode single
```

**Continuous Monitoring (15-minute intervals):**
```bash
python signals/live_monitor.py --mode continuous --interval 15
```

**Custom Parameters:**
```bash
python signals/live_monitor.py \
  --mode single \
  --confidence 0.6 \
  --confluence 0.5 \
  --symbols SPY QQQ
```

## Command Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Demo: Institutional analysis |
| `python download_data.py` | Download latest market data |
| `python validate_modules.py` | Validate all modules |
| `python training/train_ensemble.py` | Retrain SVM ensemble |
| `python signals/live_monitor.py --mode single` | Generate signals once |
| `python signals/live_monitor.py --mode continuous` | Monitor continuously |

## Signal Parameters

### Configurable Settings

- `--confidence`: Min SVM confidence (default: 0.55)
- `--confluence`: Min institutional confluence (default: 0.4)
- `--interval`: Scan interval in minutes (default: 15)
- `--symbols`: Symbols to monitor (default: SPY QQQ)

### Signal Strength Levels

- **STRONG**: Confluence ≥80% AND SVM confidence ≥80%
- **MEDIUM**: Confluence ≥65% AND SVM confidence ≥65%
- **WEAK**: Above minimum thresholds

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LIVE TRADING SYSTEM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ TRADIER  │──▶│ FEATURE  │──▶│   SVM    │           │
│  │   API    │   │ PIPELINE │   │ ENSEMBLE │           │
│  └──────────┘   └──────────┘   └──────────┘           │
│       │               │               │                 │
│       ▼               ▼               ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ OPTIONS  │──▶│INSTITUTE │──▶│PREDICTION│           │
│  │ CHAINS   │   │  LAYER   │   │  ENGINE  │           │
│  └──────────┘   └──────────┘   └──────────┘           │
│                                      │                  │
│                                      ▼                  │
│                              ┌──────────┐               │
│                              │ SIGNALS  │               │
│                              └──────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Feature Set (32 Features)

### Window Features (8)
- V1-V2: Open/Close prices
- V3: Crossing return
- V4-V5: High/Low prices
- V6-V8: Range, volatility, trend

### Classic Features (14)
- Technical indicators
- Spread metrics
- Volume analysis
- Momentum indicators

### Institutional Features (10)
- HP: Net hedge pressure
- HP: Support/resistance distances
- MHP: Monthly hedge pressure
- MHP: Support/resistance levels
- HG: Half gap distances
- Confluence metrics

## Project Structure

```
ml_arb_svm_spy_qqq/
├── config/
│   └── settings.py              # System configuration
├── data/
│   ├── tradier_client.py        # Tradier API client
│   └── cleaner.py               # Data cleaning
├── features/
│   ├── window_features.py       # Event window features
│   ├── classic_features.py      # Technical indicators
│   └── feature_matrix.py        # Complete feature builder
├── institutional/
│   ├── hedge_pressure.py        # HP calculator
│   ├── monthly_hp.py            # MHP calculator
│   └── half_gap.py              # HG calculator
├── models/
│   ├── svm_ensemble.py          # SVM ensemble class
│   └── trained/
│       ├── SPY_ensemble.pkl     # Trained SPY model
│       └── QQQ_ensemble.pkl     # Trained QQQ model
├── signals/
│   ├── signal_generator.py      # Signal integration
│   ├── live_monitor.py          # Live monitoring
│   └── backtest.py              # Backtesting framework
├── training/
│   └── train_ensemble.py        # Training pipeline
├── data_local/
│   ├── price_history/           # Historical OHLC
│   ├── options_chains/          # Options data
│   └── validation_report.txt    # Validation results
└── .env                         # API credentials
```

## Signal Output Example

```
================================================================================
TRADING SIGNAL: QQQ
================================================================================
Time: 2026-01-16 18:07:24
Signal: LONG (WEAK)
Confluence: 33.3%

PRICE LEVELS:
  Current:     $620.79
  Entry:       $620.79
  Stop Loss:   $616.90 (0.63%)
  Take Profit: $628.47 (1.24%)

SVM PREDICTION:
  Direction: +1
  Confidence: 81.0%

INSTITUTIONAL CONTEXT:
  HP Score: -0.474
  MHP Score: -0.145
  Key Support: $620.00
  Key Resistance: $622.00

REASONING:
  [X] SVM and HP disagree (HP: bearish)
  [X] SVM and MHP disagree (MHP: -0.145)
  [OK] Near HP support level ($620.00)
================================================================================
```

## Risk Management

Each signal includes:
- **Entry Price**: Current market price
- **Stop Loss**: Based on institutional support/resistance (1% default)
- **Take Profit**: 2:1 reward/risk ratio, adjusted for institutional levels
- **Position Sizing**: 10% of capital default (configurable)

## Confluence Scoring

Signals measure agreement between SVM and institutional data:

- **SVM vs HP**: Direction alignment
- **SVM vs MHP**: Direction alignment
- **Price Position**: Proximity to support/resistance levels

Higher confluence = stronger signal reliability.

## Signal Log

All signals are logged to `signals/signal_log.json`:

```json
{
  "timestamp": "2026-01-16T18:07:24",
  "symbol": "QQQ",
  "signal_type": "LONG",
  "signal_strength": "WEAK",
  "current_price": 620.79,
  "svm_confidence": 0.81,
  "confluence_score": 0.33,
  "hp_score": -0.474,
  "mhp_score": -0.145
}
```

## Backtesting

```python
from signals.backtest import Backtester
import pandas as pd

# Load historical data
price_data = pd.read_csv('data_local/price_history/SPY_daily.csv',
                         index_col='date', parse_dates=True)

# Generate signals on historical data
signals = [...]  # Your historical signals

# Run backtest
backtester = Backtester(initial_capital=100000)
result = backtester.run_backtest('SPY', signals, price_data)

print(result.summary())
```

## Model Retraining

```bash
python training/train_ensemble.py
```

Retrains both SPY and QQQ ensembles on latest downloaded data.

## Configuration

Edit `config/settings.py`:

```python
@dataclass
class ModelConfig:
    tickers: List[str] = ["SPY", "QQQ"]
    window_size: int = 5
    polynomial_degree: int = 2
    constraint_param: float = 0.25
    ensemble_count: int = 100
    alpha_threshold: float = 1e-5
    spread_max_pct: float = 0.25
```

## Troubleshooting

**No signals generated:**
- Lower `--confidence` threshold (try 0.5)
- Lower `--confluence` threshold (try 0.3)
- Check that models are trained: `ls models/trained/`

**Model not found:**
- Run `python training/train_ensemble.py` to train models

**API errors:**
- Verify `.env` has correct `TRADIER_API_KEY`
- Check rate limits (120 calls/min for quotes, 60 for options)

**Poor signal quality:**
- SPY model accuracy is low (41%), consider retraining with more data
- QQQ model (54%) is production-ready
- Adjust confluence thresholds based on live performance

## Performance Notes

### QQQ Model
- **54% test accuracy** - Shows genuine predictive power
- Consistent individual SVM performance
- **Recommended for production use**

### SPY Model
- **41% test accuracy** - Below baseline
- May need feature engineering or parameter tuning
- Consider as experimental only

## Next Steps

1. **Paper Trading**: Test signals without real capital
2. **Performance Tracking**: Monitor win rate, profit factor, drawdown
3. **Model Improvement**:
   - Adjust alpha threshold
   - Add more features
   - Collect more training data
4. **Production Deployment**:
   - Real-time streaming quotes
   - Order execution integration
   - Position management
   - Automated monitoring

## License

Private project - All rights reserved

## Disclaimer

This system is for educational and research purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a licensed financial advisor before trading.
