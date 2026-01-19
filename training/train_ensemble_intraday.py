"""
SVM Ensemble Training - Real Intraday Data
Uses actual 1-minute or 5-minute bars instead of synthetic quotes
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import model_config
from models.svm_ensemble import SVMEnsemble
from features.feature_matrix import FeatureMatrixBuilder
from data.cleaner import DataCleaner, EventWindowBuilder
from data.tradier_client import TradierClient, QuoteData


class IntradayTrainingBuilder:
    """Build training dataset from real intraday bars"""

    def __init__(self, symbol: str, interval: str = '5min'):
        self.symbol = symbol
        self.interval = interval
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)

    def load_intraday_data(self) -> pd.DataFrame:
        """Load real intraday data from CSV"""
        data_dir = Path(__file__).parent.parent / "data_local/intraday"

        # Find the most recent file
        pattern = f"{self.symbol}_{self.interval}_*.csv"
        files = list(data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No intraday data found for {self.symbol} {self.interval}")

        # Use the most recent file
        latest_file = sorted(files)[-1]
        print(f"[OK] Loading {latest_file.name}")

        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        print(f"     {len(df)} bars from {df.index.min()} to {df.index.max()}")

        return df

    def bars_to_quotes(self, df: pd.DataFrame) -> List[QuoteData]:
        """
        Convert OHLCV bars to QuoteData format

        Uses:
        - low as bid proxy
        - high as ask proxy
        - close as last price
        """
        print(f"\nConverting {len(df)} bars to quote format...")

        quotes = []
        for idx, row in df.iterrows():
            # Use OHLC to create realistic quote
            mid = row.get('close', (row.get('high') + row.get('low')) / 2)
            spread = (row.get('high') - row.get('low')) / 2

            quote = QuoteData(
                symbol=self.symbol,
                bid=row.get('low', mid - spread),
                ask=row.get('high', mid + spread),
                bid_size=int(row.get('volume', 0) // 2),
                ask_size=int(row.get('volume', 0) // 2),
                last=row.get('close', mid),
                volume=int(row.get('volume', 0)),
                timestamp=idx
            )
            quotes.append(quote)

        print(f"[OK] Converted to {len(quotes)} quotes")
        return quotes

    def build_feature_dataset(
        self,
        quotes: List[QuoteData],
        lookforward_bars: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build feature matrix and target returns

        Args:
            quotes: List of QuoteData
            lookforward_bars: Bars to look forward for return calculation

        Returns:
            (X, y_returns, feature_names)
        """
        print(f"\nBuilding feature dataset...")
        print(f"  Window size: {model_config.window_size}")
        print(f"  Lookforward: {lookforward_bars} bars")

        # Clean quotes
        cleaned_quotes = self.cleaner.clean_quotes(quotes)
        print(f"  Cleaned: {len(cleaned_quotes)}/{len(quotes)} quotes ({len(cleaned_quotes)/len(quotes)*100:.1f}%)")

        # Build windows
        window_builder = EventWindowBuilder()
        windows = []
        window_end_quotes = []

        for quote in cleaned_quotes:
            window = window_builder.add_event(quote)
            if window:
                windows.append(window)
                window_end_quotes.append(quote)

        print(f"  Windows created: {len(windows)}")

        # Build features and compute forward returns
        X_list = []
        y_returns_list = []
        feature_names = None

        valid_samples = 0

        for i in range(len(windows)):
            # Need enough future data for return calculation
            if i + lookforward_bars >= len(window_end_quotes):
                break

            # Get current and future prices
            current_price = window_end_quotes[i].mid_price
            future_price = window_end_quotes[i + lookforward_bars].mid_price

            # Compute forward return
            forward_return = (future_price - current_price) / current_price

            # Build features
            try:
                feature_array, feature_dict = self.feature_builder.build_feature_vector(
                    self.symbol,
                    windows[i],
                    window_end_quotes[i]
                )

                X_list.append(feature_array)
                y_returns_list.append(forward_return)

                if feature_names is None:
                    feature_names = list(feature_dict.keys())

                valid_samples += 1

            except Exception as e:
                print(f"  [WARN] Failed to build features for window {i}: {e}")
                continue

        print(f"  Valid samples: {valid_samples}")

        X = np.array(X_list)
        y_returns = np.array(y_returns_list)

        print(f"  Feature dimensions: {X.shape}")
        print(f"  Return stats: mean={y_returns.mean():.6f}, std={y_returns.std():.6f}")
        print(f"  Return range: [{y_returns.min():.6f}, {y_returns.max():.6f}]")

        return X, y_returns, feature_names


def train_intraday_ensemble(
    symbol: str,
    interval: str = '5min',
    lookforward_bars: int = 5
) -> SVMEnsemble:
    """
    Train ensemble on real intraday data

    Args:
        symbol: Ticker symbol
        interval: Bar interval ('1min', '5min', '15min')
        lookforward_bars: Prediction horizon in bars

    Returns:
        Trained ensemble
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PIPELINE: {symbol} ({interval} bars)")
    print(f"{'='*80}")

    # Build training data
    builder = IntradayTrainingBuilder(symbol, interval)
    df = builder.load_intraday_data()
    quotes = builder.bars_to_quotes(df)
    X, y_returns, feature_names = builder.build_feature_dataset(quotes, lookforward_bars)

    # Train ensemble
    ensemble = SVMEnsemble()
    metrics = ensemble.train(
        X=X,
        y_returns=y_returns,
        feature_names=feature_names,
        symbol=symbol,
        test_size=0.2,
        subset_fraction=0.8,
        verbose=True
    )

    # Save model
    output_dir = Path(__file__).parent.parent / "models/trained"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / f"{symbol}_ensemble_intraday_{interval}.pkl"
    ensemble.save(str(model_file))

    print(f"\n[OK] {symbol} ensemble trained successfully")

    return ensemble


def main():
    """Train ensembles on real intraday data"""
    print("\n" + "="*80)
    print("SVM ENSEMBLE TRAINING - REAL INTRADAY DATA")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    symbols = model_config.tickers
    interval = '5min'  # 5-minute bars balance data quantity and prediction horizon
    lookforward_bars = 5  # Predict 5 bars ahead (25 minutes for 5min bars)

    results = {}

    for symbol in symbols:
        try:
            ensemble = train_intraday_ensemble(
                symbol=symbol,
                interval=interval,
                lookforward_bars=lookforward_bars
            )

            results[symbol] = ensemble.training_metrics

        except Exception as e:
            print(f"\n[ERROR] Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY - REAL INTRADAY DATA")
    print("="*80)
    print()

    for symbol, metrics in results.items():
        print(f"{symbol}:")
        print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")
        print(f"  Train Accuracy: {metrics.train_accuracy*100:.2f}%")
        print(f"  Training Time: {metrics.training_time:.1f}s")
        print(f"  Samples: {metrics.train_size + metrics.test_size}")
        print(f"  Individual SVM Avg: {metrics.avg_individual_accuracy*100:.2f}%")
        print()

    print(f"[OK] All models saved to models/trained/")
    print(f"Interval: {interval} ({lookforward_bars} bars = {lookforward_bars * {'1min': 1, '5min': 5, '15min': 15}[interval]} min prediction)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
