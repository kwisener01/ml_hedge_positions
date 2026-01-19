"""
Training Pipeline for SVM Ensemble
Builds training data from historical data and trains the ensemble
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient, QuoteData
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder
from models.svm_ensemble import SVMEnsemble


class TrainingDataBuilder:
    """Build training dataset from historical data"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)

    def load_historical_data(self) -> pd.DataFrame:
        """Load historical price data from CSV"""
        filepath = Path(__file__).parent.parent / f"data_local/price_history/{self.symbol}_daily.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Historical data not found: {filepath}")

        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        print(f"[OK] Loaded {len(df)} days of historical data for {self.symbol}")
        print(f"     Date range: {df.index.min()} to {df.index.max()}")

        return df

    def create_synthetic_quotes(self, daily_data: pd.DataFrame, quotes_per_day: int = 10) -> List[QuoteData]:
        """
        Create synthetic intraday quotes from daily OHLC data

        Args:
            daily_data: DataFrame with OHLC data
            quotes_per_day: Number of quotes to generate per day

        Returns:
            List of QuoteData objects
        """
        print(f"\nGenerating synthetic quotes ({quotes_per_day} per day)...")

        quotes = []

        for date, row in daily_data.iterrows():
            # Generate intraday price path
            prices = np.linspace(row['open'], row['close'], quotes_per_day)

            # Add some randomness within high/low range
            noise_range = (row['high'] - row['low']) / 4
            prices += np.random.normal(0, noise_range, quotes_per_day)

            # Clip to high/low
            prices = np.clip(prices, row['low'], row['high'])

            # Create quotes
            for i, price in enumerate(prices):
                # Synthetic bid/ask with realistic spread
                spread_pct = 0.0002  # 2 basis points
                spread = price * spread_pct
                bid = price - spread / 2
                ask = price + spread / 2

                quote = QuoteData(
                    symbol=self.symbol,
                    bid=bid,
                    ask=ask,
                    bid_size=100,
                    ask_size=100,
                    last=price,
                    volume=int(row['volume'] / quotes_per_day) if 'volume' in row else 1000000,
                    timestamp=datetime.combine(date, datetime.min.time()) + timedelta(hours=9.5 + i * 6.5 / quotes_per_day)
                )

                quotes.append(quote)

        print(f"[OK] Generated {len(quotes)} synthetic quotes")

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
                valid_samples += 1

                if feature_names is None:
                    feature_names = list(feature_dict.keys())

            except Exception as e:
                # Skip samples with feature building errors
                continue

        X = np.vstack(X_list)
        y_returns = np.array(y_returns_list)

        print(f"  Valid samples: {valid_samples}")
        print(f"  Feature dimensions: {X.shape}")
        print(f"  Return stats: mean={y_returns.mean():.6f}, std={y_returns.std():.6f}")
        print(f"  Return range: [{y_returns.min():.6f}, {y_returns.max():.6f}]")

        return X, y_returns, feature_names


def train_symbol_ensemble(
    symbol: str,
    save_model: bool = True,
    model_dir: str = None,
    quotes_per_day: int = 10
) -> SVMEnsemble:
    """
    Complete training pipeline for a symbol

    Args:
        symbol: Ticker symbol (SPY or QQQ)
        save_model: Whether to save trained model
        model_dir: Directory to save models
        quotes_per_day: Number of synthetic quotes to generate per day (10 or 100)

    Returns:
        Trained SVMEnsemble
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PIPELINE: {symbol}")
    print(f"{'='*80}")

    # Build training data
    builder = TrainingDataBuilder(symbol)
    historical_data = builder.load_historical_data()
    quotes = builder.create_synthetic_quotes(historical_data, quotes_per_day=quotes_per_day)
    X, y_returns, feature_names = builder.build_feature_dataset(quotes, lookforward_bars=5)

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
    if save_model:
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models/trained"
        else:
            model_dir = Path(model_dir)

        model_path = model_dir / f"{symbol}_ensemble.pkl"
        ensemble.save(str(model_path))

    return ensemble


def main(quotes_per_day: int = 100):
    """
    Train ensembles for all symbols

    Args:
        quotes_per_day: Number of synthetic quotes per day (default: 100 for extended data)
    """
    print("\n" + "="*80)
    print("SVM ENSEMBLE TRAINING - EXTENDED DATA")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quotes per day: {quotes_per_day}")
    print(f"Expected samples: ~{1256 * quotes_per_day} per symbol\n")

    symbols = ["SPY", "QQQ"]
    trained_ensembles = {}

    for symbol in symbols:
        try:
            ensemble = train_symbol_ensemble(symbol, save_model=True, quotes_per_day=quotes_per_day)
            trained_ensembles[symbol] = ensemble

            print(f"\n[OK] {symbol} ensemble trained successfully")

        except Exception as e:
            print(f"\n[ERROR] Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for symbol, ensemble in trained_ensembles.items():
        metrics = ensemble.training_metrics
        print(f"\n{symbol}:")
        print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")
        print(f"  Train Accuracy: {metrics.train_accuracy*100:.2f}%")
        print(f"  Training Time: {metrics.training_time:.1f}s")
        print(f"  Samples: {metrics.train_size + metrics.test_size}")
        print(f"  Individual SVM Avg: {metrics.avg_individual_accuracy*100:.2f}%")

    print(f"\n[OK] All models saved to models/trained/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SVM ensemble models")
    parser.add_argument(
        "--quotes-per-day",
        type=int,
        default=100,
        help="Number of synthetic quotes to generate per day (default: 100 for extended data)"
    )

    args = parser.parse_args()
    main(quotes_per_day=args.quotes_per_day)
