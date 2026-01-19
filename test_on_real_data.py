"""
Test trained XGBoost models on actual intraday data
Compares performance on synthetic vs real market data
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import model_config
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class RealDataTester:
    """Test models on actual intraday data"""

    def __init__(self, symbol: str, model_path: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = TargetCalculator()

        # Load trained model
        print(f"\nLoading model for {symbol}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.training_metrics = model_data.get('metrics', {})
        print(f"[OK] Model loaded from {model_path}")

    def load_intraday_data(self, interval: str = '5min') -> pd.DataFrame:
        """Load actual intraday data from CSV"""
        # Find the most recent intraday file
        intraday_dir = Path(__file__).parent / "data_local/intraday"
        pattern = f"{self.symbol}_{interval}_*.csv"

        files = list(intraday_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No intraday data found for {self.symbol} ({interval})")

        # Use most recent file
        filepath = sorted(files)[-1]

        print(f"\nLoading intraday data from {filepath.name}...")
        df = pd.read_csv(filepath, parse_dates=['time'])
        print(f"[OK] Loaded {len(df)} {interval} bars")
        print(f"     Date: {df['time'].min()} to {df['time'].max()}")

        return df

    def convert_to_quotes(self, intraday_df: pd.DataFrame) -> List[QuoteData]:
        """Convert intraday OHLCV data to QuoteData format"""
        print(f"\nConverting {len(intraday_df)} bars to quotes...")

        quotes = []

        for idx, row in intraday_df.iterrows():
            # Use close price as the quote price
            price = row['close']

            # Estimate bid/ask from high/low
            # Assume spread is roughly 1/4 of the bar range
            bar_range = row['high'] - row['low']
            spread = max(bar_range / 4, price * 0.0002)  # At least 2 bps

            bid = price - spread / 2
            ask = price + spread / 2

            quote = QuoteData(
                symbol=self.symbol,
                bid=bid,
                ask=ask,
                bid_size=100,  # Placeholder
                ask_size=100,  # Placeholder
                last=price,
                volume=int(row['volume']),
                timestamp=row['time']
            )

            quotes.append(quote)

        print(f"[OK] Created {len(quotes)} quotes")
        return quotes

    def build_test_dataset(
        self,
        quotes: List[QuoteData],
        lookforward_bars: int = 5
    ) -> tuple:
        """Build test dataset from real quotes"""
        print(f"\nBuilding test dataset...")
        print(f"  Window size: {model_config.window_size}")
        print(f"  Lookforward: {lookforward_bars} bars")

        # Clean quotes
        cleaned_quotes = self.cleaner.clean_quotes(quotes)
        print(f"  Cleaned: {len(cleaned_quotes)}/{len(quotes)} quotes ({len(cleaned_quotes)/len(quotes)*100:.1f}%)")

        # Build non-overlapping windows
        window_builder = EventWindowBuilder()
        windows = []
        window_end_quotes = []

        for quote in cleaned_quotes:
            window = window_builder.add_event(quote)
            if window:
                windows.append(window)
                window_end_quotes.append(quote)

        print(f"  Windows created: {len(windows)}")

        # Build features
        X_list = []
        y_list = []
        feature_names = None
        valid_samples = 0

        for i in range(len(windows)):
            # Need enough future data for target calculation
            if i + lookforward_bars >= len(window_end_quotes):
                break

            # Get current and future quotes
            current_quote = window_end_quotes[i]
            future_quotes = window_end_quotes[i+1:i+1+lookforward_bars]

            if len(future_quotes) < lookforward_bars:
                continue

            # Build features
            try:
                feature_array, feature_dict = self.feature_builder.build_feature_vector(
                    self.symbol,
                    windows[i],
                    current_quote
                )

                # Calculate target
                target = self.target_calc.calculate_target(current_quote.mid_price, future_quotes)
                target_encoded = self.target_calc.encode_target(target)

                X_list.append(feature_array)
                y_list.append(target_encoded)
                valid_samples += 1

                if feature_names is None:
                    feature_names = list(feature_dict.keys())

            except Exception as e:
                # Skip samples with errors
                continue

        X = np.vstack(X_list) if X_list else np.array([])
        y = np.array(y_list)

        print(f"  Valid samples: {valid_samples}")
        print(f"  Feature dimensions: {X.shape}")

        # Print target distribution
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Target distribution:")
            for label, count in zip(unique, counts):
                label_name = self.target_calc.decode_target(label)
                print(f"    {label_name.upper():<8} ({label}): {count} ({count/len(y)*100:.1f}%)")

        return X, y, feature_names

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model on real data"""
        print(f"\n{'='*80}")
        print(f"EVALUATING ON REAL DATA: {self.symbol}")
        print(f"{'='*80}")

        # Make predictions
        y_pred = self.model.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)

        print(f"\nTest samples: {len(X)}")
        print(f"Features: {X.shape[1]}")

        print(f"\n{'='*80}")
        print(f"REAL DATA ACCURACY: {accuracy*100:.2f}%")
        print(f"{'='*80}")

        # Compare to training metrics
        if self.training_metrics:
            train_acc = self.training_metrics.get('test_accuracy', 0)
            print(f"\nComparison:")
            print(f"  Synthetic data (test): {train_acc*100:.2f}%")
            print(f"  Real data (actual):    {accuracy*100:.2f}%")
            print(f"  Difference:            {(accuracy - train_acc)*100:+.2f}%")

        # Classification report
        print(f"\nClassification Report:")
        target_names = ['DOWN', 'NEUTRAL', 'UP']
        print(classification_report(y, y_pred, target_names=target_names))

        # Confusion matrix
        print(f"Confusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(f"              Predicted")
        print(f"              DOWN  NEUTRAL  UP")
        for i, row_label in enumerate(['DOWN', 'NEUTRAL', 'UP']):
            print(f"  Actual {row_label:<8} {cm[i][0]:4d}  {cm[i][1]:7d}  {cm[i][2]:3d}")

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y,
            'confusion_matrix': cm
        }


def test_all_symbols(interval: str = '5min'):
    """Test all trained models on real intraday data"""
    print("\n" + "="*80)
    print("TESTING MODELS ON REAL INTRADAY DATA")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {interval}\n")

    symbols = ["SPY", "QQQ"]
    results = {}

    for symbol in symbols:
        try:
            # Find model file
            model_dir = Path(__file__).parent / "models/trained"
            model_path = model_dir / f"{symbol}_xgboost.pkl"

            if not model_path.exists():
                print(f"\n[SKIP] Model not found for {symbol}: {model_path}")
                continue

            # Test on real data
            tester = RealDataTester(symbol, str(model_path))
            intraday_df = tester.load_intraday_data(interval=interval)
            quotes = tester.convert_to_quotes(intraday_df)
            X, y, feature_names = tester.build_test_dataset(quotes, lookforward_bars=5)

            if len(X) == 0:
                print(f"\n[ERROR] No valid samples for {symbol}")
                continue

            metrics = tester.evaluate(X, y)
            results[symbol] = metrics

        except Exception as e:
            print(f"\n[ERROR] Failed to test {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: SYNTHETIC vs REAL DATA")
    print("="*80)

    for symbol in symbols:
        if symbol in results:
            print(f"\n{symbol}:")
            print(f"  Real Data Accuracy: {results[symbol]['accuracy']*100:.2f}%")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test models on real intraday data")
    parser.add_argument(
        "--interval",
        type=str,
        default="5min",
        choices=["1min", "5min", "15min"],
        help="Intraday interval to test (default: 5min)"
    )

    args = parser.parse_args()
    test_all_symbols(interval=args.interval)
