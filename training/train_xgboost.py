"""
XGBoost Training Pipeline
Alternative to SVM ensemble using gradient boosting
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient, QuoteData
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


class XGBoostTrainer:
    """XGBoost model trainer for market prediction"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = TargetCalculator()
        self.model = None
        self.training_metrics = {}

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
        """Create synthetic intraday quotes from daily OHLC data"""
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
        """Build feature matrix and targets (non-overlapping windows like SVM)"""
        print(f"\nBuilding feature dataset...")
        print(f"  Window size: {model_config.window_size}")
        print(f"  Lookforward: {lookforward_bars} bars")

        # Clean quotes
        cleaned_quotes = self.cleaner.clean_quotes(quotes)
        print(f"  Cleaned: {len(cleaned_quotes)}/{len(quotes)} quotes ({len(cleaned_quotes)/len(quotes)*100:.1f}%)")

        # Build NON-OVERLAPPING windows (same as SVM)
        window_builder = EventWindowBuilder()
        windows = []
        window_end_quotes = []

        for quote in cleaned_quotes:
            window = window_builder.add_event(quote)
            if window:
                windows.append(window)
                window_end_quotes.append(quote)

        print(f"  Windows created: {len(windows)}")

        # Build features (same approach as SVM)
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
                # Skip samples with feature building errors
                continue

        X = np.vstack(X_list) if X_list else np.array([])
        y = np.array(y_list)

        print(f"  Valid samples: {valid_samples}")
        print(f"  Feature dimensions: {X.shape}")

        # Print target statistics
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Target distribution:")
            for label, count in zip(unique, counts):
                label_name = self.target_calc.decode_target(label)
                print(f"    {label_name.upper():<8} ({label}): {count} ({count/len(y)*100:.1f}%)")

        return X, y, feature_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2
    ) -> Dict:
        """
        Train XGBoost model

        Args:
            X: Feature matrix
            y: Target labels (encoded)
            feature_names: List of feature names
            test_size: Test set proportion

        Returns:
            Dictionary of training metrics
        """
        print(f"\n{'='*80}")
        print(f"Training XGBoost Model: {self.symbol}")
        print(f"{'='*80}")

        # Print dataset info
        print(f"Samples: {len(X)}")
        print(f"Features: {X.shape[1]}")

        # Target distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nTarget distribution:")
        for label, count in zip(unique, counts):
            label_name = self.target_calc.decode_target(label)
            print(f"  {label_name.upper():<8} ({label:>2}): {count:>5} ({count/len(y)*100:>5.1f}%)")

        # Train/test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
        except ValueError:
            # If stratification fails (e.g., class has too few samples)
            print("\n[WARNING] Stratification failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")

        # XGBoost parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }

        print(f"\nXGBoost parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train model
        print(f"\nTraining XGBoost...")
        start_time = datetime.now()

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        print(f"\nEvaluating model...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Training time: {training_time:.1f}s")
        print(f"Train accuracy: {train_acc*100:.2f}%")
        print(f"Test accuracy: {test_acc*100:.2f}%")

        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")

        # Store metrics
        self.training_metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_importance': importance_df
        }

        return self.training_metrics

    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'symbol': self.symbol,
                'metrics': self.training_metrics
            }, f)
        print(f"[OK] Model saved to {filepath}")


def train_symbol_xgboost(
    symbol: str,
    save_model: bool = True,
    model_dir: str = None,
    quotes_per_day: int = 10
) -> XGBoostTrainer:
    """
    Complete XGBoost training pipeline for a symbol

    Args:
        symbol: Ticker symbol (SPY or QQQ)
        save_model: Whether to save trained model
        model_dir: Directory to save models
        quotes_per_day: Number of synthetic quotes per day

    Returns:
        Trained XGBoostTrainer
    """
    print(f"\n{'='*80}")
    print(f"XGBOOST TRAINING PIPELINE: {symbol}")
    print(f"{'='*80}")

    # Build training data
    trainer = XGBoostTrainer(symbol)
    historical_data = trainer.load_historical_data()
    quotes = trainer.create_synthetic_quotes(historical_data, quotes_per_day=quotes_per_day)
    X, y, feature_names = trainer.build_feature_dataset(quotes, lookforward_bars=5)

    # Train model
    metrics = trainer.train(X=X, y=y, feature_names=feature_names, test_size=0.2)

    # Save model
    if save_model:
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models/trained"
        else:
            model_dir = Path(model_dir)

        model_path = model_dir / f"{symbol}_xgboost.pkl"
        trainer.save(str(model_path))

    return trainer


def main(quotes_per_day: int = 10):
    """Train XGBoost models for all symbols"""
    print("\n" + "="*80)
    print("XGBOOST MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quotes per day: {quotes_per_day}\n")

    symbols = ["SPY", "QQQ"]
    trained_models = {}

    for symbol in symbols:
        try:
            trainer = train_symbol_xgboost(symbol, save_model=True, quotes_per_day=quotes_per_day)
            trained_models[symbol] = trainer

            print(f"\n[OK] {symbol} XGBoost model trained successfully")

        except Exception as e:
            print(f"\n[ERROR] Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for symbol, trainer in trained_models.items():
        metrics = trainer.training_metrics
        print(f"\n{symbol}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  Training Time: {metrics['training_time']:.1f}s")
        print(f"  Samples: {metrics['train_size'] + metrics['test_size']}")

    print(f"\n[OK] All models saved to models/trained/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost models")
    parser.add_argument(
        "--quotes-per-day",
        type=int,
        default=10,
        help="Number of synthetic quotes per day (default: 10)"
    )

    args = parser.parse_args()
    main(quotes_per_day=args.quotes_per_day)
