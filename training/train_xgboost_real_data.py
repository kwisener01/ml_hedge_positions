"""
XGBoost Training on REAL Intraday Data
Uses actual 1-minute market data instead of synthetic quotes
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import model_config
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb


class RealDataXGBoostTrainer:
    """XGBoost trainer using real intraday data"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = TargetCalculator()
        self.model = None
        self.training_metrics = {}

    def load_real_intraday_data(self, interval: str = '1min') -> pd.DataFrame:
        """Load real intraday data from CSV"""
        intraday_dir = Path(__file__).parent.parent / "data_local/intraday"
        pattern = f"{self.symbol}_{interval}_*.csv"

        files = list(intraday_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {interval} data found for {self.symbol}")

        # Use most recent file
        filepath = sorted(files)[-1]

        print(f"\n[OK] Loading {filepath.name}...")
        df = pd.read_csv(filepath, parse_dates=['time'])
        print(f"     Loaded {len(df)} {interval} bars")
        print(f"     Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def convert_to_quotes(self, intraday_df: pd.DataFrame) -> List[QuoteData]:
        """Convert intraday OHLCV data to QuoteData format"""
        print(f"\nConverting {len(intraday_df)} bars to quotes...")

        quotes = []

        for idx, row in intraday_df.iterrows():
            price = row['close']

            # Estimate bid/ask from high/low
            bar_range = row['high'] - row['low']
            spread = max(bar_range / 4, price * 0.0002)  # At least 2 bps

            bid = price - spread / 2
            ask = price + spread / 2

            quote = QuoteData(
                symbol=self.symbol,
                bid=bid,
                ask=ask,
                bid_size=100,
                ask_size=100,
                last=price,
                volume=int(row['volume']),
                timestamp=row['time']
            )

            quotes.append(quote)

        print(f"[OK] Created {len(quotes)} quotes")
        return quotes

    def build_feature_dataset(
        self,
        quotes: List[QuoteData],
        lookforward_bars: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build feature matrix and targets from real quotes"""
        print(f"\nBuilding feature dataset...")
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
            if i + lookforward_bars >= len(window_end_quotes):
                break

            current_quote = window_end_quotes[i]
            future_quotes = window_end_quotes[i+1:i+1+lookforward_bars]

            if len(future_quotes) < lookforward_bars:
                continue

            try:
                feature_array, feature_dict = self.feature_builder.build_feature_vector(
                    self.symbol,
                    windows[i],
                    current_quote
                )

                target = self.target_calc.calculate_target(current_quote.mid_price, future_quotes)
                target_encoded = self.target_calc.encode_target(target)

                X_list.append(feature_array)
                y_list.append(target_encoded)
                valid_samples += 1

                if feature_names is None:
                    feature_names = list(feature_dict.keys())

            except Exception as e:
                continue

        X = np.vstack(X_list) if X_list else np.array([])
        y = np.array(y_list)

        print(f"  Valid samples: {valid_samples}")
        print(f"  Feature dimensions: {X.shape}")

        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Target distribution:")
            for label, count in zip(unique, counts):
                label_name = self.target_calc.decode_target(label)
                print(f"    {label_name.upper():<12} ({label}): {count:>5} ({count/len(y)*100:>5.1f}%)")

        return X, y, feature_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2
    ) -> Dict:
        """Train XGBoost model on real data"""
        print(f"\n{'='*80}")
        print(f"TRAINING XGBOOST ON REAL DATA: {self.symbol}")
        print(f"{'='*80}")

        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")

        # Target distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nTarget distribution:")
        for label, count in zip(unique, counts):
            label_name = self.target_calc.decode_target(label)
            print(f"  {label_name.upper():<12} ({label:>2}): {count:>5} ({count/len(y)*100:>5.1f}%)")

        # Time-based split (not random) - more realistic for time series
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\nTime-based train/test split:")
        print(f"  Train size: {len(X_train)} (first {(1-test_size)*100:.0f}%)")
        print(f"  Test size: {len(X_test)} (last {test_size*100:.0f}%)")

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

        print(f"\nXGBoost hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train
        print(f"\nTraining XGBoost on REAL market data...")
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
        print("TRAINING COMPLETE - REAL DATA RESULTS")
        print(f"{'='*80}")
        print(f"Training time: {training_time:.1f}s")
        print(f"Train accuracy: {train_acc*100:.2f}%")
        print(f"Test accuracy: {test_acc*100:.2f}%")
        print(f"Train-test gap: {(train_acc - test_acc)*100:.2f}%")

        # Detailed test set metrics
        print(f"\nDetailed Test Set Results:")
        print(classification_report(
            y_test, test_pred,
            target_names=['DOWN', 'NEUTRAL', 'UP'],
            digits=4
        ))

        print(f"Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"              Predicted")
        print(f"              DOWN  NEUTRAL  UP")
        for i, row_label in enumerate(['DOWN', 'NEUTRAL', 'UP']):
            print(f"  Actual {row_label:<8} {cm[i][0]:4d}  {cm[i][1]:7d}  {cm[i][2]:3d}")

        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 15 Most Important Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

        # Store metrics
        self.training_metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_importance': importance_df,
            'confusion_matrix': cm,
            'data_source': 'real_intraday'
        }

        return self.training_metrics

    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'symbol': self.symbol,
                'metrics': self.training_metrics,
                'data_source': 'real_intraday_1min'
            }, f)
        print(f"\n[OK] Model saved to {filepath}")


def train_symbol_real_data(
    symbol: str,
    interval: str = '1min',
    save_model: bool = True
) -> RealDataXGBoostTrainer:
    """
    Train XGBoost on real intraday data

    Args:
        symbol: SPY or QQQ
        interval: Data interval (1min, 5min, etc.)
        save_model: Whether to save the model

    Returns:
        Trained model
    """
    print(f"\n{'='*80}")
    print(f"XGBOOST TRAINING ON REAL DATA: {symbol}")
    print(f"{'='*80}")

    trainer = RealDataXGBoostTrainer(symbol)
    intraday_df = trainer.load_real_intraday_data(interval=interval)
    quotes = trainer.convert_to_quotes(intraday_df)
    X, y, feature_names = trainer.build_feature_dataset(quotes, lookforward_bars=5)

    if len(X) == 0:
        print(f"[ERROR] No valid samples for {symbol}")
        return None

    metrics = trainer.train(X=X, y=y, feature_names=feature_names, test_size=0.2)

    if save_model:
        model_dir = Path(__file__).parent.parent / "models/trained"
        model_path = model_dir / f"{symbol}_xgboost_real.pkl"
        trainer.save(str(model_path))

    return trainer


def main():
    """Train XGBoost models on real data for all symbols"""
    print("\n" + "="*80)
    print("XGBOOST TRAINING - REAL INTRADAY DATA")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    symbols = ["SPY", "QQQ"]
    interval = '1min'
    trained_models = {}

    for symbol in symbols:
        try:
            trainer = train_symbol_real_data(symbol, interval=interval, save_model=True)

            if trainer:
                trained_models[symbol] = trainer
                print(f"\n[OK] {symbol} XGBoost model trained successfully on real data")

        except Exception as e:
            print(f"\n[ERROR] Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - REAL DATA TRAINING")
    print("="*80)

    for symbol, trainer in trained_models.items():
        metrics = trainer.training_metrics
        print(f"\n{symbol}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  Training Time: {metrics['training_time']:.1f}s")
        print(f"  Total Samples: {metrics['train_size'] + metrics['test_size']}")
        print(f"  Data Source: {metrics['data_source']}")

    print(f"\n[OK] Models saved to models/trained/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
