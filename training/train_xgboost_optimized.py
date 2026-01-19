"""
Optimized XGBoost Training for Small Real Data
Implements quick fixes:
1. Binary classification (UP/DOWN only)
2. Reduced model complexity
3. Feature selection (top 15)
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


class BinaryTargetCalculator:
    """Binary target calculator - UP or DOWN only (no NEUTRAL)"""

    def calculate_target(self, current_price: float, future_quotes: List[QuoteData]) -> str:
        """
        Calculate binary target based on future price movement

        Args:
            current_price: Current mid price
            future_quotes: Future quotes for lookforward period

        Returns:
            'up' or 'down'
        """
        if not future_quotes:
            return 'down'  # Default

        future_price = future_quotes[-1].mid_price

        # Simple: up if future > current, down otherwise
        return 'up' if future_price > current_price else 'down'

    def encode_target(self, target: str) -> int:
        """Encode target as integer: 0=down, 1=up"""
        return 1 if target == 'up' else 0

    def decode_target(self, encoded: int) -> str:
        """Decode integer target to string"""
        return 'up' if encoded == 1 else 'down'


class OptimizedXGBoostTrainer:
    """XGBoost trainer with optimizations for small datasets"""

    def __init__(self, symbol: str, top_n_features: int = 15):
        self.symbol = symbol
        self.top_n_features = top_n_features
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = BinaryTargetCalculator()  # Binary targets
        self.model = None
        self.training_metrics = {}
        self.selected_features = None

    def load_real_intraday_data(self, interval: str = '1min') -> pd.DataFrame:
        """Load real intraday data from CSV"""
        intraday_dir = Path(__file__).parent.parent / "data_local/intraday"
        pattern = f"{self.symbol}_{interval}_*.csv"

        files = list(intraday_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {interval} data found for {self.symbol}")

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
            bar_range = row['high'] - row['low']
            spread = max(bar_range / 4, price * 0.0002)

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
        """Build feature matrix and binary targets from real quotes"""
        print(f"\nBuilding feature dataset (BINARY classification)...")
        print(f"  Window size: {model_config.window_size}")
        print(f"  Lookforward: {lookforward_bars} bars")
        print(f"  Target classes: UP, DOWN (no NEUTRAL)")

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

                # Binary target
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

    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Select top N most important features using a quick XGBoost fit

        Args:
            X: Full feature matrix
            y: Target labels
            feature_names: All feature names

        Returns:
            Reduced feature matrix and selected feature names
        """
        print(f"\nPerforming feature selection...")
        print(f"  Current features: {X.shape[1]} (feature matrix)")
        print(f"  Feature names provided: {len(feature_names)}")

        # Trim feature_names to match actual X shape (in case of mismatch)
        actual_feature_names = feature_names[:X.shape[1]]

        # Adjust top_n if we have fewer features available
        actual_top_n = min(self.top_n_features, X.shape[1])
        print(f"  Target features: {actual_top_n}")

        # Quick fit to get feature importance
        quick_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=3,
            n_estimators=20,
            random_state=42,
            tree_method='hist'
        )

        quick_model.fit(X, y, verbose=False)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': actual_feature_names,
            'importance': quick_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top N (adjusted for available features)
        top_features = importance_df.head(actual_top_n)['feature'].tolist()
        top_indices = [actual_feature_names.index(f) for f in top_features]

        X_reduced = X[:, top_indices]

        print(f"\n  Selected top {actual_top_n} features:")
        for i, (idx, row) in enumerate(importance_df.head(actual_top_n).iterrows()):
            print(f"    {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

        self.selected_features = top_features

        return X_reduced, top_features

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2
    ) -> Dict:
        """Train optimized XGBoost model on real data"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZED XGBOOST TRAINING: {self.symbol}")
        print(f"{'='*80}")

        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")

        # Target distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nTarget distribution:")
        for label, count in zip(unique, counts):
            label_name = self.target_calc.decode_target(label)
            print(f"  {label_name.upper():<12} ({label:>2}): {count:>5} ({count/len(y)*100:>5.1f}%)")

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\nTime-based train/test split:")
        print(f"  Train size: {len(X_train)} (first {(1-test_size)*100:.0f}%)")
        print(f"  Test size: {len(X_test)} (last {test_size*100:.0f}%)")

        # OPTIMIZED parameters for small dataset
        params = {
            'objective': 'binary:logistic',  # Binary classification
            'max_depth': 3,                   # Reduced from 6 (shallower trees)
            'learning_rate': 0.05,            # Reduced from 0.1 (slower learning)
            'n_estimators': 20,               # Reduced from 100 (fewer trees)
            'subsample': 0.6,                 # Reduced from 0.8 (more randomness)
            'colsample_bytree': 0.6,          # Reduced from 0.8 (more randomness)
            'reg_alpha': 1.0,                 # Increased from 0.1 (stronger L1)
            'reg_lambda': 5.0,                # Increased from 1.0 (stronger L2)
            'min_child_weight': 5,            # NEW: require more samples per leaf
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }

        print(f"\nOptimized XGBoost hyperparameters (for small dataset):")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Train
        print(f"\nTraining optimized XGBoost on REAL market data...")
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
        print("OPTIMIZED TRAINING COMPLETE - REAL DATA RESULTS")
        print(f"{'='*80}")
        print(f"Training time: {training_time:.1f}s")
        print(f"Train accuracy: {train_acc*100:.2f}%")
        print(f"Test accuracy: {test_acc*100:.2f}%")
        print(f"Train-test gap: {(train_acc - test_acc)*100:.2f}%")

        # Detailed test set metrics
        print(f"\nDetailed Test Set Results:")
        print(classification_report(
            y_test, test_pred,
            target_names=['DOWN', 'UP'],
            digits=4
        ))

        print(f"Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"              Predicted")
        print(f"              DOWN    UP")
        for i, row_label in enumerate(['DOWN', 'UP']):
            print(f"  Actual {row_label:<8} {cm[i][0]:4d}  {cm[i][1]:4d}")

        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop {min(10, len(feature_names))} Most Important Features:")
        for i, (idx, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

        # Store metrics
        self.training_metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_importance': importance_df,
            'confusion_matrix': cm,
            'data_source': 'real_intraday_1min',
            'optimizations': 'binary_classification + reduced_complexity + feature_selection',
            'num_features': len(feature_names)
        }

        return self.training_metrics

    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'symbol': self.symbol,
                'metrics': self.training_metrics,
                'selected_features': self.selected_features,
                'data_source': 'real_intraday_1min_optimized'
            }, f)
        print(f"\n[OK] Model saved to {filepath}")


def train_symbol_optimized(
    symbol: str,
    interval: str = '1min',
    top_n_features: int = 15,
    save_model: bool = True
) -> OptimizedXGBoostTrainer:
    """
    Train optimized XGBoost on real intraday data

    Args:
        symbol: SPY or QQQ
        interval: Data interval (1min, 5min, etc.)
        top_n_features: Number of features to select
        save_model: Whether to save the model

    Returns:
        Trained model
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZED XGBOOST TRAINING: {symbol}")
    print(f"{'='*80}")

    trainer = OptimizedXGBoostTrainer(symbol, top_n_features=top_n_features)
    intraday_df = trainer.load_real_intraday_data(interval=interval)
    quotes = trainer.convert_to_quotes(intraday_df)
    X, y, feature_names = trainer.build_feature_dataset(quotes, lookforward_bars=5)

    if len(X) == 0:
        print(f"[ERROR] No valid samples for {symbol}")
        return None

    # Feature selection
    X_reduced, selected_features = trainer.select_features(X, y, feature_names)

    # Train on reduced features
    metrics = trainer.train(X=X_reduced, y=y, feature_names=selected_features, test_size=0.2)

    if save_model:
        model_dir = Path(__file__).parent.parent / "models/trained"
        model_path = model_dir / f"{symbol}_xgboost_optimized.pkl"
        trainer.save(str(model_path))

    return trainer


def main():
    """Train optimized XGBoost models on real data for all symbols"""
    print("\n" + "="*80)
    print("OPTIMIZED XGBOOST TRAINING - REAL INTRADAY DATA")
    print("="*80)
    print("Optimizations:")
    print("  1. Binary classification (UP/DOWN only)")
    print("  2. Reduced model complexity (max_depth=3, n_estimators=20)")
    print("  3. Feature selection (top 15 features)")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    symbols = ["SPY", "QQQ"]
    interval = '1min'
    trained_models = {}

    for symbol in symbols:
        try:
            trainer = train_symbol_optimized(
                symbol,
                interval=interval,
                top_n_features=15,
                save_model=True
            )

            if trainer:
                trained_models[symbol] = trainer
                print(f"\n[OK] {symbol} optimized XGBoost model trained successfully")

        except Exception as e:
            print(f"\n[ERROR] Failed to train {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - OPTIMIZED TRAINING")
    print("="*80)
    print("\nComparison with Original:")
    print(f"{'Symbol':<8} {'Original Train':<15} {'Optimized Train':<16} {'Original Test':<15} {'Optimized Test':<16}")
    print("-" * 80)

    for symbol, trainer in trained_models.items():
        metrics = trainer.training_metrics
        # Original results were ~99% train, ~50% test
        orig_features = 47 if symbol == "SPY" else 37
        print(f"{symbol:<8} {'99.0%':<15} {metrics['train_accuracy']*100:>6.2f}% {'':<8} {'50.0%':<15} {metrics['test_accuracy']*100:>6.2f}%")
        print(f"         Features: {orig_features} -> {metrics['num_features']}")
        print(f"         Classes: 3 (UP/DOWN/NEUTRAL) -> 2 (UP/DOWN)")
        print(f"         Trees: 100 -> 20")
        print(f"         Max depth: 6 -> 3")
        print()

    print(f"[OK] Optimized models saved to models/trained/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
