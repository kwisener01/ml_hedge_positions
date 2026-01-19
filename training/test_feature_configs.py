"""
Quick Feature Configuration Testing
Tests specific feature combinations for optimal performance
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tradier_client import TradierClient, QuoteData
from features.feature_matrix import FeatureMatrixBuilder
from models.svm_ensemble import SVMEnsemble
from sklearn.model_selection import train_test_split


class FeatureConfigTester:
    """
    Test different feature configurations

    Configs:
    - base_only: V1-V22 (22 features)
    - base_inst: V1-V22 + HP/MHP/HG (32 features)
    - base_inst_gamma: V1-V22 + Inst + Gamma/Vanna (37 features)
    - full: All features (47 features for SPY, 37 for QQQ)
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = TradierClient()

    def load_data(self):
        """Load historical data and generate synthetic quotes"""
        print(f"\nLoading data for {self.symbol}...")

        # Load historical data from CSV
        filepath = Path(__file__).parent.parent / f"data_local/price_history/{self.symbol}_daily.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Historical data not found: {filepath}")

        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        print(f"  Loaded {len(df)} days of historical data")

        # Generate synthetic quotes (same as train_ensemble.py)
        quotes_per_day = 10
        quotes = []

        for date, row in df.iterrows():
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

        print(f"  Generated {len(quotes)} synthetic quotes")
        return quotes

    def build_features_base_only(self, quotes):
        """22 features: V1-V22 only"""
        from features.window_features import WindowFeatureExtractor
        from features.classic_features import ClassicFeatureExtractor
        from data.cleaner import DataCleaner, EventWindowBuilder
        from features.feature_matrix import TargetCalculator

        window_extractor = WindowFeatureExtractor()
        classic_extractor = ClassicFeatureExtractor()
        cleaner = DataCleaner()
        target_calc = TargetCalculator()

        # Clean quotes
        cleaned = cleaner.clean_quotes(quotes)

        # Build windows
        window_builder = EventWindowBuilder(window_size=5, lookforward=5)
        windows = window_builder.build_windows(cleaned)

        X = []
        y = []

        for current_quote, window, future in windows:
            # Window features (V1-V10)
            window_feat = window_extractor.extract(window)

            # Classic features (V11-V22)
            classic_feat = classic_extractor.extract(current_quote, window)

            # Combine
            features = np.concatenate([
                window_feat.to_array(),
                classic_feat.to_array()
            ])

            # Target
            target = target_calc.calculate_target(current_quote.mid_price, future)
            target_encoded = target_calc.encode_target(target)

            X.append(features)
            y.append(target_encoded)

        return np.array(X), np.array(y)

    def build_features_masked(self, quotes, include_gamma: bool = True, include_lob: bool = False):
        """
        Build features with masking

        Args:
            include_gamma: Include gamma/vanna features
            include_lob: Include LOB features (SPY only)
        """
        from data.cleaner import DataCleaner, EventWindowBuilder
        from features.feature_matrix import TargetCalculator

        # Override builder settings
        builder = FeatureMatrixBuilder(self.client, enable_lob_for_all=include_lob)
        cleaner = DataCleaner()
        target_calc = TargetCalculator()

        # Clean quotes
        cleaned = cleaner.clean_quotes(quotes)

        # Build windows
        window_builder = EventWindowBuilder(window_size=5, lookforward=5)
        windows = window_builder.build_windows(cleaned)

        X = []
        y = []

        for current_quote, window, future in windows:
            # Get full features
            features, _ = builder.build_feature_vector(self.symbol, window, current_quote)

            # Mask features based on config
            if not include_gamma and not include_lob:
                # Base + Inst only (32 features)
                # Remove gamma/vanna (5 features) and LOB (10 features)
                # Keep: V1-V22 (22) + Inst base (10)
                features = features[:32]
            elif not include_gamma and include_lob:
                # Base + Inst + LOB, no gamma (42 features)
                # Remove gamma/vanna (5 features) from position 32-37
                # Keep: V1-V22 (22) + Inst base (10) + LOB (10)
                features = np.concatenate([features[:32], features[37:47]])
            elif include_gamma and not include_lob:
                # Base + Inst + Gamma, no LOB (37 features)
                # This is what we get naturally for QQQ
                features = features[:37]
            # else: full features (47 for SPY when LOB enabled)

            # Target
            target = target_calc.calculate_target(current_quote.mid_price, future)
            target_encoded = target_calc.encode_target(target)

            # Filter neutral if only 1 sample
            if target_encoded == 1:  # Neutral
                # Count existing neutrals
                neutral_count = sum(1 for t in y if t == 1)
                if neutral_count >= 2:
                    continue

            X.append(features)
            y.append(target_encoded)

        return np.array(X), np.array(y)

    def train_config(self, X, y, config_name: str):
        """Train ensemble and return test accuracy"""
        from sklearn.utils import resample

        # Handle class imbalance
        classes, counts = np.unique(y, return_counts=True)

        # If neutral class has <3 samples, stratify won't work
        if len(classes) < 3 or min(counts) < 3:
            # Remove neutral class or use simple split
            mask = y != 1  # Remove neutral
            X_filtered = X[mask]
            y_filtered = y[mask]

            if len(y_filtered) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_filtered, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

        print(f"\n  Training {config_name}:")
        print(f"    Samples: {len(X)} | Features: {X.shape[1]}")
        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

        # Train ensemble
        ensemble = SVMEnsemble(n_estimators=100)
        ensemble.fit(X_train, y_train)

        # Evaluate
        train_acc = ensemble.score(X_train, y_train)
        test_acc = ensemble.score(X_test, y_test)

        print(f"    Train Accuracy: {train_acc:.2%}")
        print(f"    Test Accuracy:  {test_acc:.2%}")

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'features': X.shape[1],
            'samples': len(X)
        }

    def run_all_configs(self, trials: int = 1):
        """Run all feature configurations"""

        print("\n" + "="*80)
        print(f"FEATURE CONFIGURATION TEST: {self.symbol}")
        print("="*80)
        print(f"Trials per config: {trials}")

        # Load data once
        quotes = self.load_data()

        configs = [
            ("Base Only (V1-V22)", "base_only", 22),
            ("Base + Inst (HP/MHP/HG)", "base_inst", 32),
            ("Base + Inst + Gamma/Vanna", "base_inst_gamma", 37),
        ]

        # Only test LOB for SPY
        if self.symbol == "SPY":
            configs.append(("Base + Inst + Gamma + LOB (All)", "full", 47))
            configs.append(("Base + Inst + LOB (no Gamma)", "base_inst_lob", 42))

        results = {}

        for config_name, config_code, expected_features in configs:
            print(f"\n{'='*80}")
            print(f"CONFIG: {config_name}")
            print(f"Expected features: {expected_features}")
            print(f"{'='*80}")

            trial_results = []

            for trial in range(trials):
                print(f"\nTrial {trial + 1}/{trials}")

                # Build features based on config
                if config_code == "base_only":
                    X, y = self.build_features_base_only(quotes)
                elif config_code == "base_inst":
                    X, y = self.build_features_masked(quotes, include_gamma=False, include_lob=False)
                elif config_code == "base_inst_gamma":
                    X, y = self.build_features_masked(quotes, include_gamma=True, include_lob=False)
                elif config_code == "full":
                    X, y = self.build_features_masked(quotes, include_gamma=True, include_lob=True)
                elif config_code == "base_inst_lob":
                    X, y = self.build_features_masked(quotes, include_gamma=False, include_lob=True)

                # Train and evaluate
                result = self.train_config(X, y, config_name)
                trial_results.append(result)

            # Average results
            avg_train = np.mean([r['train_acc'] for r in trial_results])
            avg_test = np.mean([r['test_acc'] for r in trial_results])
            std_test = np.std([r['test_acc'] for r in trial_results])

            results[config_name] = {
                'avg_train': avg_train,
                'avg_test': avg_test,
                'std_test': std_test,
                'features': expected_features
            }

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(f"\nSymbol: {self.symbol}")
        print(f"Trials: {trials}\n")
        print(f"{'Configuration':<40} {'Features':<10} {'Test Acc':<15} {'Std Dev':<10}")
        print("-"*80)

        for config_name, result in results.items():
            print(f"{config_name:<40} {result['features']:<10} "
                  f"{result['avg_test']:.2%}        {result['std_test']:.2%}")

        # Find best
        best_config = max(results.items(), key=lambda x: x[1]['avg_test'])
        print(f"\n{'='*80}")
        print(f"BEST CONFIG: {best_config[0]}")
        print(f"Test Accuracy: {best_config[1]['avg_test']:.2%} ± {best_config[1]['std_test']:.2%}")
        print(f"Features: {best_config[1]['features']}")
        print(f"{'='*80}\n")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test feature configurations")
    parser.add_argument("--symbol", type=str, default="QQQ", help="Symbol to test (SPY or QQQ)")
    parser.add_argument("--trials", type=int, default=3, help="Trials per configuration")

    args = parser.parse_args()

    tester = FeatureConfigTester(args.symbol)
    results = tester.run_all_configs(trials=args.trials)
