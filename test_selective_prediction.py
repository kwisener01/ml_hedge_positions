"""
Selective Prediction Strategy - Only predict near option Greek levels

Key insight: Don't try to predict every bar. Only predict when price is near:
- Hedge Pressure support/resistance
- Monthly Hedge Pressure support/resistance
- Gamma strikes
- Vanna levels

These levels create predictable dealer hedging behavior.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import model_config
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from sklearn.metrics import accuracy_score, classification_report


class BinaryTargetCalculator:
    """Binary target calculator - UP or DOWN only (matching optimized models)"""

    def calculate_target(self, current_price: float, future_quotes) -> str:
        if not future_quotes:
            return 'down'
        future_price = future_quotes[-1].mid_price
        return 'up' if future_price > current_price else 'down'

    def encode_target(self, target: str) -> int:
        return 1 if target == 'up' else 0

    def decode_target(self, encoded: int) -> str:
        return 'up' if encoded == 1 else 'down'


class SelectivePredictionAnalyzer:
    """Analyze model performance when predicting only near key option levels"""

    def __init__(self, symbol: str, model_path: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = BinaryTargetCalculator()  # Use binary targets

        # Load model
        print(f"\nLoading optimized model for {symbol}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.selected_features = model_data.get('selected_features', None)
        print(f"[OK] Model loaded")

    def load_and_prepare_data(self, interval: str = '1min'):
        """Load real intraday data and build features"""
        # Load intraday data
        intraday_dir = Path(__file__).parent / "data_local/intraday"
        pattern = f"{self.symbol}_{interval}_*.csv"
        files = list(intraday_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {interval} data found for {self.symbol}")

        filepath = sorted(files)[-1]
        df = pd.read_csv(filepath, parse_dates=['time'])
        print(f"\n[OK] Loaded {len(df)} bars from {filepath.name}")

        # Convert to quotes
        quotes = []
        for idx, row in df.iterrows():
            price = row['close']
            bar_range = row['high'] - row['low']
            spread = max(bar_range / 4, price * 0.0002)

            quote = QuoteData(
                symbol=self.symbol,
                bid=price - spread / 2,
                ask=price + spread / 2,
                bid_size=100,
                ask_size=100,
                last=price,
                volume=int(row['volume']),
                timestamp=row['time']
            )
            quotes.append(quote)

        # Build features
        print(f"\nBuilding features...")
        cleaned_quotes = self.cleaner.clean_quotes(quotes)

        window_builder = EventWindowBuilder()
        windows = []
        window_end_quotes = []

        for quote in cleaned_quotes:
            window = window_builder.add_event(quote)
            if window:
                windows.append(window)
                window_end_quotes.append(quote)

        print(f"  Created {len(windows)} windows")

        # Build feature dataset
        X_list = []
        y_list = []
        feature_dict_list = []
        prices = []

        for i in range(len(windows)):
            if i + 5 >= len(window_end_quotes):
                break

            current_quote = window_end_quotes[i]
            future_quotes = window_end_quotes[i+1:i+6]

            if len(future_quotes) < 5:
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
                feature_dict_list.append(feature_dict)
                prices.append(current_quote.mid_price)

            except Exception as e:
                continue

        X = np.vstack(X_list)
        y = np.array(y_list)

        print(f"  Valid samples: {len(X)}")

        return X, y, feature_dict_list, prices

    def identify_near_level_samples(
        self,
        feature_dict_list: List[Dict],
        prices: List[float],
        threshold_pct: float = 0.5
    ) -> np.ndarray:
        """
        Identify samples where price is near key option Greek levels

        Args:
            feature_dict_list: List of feature dictionaries for each sample
            prices: List of prices for each sample
            threshold_pct: Consider "near" if within this % of level

        Returns:
            Boolean array indicating which samples are near levels
        """
        print(f"\nIdentifying samples near option Greek levels...")
        print(f"  Threshold: within {threshold_pct}% of key levels")

        near_level_mask = np.zeros(len(feature_dict_list), dtype=bool)

        level_counts = {
            'hp_support': 0,
            'hp_resist': 0,
            'mhp_support': 0,
            'mhp_resist': 0,
            'gamma': 0,
            'vanna': 0
        }

        for i, (feature_dict, price) in enumerate(zip(feature_dict_list, prices)):
            near_any_level = False

            # Check Hedge Pressure levels
            if 'inst_hp_support_dist' in feature_dict:
                hp_support_dist_pct = abs(feature_dict['inst_hp_support_dist']) * 100
                if hp_support_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['hp_support'] += 1

            if 'inst_hp_resist_dist' in feature_dict:
                hp_resist_dist_pct = abs(feature_dict['inst_hp_resist_dist']) * 100
                if hp_resist_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['hp_resist'] += 1

            # Check Monthly Hedge Pressure levels
            if 'inst_mhp_support_dist' in feature_dict:
                mhp_support_dist_pct = abs(feature_dict['inst_mhp_support_dist']) * 100
                if mhp_support_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['mhp_support'] += 1

            if 'inst_mhp_resist_dist' in feature_dict:
                mhp_resist_dist_pct = abs(feature_dict['inst_mhp_resist_dist']) * 100
                if mhp_resist_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['mhp_resist'] += 1

            # Check Gamma levels
            if 'inst_hg_support_dist' in feature_dict:
                gamma_support_dist_pct = abs(feature_dict['inst_hg_support_dist']) * 100
                if gamma_support_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['gamma'] += 1

            if 'inst_hg_resist_dist' in feature_dict:
                gamma_resist_dist_pct = abs(feature_dict['inst_hg_resist_dist']) * 100
                if gamma_resist_dist_pct <= threshold_pct:
                    near_any_level = True
                    level_counts['gamma'] += 1

            # Check Vanna levels (if available)
            # Vanna is stored in gamma/vanna features
            if 'inst_vanna' in feature_dict:
                # High absolute vanna indicates near important level
                if abs(feature_dict.get('inst_vanna', 0)) > 0.1:  # Threshold for "high" vanna
                    near_any_level = True
                    level_counts['vanna'] += 1

            near_level_mask[i] = near_any_level

        print(f"\n  Samples near levels by type:")
        for level_type, count in level_counts.items():
            print(f"    {level_type:<20}: {count:>5}")

        total_near = near_level_mask.sum()
        print(f"\n  Total samples near ANY level: {total_near} ({total_near/len(near_level_mask)*100:.1f}%)")
        print(f"  Samples NOT near levels: {len(near_level_mask) - total_near} ({(1 - total_near/len(near_level_mask))*100:.1f}%)")

        return near_level_mask

    def evaluate_selective_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        near_level_mask: np.ndarray,
        test_start_idx: int,
        all_feature_names: List[str]
    ):
        """
        Evaluate model performance on:
        1. All predictions
        2. Only predictions near option levels
        3. Only predictions NOT near levels
        """
        print(f"\n{'='*80}")
        print(f"SELECTIVE PREDICTION ANALYSIS: {self.symbol}")
        print(f"{'='*80}")

        # Filter features if model uses selected features
        if self.selected_features:
            print(f"\nSelecting {len(self.selected_features)} features from {X.shape[1]} total...")
            # Get indices of selected features
            selected_indices = [all_feature_names.index(f) for f in self.selected_features if f in all_feature_names]
            X = X[:, selected_indices]
            print(f"  Feature matrix shape: {X.shape}")

        # Split into train/test (time-based)
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]
        near_level_mask_test = near_level_mask[test_start_idx:]

        # Make predictions on all test samples
        y_pred_all = self.model.predict(X_test)

        # Split into near-level and not-near-level
        near_indices = np.where(near_level_mask_test)[0]
        not_near_indices = np.where(~near_level_mask_test)[0]

        print(f"\nTest set breakdown:")
        print(f"  Total test samples: {len(y_test)}")
        print(f"  Near levels: {len(near_indices)} ({len(near_indices)/len(y_test)*100:.1f}%)")
        print(f"  NOT near levels: {len(not_near_indices)} ({len(not_near_indices)/len(y_test)*100:.1f}%)")

        # Calculate accuracies
        acc_all = accuracy_score(y_test, y_pred_all)

        if len(near_indices) > 0:
            acc_near = accuracy_score(y_test[near_indices], y_pred_all[near_indices])
        else:
            acc_near = None

        if len(not_near_indices) > 0:
            acc_not_near = accuracy_score(y_test[not_near_indices], y_pred_all[not_near_indices])
        else:
            acc_not_near = None

        print(f"\n{'='*80}")
        print("ACCURACY COMPARISON")
        print(f"{'='*80}")
        print(f"  ALL predictions:          {acc_all*100:>6.2f}% (n={len(y_test)})")
        if acc_near is not None:
            print(f"  NEAR levels only:         {acc_near*100:>6.2f}% (n={len(near_indices)})")
            improvement = (acc_near - acc_all) * 100
            print(f"  Improvement:              {improvement:>+6.2f}%")
        if acc_not_near is not None:
            print(f"  NOT NEAR levels:          {acc_not_near*100:>6.2f}% (n={len(not_near_indices)})")

        # Detailed metrics for near-level predictions
        if len(near_indices) > 0 and acc_near is not None:
            print(f"\n{'='*80}")
            print("NEAR-LEVEL PREDICTIONS - DETAILED METRICS")
            print(f"{'='*80}")

            # Determine if binary or 3-class
            unique_labels = np.unique(y_test)
            if len(unique_labels) == 2:
                target_names = ['DOWN', 'UP']
            else:
                target_names = ['DOWN', 'NEUTRAL', 'UP']

            print(classification_report(
                y_test[near_indices],
                y_pred_all[near_indices],
                target_names=target_names,
                digits=4
            ))

        return {
            'accuracy_all': acc_all,
            'accuracy_near_levels': acc_near,
            'accuracy_not_near': acc_not_near,
            'n_total': len(y_test),
            'n_near': len(near_indices),
            'n_not_near': len(not_near_indices)
        }


def analyze_both_symbols(threshold_pct: float = 0.5):
    """Analyze selective prediction strategy for both SPY and QQQ"""
    print("\n" + "="*80)
    print("SELECTIVE PREDICTION STRATEGY")
    print("="*80)
    print("Strategy: Only predict when price is near option Greek levels")
    print(f"Threshold: Within {threshold_pct}% of HP/MHP/Gamma/Vanna levels")
    print("="*80)

    symbols = ["SPY", "QQQ"]
    results = {}

    for symbol in symbols:
        try:
            model_path = Path(__file__).parent / f"models/trained/{symbol}_xgboost_optimized.pkl"

            if not model_path.exists():
                print(f"\n[SKIP] No optimized model found for {symbol}")
                continue

            print(f"\n\n{'='*80}")
            print(f"ANALYZING {symbol}")
            print(f"{'='*80}")

            analyzer = SelectivePredictionAnalyzer(symbol, str(model_path))

            # Load and prepare data
            X, y, feature_dict_list, prices = analyzer.load_and_prepare_data(interval='1min')

            # Get all feature names from first feature dict
            all_feature_names = list(feature_dict_list[0].keys())

            # Identify samples near levels
            near_level_mask = analyzer.identify_near_level_samples(
                feature_dict_list,
                prices,
                threshold_pct=threshold_pct
            )

            # Evaluate (80/20 split)
            test_start_idx = int(len(X) * 0.8)

            metrics = analyzer.evaluate_selective_predictions(
                X, y, near_level_mask, test_start_idx, all_feature_names
            )

            results[symbol] = metrics

        except Exception as e:
            print(f"\n[ERROR] Failed to analyze {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - SELECTIVE PREDICTION STRATEGY")
    print(f"{'='*80}")

    print(f"\n{'Symbol':<8} {'All Pred':<12} {'Near Levels':<12} {'Improvement':<12} {'Coverage':<12}")
    print("-" * 80)

    for symbol, metrics in results.items():
        acc_all = metrics['accuracy_all'] * 100
        acc_near = metrics['accuracy_near_levels'] * 100 if metrics['accuracy_near_levels'] else 0
        improvement = acc_near - acc_all
        coverage = metrics['n_near'] / metrics['n_total'] * 100 if metrics['n_total'] > 0 else 0

        print(f"{symbol:<8} {acc_all:>6.2f}%      {acc_near:>6.2f}%      {improvement:>+6.2f}%      {coverage:>6.1f}%")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("- 'All Pred': Accuracy when predicting every bar")
    print("- 'Near Levels': Accuracy when predicting ONLY near option levels")
    print("- 'Improvement': How much better predictions are near levels")
    print("- 'Coverage': % of time price is near a level (trading opportunities)")
    print()
    print("Higher 'Near Levels' accuracy means option Greeks create predictable behavior!")
    print("Higher 'Coverage' means more trading opportunities.")
    print("="*80)


if __name__ == "__main__":
    # Test with 0.2% threshold (within 0.2% of level)
    # This is more realistic for "near" a level
    analyze_both_symbols(threshold_pct=0.2)
