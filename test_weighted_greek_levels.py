"""
Weighted Greek Levels Strategy

Key insight: Not all levels are equal!
- Highest gamma strike = most predictable (maximum hedging)
- Highest vanna = strong volatility sensitivity
- Monthly HP > Daily HP (stronger, longer-term)
- Multiple levels overlapping = reinforced signal
- Weak levels = skip (unpredictable)

Level Strength Score = weighted combination of:
1. Gamma magnitude (higher = better)
2. Vanna magnitude (higher = better)
3. Distance to HP (closer = better)
4. Distance to MHP (closer = better, weighted higher)
5. Multiple levels overlapping (bonus)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import model_config
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from sklearn.metrics import accuracy_score, classification_report


class BinaryTargetCalculator:
    """Binary target calculator"""
    def calculate_target(self, current_price: float, future_quotes) -> str:
        if not future_quotes:
            return 'down'
        future_price = future_quotes[-1].mid_price
        return 'up' if future_price > current_price else 'down'

    def encode_target(self, target: str) -> int:
        return 1 if target == 'up' else 0

    def decode_target(self, encoded: int) -> str:
        return 'up' if encoded == 1 else 'down'


class WeightedGreekAnalyzer:
    """Analyze predictions weighted by Greek level strength"""

    def __init__(self, symbol: str, model_path: str):
        self.symbol = symbol
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.target_calc = BinaryTargetCalculator()

        # Load model
        print(f"\nLoading optimized model for {symbol}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.selected_features = model_data.get('selected_features', None)
        print(f"[OK] Model loaded")

    def load_and_prepare_data(self, interval: str = '1min'):
        """Load real intraday data and build features"""
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

    def calculate_level_strength(self, feature_dict: Dict, price: float) -> Tuple[float, Dict]:
        """
        Calculate strength of Greek levels at current price

        Higher score = stronger level = more predictable

        Components:
        1. Gamma magnitude (0-1, normalized)
        2. Vanna magnitude (0-1, normalized)
        3. HP proximity (inverse of distance, weighted)
        4. MHP proximity (inverse of distance, weighted 2x)
        5. Level overlap bonus

        Returns:
            strength_score: 0-100 scale
            components: Dict of individual components for analysis
        """
        components = {}

        # 1. Gamma magnitude (absolute value, normalized)
        gamma = abs(feature_dict.get('inst_gamma', 0))
        gamma_normalized = min(gamma / 0.01, 1.0)  # Cap at 0.01 as "max"
        components['gamma'] = gamma_normalized * 20  # 0-20 points

        # 2. Vanna magnitude
        vanna = abs(feature_dict.get('inst_vanna', 0))
        vanna_normalized = min(vanna / 0.01, 1.0)  # Cap at 0.01 as "max"
        components['vanna'] = vanna_normalized * 15  # 0-15 points

        # 3. HP proximity (daily hedge pressure)
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))

        # Closer = higher score (inverse of distance)
        # Within 0.1% = 10 points, 0.2% = 5 points, etc.
        hp_score = 0
        if hp_support_dist < 0.005:  # Within 0.5%
            hp_score = max(hp_score, (0.005 - hp_support_dist) / 0.005 * 10)
        if hp_resist_dist < 0.005:
            hp_score = max(hp_score, (0.005 - hp_resist_dist) / 0.005 * 10)

        components['hp'] = hp_score  # 0-10 points

        # 4. MHP proximity (monthly hedge pressure - STRONGER)
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))

        # Monthly is 2x weight of daily
        mhp_score = 0
        if mhp_support_dist < 0.005:
            mhp_score = max(mhp_score, (0.005 - mhp_support_dist) / 0.005 * 20)
        if mhp_resist_dist < 0.005:
            mhp_score = max(mhp_score, (0.005 - mhp_resist_dist) / 0.005 * 20)

        components['mhp'] = mhp_score  # 0-20 points

        # 5. Highest Gamma (HG) proximity
        if 'inst_hg_support_dist' in feature_dict and 'inst_hg_resist_dist' in feature_dict:
            hg_support_dist = abs(feature_dict.get('inst_hg_support_dist', 1.0))
            hg_resist_dist = abs(feature_dict.get('inst_hg_resist_dist', 1.0))

            hg_score = 0
            if hg_support_dist < 0.005:
                hg_score = max(hg_score, (0.005 - hg_support_dist) / 0.005 * 25)
            if hg_resist_dist < 0.005:
                hg_score = max(hg_score, (0.005 - hg_resist_dist) / 0.005 * 25)

            components['hg'] = hg_score  # 0-25 points
        else:
            components['hg'] = 0

        # 6. Overlap bonus (multiple levels close together = stronger)
        near_levels_count = 0
        if hp_support_dist < 0.002 or hp_resist_dist < 0.002:
            near_levels_count += 1
        if mhp_support_dist < 0.002 or mhp_resist_dist < 0.002:
            near_levels_count += 1
        if components['hg'] > 0:
            near_levels_count += 1

        overlap_bonus = (near_levels_count - 1) * 5 if near_levels_count > 1 else 0
        components['overlap'] = overlap_bonus  # 0-10 points

        # Total strength score (0-100)
        total_strength = (
            components['gamma'] +
            components['vanna'] +
            components['hp'] +
            components['mhp'] +
            components['hg'] +
            components['overlap']
        )

        return total_strength, components

    def evaluate_by_strength_tiers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_dict_list: List[Dict],
        prices: List[float],
        all_feature_names: List[str],
        test_start_idx: int
    ):
        """
        Evaluate model at different strength tiers

        Tiers:
        - Very Strong: score >= 50
        - Strong: 30 <= score < 50
        - Moderate: 15 <= score < 30
        - Weak: 5 <= score < 15
        - Very Weak: score < 5
        """
        print(f"\n{'='*80}")
        print(f"WEIGHTED GREEK LEVELS ANALYSIS: {self.symbol}")
        print(f"{'='*80}")

        # Calculate strength for all samples
        print(f"\nCalculating level strength for all samples...")
        strength_scores = []
        strength_components_list = []

        for feature_dict, price in zip(feature_dict_list, prices):
            score, components = self.calculate_level_strength(feature_dict, price)
            strength_scores.append(score)
            strength_components_list.append(components)

        strength_scores = np.array(strength_scores)

        print(f"\nStrength score statistics:")
        print(f"  Mean: {strength_scores.mean():.2f}")
        print(f"  Std: {strength_scores.std():.2f}")
        print(f"  Min: {strength_scores.min():.2f}")
        print(f"  Max: {strength_scores.max():.2f}")
        print(f"  Median: {np.median(strength_scores):.2f}")

        # Filter features if needed
        if self.selected_features:
            print(f"\nSelecting {len(self.selected_features)} features from {X.shape[1]} total...")
            selected_indices = [all_feature_names.index(f) for f in self.selected_features if f in all_feature_names]
            X = X[:, selected_indices]

        # Split into test set
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]
        strength_test = strength_scores[test_start_idx:]

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Define strength tiers
        tiers = [
            ('Very Strong', 50, 100),
            ('Strong', 30, 50),
            ('Moderate', 15, 30),
            ('Weak', 5, 15),
            ('Very Weak', 0, 5)
        ]

        print(f"\n{'='*80}")
        print("ACCURACY BY LEVEL STRENGTH")
        print(f"{'='*80}")
        print(f"{'Tier':<15} {'Strength':<15} {'Samples':<10} {'Accuracy':<10} {'vs Baseline'}")
        print("-" * 80)

        results = {}
        baseline_acc = accuracy_score(y_test, y_pred)

        for tier_name, min_score, max_score in tiers:
            mask = (strength_test >= min_score) & (strength_test < max_score)
            n_samples = mask.sum()

            if n_samples > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                improvement = (acc - baseline_acc) * 100

                print(f"{tier_name:<15} {min_score:>3}-{max_score:<3}       {n_samples:<10} {acc*100:>6.2f}%   {improvement:>+6.2f}%")

                results[tier_name] = {
                    'accuracy': acc,
                    'n_samples': n_samples,
                    'improvement': improvement
                }
            else:
                print(f"{tier_name:<15} {min_score:>3}-{max_score:<3}       {n_samples:<10} {'N/A':<10}")
                results[tier_name] = None

        print(f"\nBaseline (all predictions): {baseline_acc*100:.2f}% (n={len(y_test)})")

        # Detailed analysis of strongest tier
        strongest_tier = None
        for tier_name, min_score, max_score in tiers:
            if results.get(tier_name) and results[tier_name]['n_samples'] >= 20:
                strongest_tier = (tier_name, min_score, max_score)
                break

        if strongest_tier:
            tier_name, min_score, max_score = strongest_tier
            mask = (strength_test >= min_score) & (strength_test < max_score)

            print(f"\n{'='*80}")
            print(f"DETAILED METRICS - {tier_name.upper()} TIER (>={min_score})")
            print(f"{'='*80}")

            print(classification_report(
                y_test[mask],
                y_pred[mask],
                target_names=['DOWN', 'UP'],
                digits=4
            ))

        # Component analysis
        print(f"\n{'='*80}")
        print("STRENGTH COMPONENT CONTRIBUTIONS (Test Set Average)")
        print(f"{'='*80}")

        test_components = strength_components_list[test_start_idx:]
        avg_components = {
            'gamma': np.mean([c['gamma'] for c in test_components]),
            'vanna': np.mean([c['vanna'] for c in test_components]),
            'hp': np.mean([c['hp'] for c in test_components]),
            'mhp': np.mean([c['mhp'] for c in test_components]),
            'hg': np.mean([c['hg'] for c in test_components]),
            'overlap': np.mean([c['overlap'] for c in test_components])
        }

        print(f"{'Component':<15} {'Avg Score':<12} {'Max Possible':<15} {'Contribution'}")
        print("-" * 80)
        max_scores = {'gamma': 20, 'vanna': 15, 'hp': 10, 'mhp': 20, 'hg': 25, 'overlap': 10}

        for comp, avg_score in sorted(avg_components.items(), key=lambda x: x[1], reverse=True):
            max_score = max_scores[comp]
            contribution_pct = (avg_score / max_score * 100) if max_score > 0 else 0
            print(f"{comp.upper():<15} {avg_score:>6.2f}        {max_score:<15} {contribution_pct:>6.1f}%")

        return results

    def find_optimal_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strength_scores: np.ndarray,
        all_feature_names: List[str],
        test_start_idx: int
    ):
        """Find optimal strength threshold for trading"""
        print(f"\n{'='*80}")
        print("OPTIMAL STRENGTH THRESHOLD")
        print(f"{'='*80}")

        # Filter features
        if self.selected_features:
            selected_indices = [all_feature_names.index(f) for f in self.selected_features if f in all_feature_names]
            X = X[:, selected_indices]

        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]
        strength_test = strength_scores[test_start_idx:]

        y_pred = self.model.predict(X_test)

        # Test different thresholds
        thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        print(f"\n{'Threshold':<12} {'Samples':<10} {'Coverage':<12} {'Accuracy':<10} {'Improvement'}")
        print("-" * 80)

        baseline_acc = accuracy_score(y_test, y_pred)
        best_threshold = None
        best_acc = 0

        for threshold in thresholds:
            mask = strength_test >= threshold
            n_samples = mask.sum()
            coverage = n_samples / len(y_test) * 100

            if n_samples >= 10:  # Minimum samples for meaningful accuracy
                acc = accuracy_score(y_test[mask], y_pred[mask])
                improvement = (acc - baseline_acc) * 100

                marker = " *" if acc > best_acc and n_samples >= 30 else ""
                print(f"{threshold:<12} {n_samples:<10} {coverage:>6.1f}%      {acc*100:>6.2f}%   {improvement:>+6.2f}%{marker}")

                if acc > best_acc and n_samples >= 30:
                    best_acc = acc
                    best_threshold = threshold
            else:
                print(f"{threshold:<12} {n_samples:<10} {coverage:>6.1f}%      {'N/A':<10}")

        print(f"\nBaseline (all): {baseline_acc*100:.2f}% (n={len(y_test)})")
        if best_threshold:
            print(f"\n🎯 Optimal threshold: {best_threshold} (Accuracy: {best_acc*100:.2f}%)")

        return best_threshold


def analyze_both_symbols():
    """Analyze weighted Greek levels strategy for both symbols"""
    print("\n" + "="*80)
    print("WEIGHTED GREEK LEVELS STRATEGY")
    print("="*80)
    print("Hypothesis: Stronger Greek levels = More predictable price behavior")
    print("="*80)

    symbols = ["SPY", "QQQ"]
    all_results = {}

    for symbol in symbols:
        try:
            model_path = Path(__file__).parent / f"models/trained/{symbol}_xgboost_optimized.pkl"

            if not model_path.exists():
                print(f"\n[SKIP] No optimized model found for {symbol}")
                continue

            print(f"\n\n{'='*80}")
            print(f"ANALYZING {symbol}")
            print(f"{'='*80}")

            analyzer = WeightedGreekAnalyzer(symbol, str(model_path))

            # Load and prepare data
            X, y, feature_dict_list, prices = analyzer.load_and_prepare_data(interval='1min')
            all_feature_names = list(feature_dict_list[0].keys())

            # Calculate strengths
            strength_scores = np.array([
                analyzer.calculate_level_strength(fd, p)[0]
                for fd, p in zip(feature_dict_list, prices)
            ])

            test_start_idx = int(len(X) * 0.8)

            # Evaluate by tiers
            results = analyzer.evaluate_by_strength_tiers(
                X, y, feature_dict_list, prices, all_feature_names, test_start_idx
            )

            # Find optimal threshold
            best_threshold = analyzer.find_optimal_threshold(
                X, y, strength_scores, all_feature_names, test_start_idx
            )

            all_results[symbol] = {
                'tier_results': results,
                'best_threshold': best_threshold
            }

        except Exception as e:
            print(f"\n[ERROR] Failed to analyze {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - WEIGHTED STRATEGY")
    print(f"{'='*80}")

    for symbol, data in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Optimal threshold: {data.get('best_threshold', 'N/A')}")

        tier_results = data.get('tier_results', {})
        for tier_name in ['Very Strong', 'Strong']:
            if tier_results.get(tier_name):
                result = tier_results[tier_name]
                print(f"  {tier_name}: {result['accuracy']*100:.2f}% (n={result['n_samples']}, {result['improvement']:+.2f}%)")


if __name__ == "__main__":
    analyze_both_symbols()
