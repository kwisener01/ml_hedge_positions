"""
Directional Greek Levels Strategy

Hypothesis: At moderate strength levels (15-30), direction is predictable:
- At SUPPORT (HP/MHP support) → Predict UP (dealers buy to hedge)
- At RESISTANCE (HP/MHP resistance) → Predict DOWN (dealers sell to hedge)

This combines:
1. Strength filtering (moderate 15-30 wins)
2. Level type identification (support vs resistance)
3. Directional bias application
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
from features.feature_matrix import FeatureMatrixBuilder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


class DirectionalGreekStrategy:
    """Strategy with directional bias at moderate Greek levels"""

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

    def calculate_level_strength(self, feature_dict: Dict) -> float:
        """Calculate strength score (same as weighted analysis)"""
        score = 0

        # Gamma (0-20 points)
        gamma = abs(feature_dict.get('inst_gamma', 0))
        score += min(gamma / 0.01, 1.0) * 20

        # Vanna (0-15 points)
        vanna = abs(feature_dict.get('inst_vanna', 0))
        score += min(vanna / 0.01, 1.0) * 15

        # HP proximity (0-10 points)
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
        if hp_support_dist < 0.005:
            score += (0.005 - hp_support_dist) / 0.005 * 10
        if hp_resist_dist < 0.005:
            score += (0.005 - hp_resist_dist) / 0.005 * 10

        # MHP proximity (0-20 points)
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))
        if mhp_support_dist < 0.005:
            score += (0.005 - mhp_support_dist) / 0.005 * 20
        if mhp_resist_dist < 0.005:
            score += (0.005 - mhp_resist_dist) / 0.005 * 20

        # HG proximity (0-25 points)
        if 'inst_hg_support_dist' in feature_dict:
            hg_support_dist = abs(feature_dict.get('inst_hg_support_dist', 1.0))
            hg_resist_dist = abs(feature_dict.get('inst_hg_resist_dist', 1.0))
            if hg_support_dist < 0.005:
                score += (0.005 - hg_support_dist) / 0.005 * 25
            if hg_resist_dist < 0.005:
                score += (0.005 - hg_resist_dist) / 0.005 * 25

        # Overlap bonus
        near_count = 0
        if hp_support_dist < 0.002 or hp_resist_dist < 0.002:
            near_count += 1
        if mhp_support_dist < 0.002 or mhp_resist_dist < 0.002:
            near_count += 1
        if near_count > 1:
            score += (near_count - 1) * 5

        return score

    def identify_level_type(self, feature_dict: Dict) -> str:
        """
        Determine if at support or resistance level

        Returns: 'support', 'resistance', or 'neutral'
        """
        # Get distances to each level type
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))

        # Weight MHP more heavily (2x) as it's stronger/longer-term
        support_score = 0
        resist_score = 0

        if hp_support_dist < 0.005:
            support_score += (0.005 - hp_support_dist) / 0.005 * 1.0
        if hp_resist_dist < 0.005:
            resist_score += (0.005 - hp_resist_dist) / 0.005 * 1.0

        if mhp_support_dist < 0.005:
            support_score += (0.005 - mhp_support_dist) / 0.005 * 2.0
        if mhp_resist_dist < 0.005:
            resist_score += (0.005 - mhp_resist_dist) / 0.005 * 2.0

        # Determine dominant level type
        if support_score > resist_score * 1.2:  # Support clearly dominant
            return 'support'
        elif resist_score > support_score * 1.2:  # Resistance clearly dominant
            return 'resistance'
        else:
            return 'neutral'  # Mixed or unclear

    def apply_directional_bias(
        self,
        model_prediction: int,
        level_type: str,
        confidence_threshold: float = 0.6
    ) -> int:
        """
        Apply directional bias based on level type

        Args:
            model_prediction: Original model prediction (0=DOWN, 1=UP)
            level_type: 'support', 'resistance', or 'neutral'
            confidence_threshold: How strongly to apply bias

        Returns:
            Adjusted prediction
        """
        if level_type == 'support':
            # At support, bias toward UP (dealers buy to hedge)
            # Override DOWN predictions with moderate confidence
            if model_prediction == 0:  # Model says DOWN
                # Override to UP (support should bounce)
                return 1
            else:
                # Model already says UP, keep it
                return 1

        elif level_type == 'resistance':
            # At resistance, bias toward DOWN (dealers sell to hedge)
            if model_prediction == 1:  # Model says UP
                # Override to DOWN (resistance should reject)
                return 0
            else:
                # Model already says DOWN, keep it
                return 0

        else:  # neutral
            # No clear level, use model prediction as-is
            return model_prediction

    def evaluate_directional_strategy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_dict_list: List[Dict],
        all_feature_names: List[str],
        test_start_idx: int,
        min_strength: float = 15,
        max_strength: float = 30
    ):
        """
        Evaluate strategy with directional bias at moderate strength levels

        Strategy:
        1. Filter for moderate strength (15-30)
        2. Identify level type (support/resistance)
        3. Apply directional bias
        4. Compare vs baseline
        """
        print(f"\n{'='*80}")
        print(f"DIRECTIONAL GREEK STRATEGY: {self.symbol}")
        print(f"{'='*80}")

        # Filter features
        if self.selected_features:
            selected_indices = [all_feature_names.index(f) for f in self.selected_features if f in all_feature_names]
            X = X[:, selected_indices]

        # Calculate strengths and level types
        print(f"\nAnalyzing {len(feature_dict_list)} samples...")
        strengths = np.array([self.calculate_level_strength(fd) for fd in feature_dict_list])
        level_types = [self.identify_level_type(fd) for fd in feature_dict_list]

        # Split test set
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]
        strengths_test = strengths[test_start_idx:]
        level_types_test = level_types[test_start_idx:]

        # Get baseline model predictions
        y_pred_baseline = self.model.predict(X_test)

        # Apply directional strategy
        print(f"\nApplying directional bias strategy...")

        # Filter for moderate strength
        moderate_mask = (strengths_test >= min_strength) & (strengths_test <= max_strength)

        # Apply directional bias only at moderate strength levels
        y_pred_directional = y_pred_baseline.copy()

        support_overrides = 0
        resistance_overrides = 0

        for i in range(len(y_pred_directional)):
            if moderate_mask[i]:
                original = y_pred_baseline[i]
                level_type = level_types_test[i]

                # Apply bias
                biased = self.apply_directional_bias(original, level_type)

                if biased != original:
                    if level_type == 'support':
                        support_overrides += 1
                    elif level_type == 'resistance':
                        resistance_overrides += 1

                y_pred_directional[i] = biased

        print(f"  Moderate strength samples: {moderate_mask.sum()}/{len(y_test)} ({moderate_mask.sum()/len(y_test)*100:.1f}%)")
        print(f"  Overrides at support: {support_overrides}")
        print(f"  Overrides at resistance: {resistance_overrides}")

        # Calculate accuracies
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        acc_directional_all = accuracy_score(y_test, y_pred_directional)

        if moderate_mask.sum() > 0:
            acc_moderate_baseline = accuracy_score(y_test[moderate_mask], y_pred_baseline[moderate_mask])
            acc_moderate_directional = accuracy_score(y_test[moderate_mask], y_pred_directional[moderate_mask])
        else:
            acc_moderate_baseline = None
            acc_moderate_directional = None

        # Results
        print(f"\n{'='*80}")
        print("RESULTS COMPARISON")
        print(f"{'='*80}")

        print(f"\n{'Strategy':<35} {'Accuracy':<12} {'Samples':<10} {'Improvement'}")
        print("-" * 80)

        print(f"{'Baseline (all predictions)':<35} {acc_baseline*100:>6.2f}%     {len(y_test):<10} -")

        if acc_moderate_baseline is not None:
            print(f"{'Moderate strength (no bias)':<35} {acc_moderate_baseline*100:>6.2f}%     {moderate_mask.sum():<10} {(acc_moderate_baseline - acc_baseline)*100:>+6.2f}%")

        if acc_moderate_directional is not None:
            print(f"{'Moderate + Directional Bias':<35} {acc_moderate_directional*100:>6.2f}%     {moderate_mask.sum():<10} {(acc_moderate_directional - acc_baseline)*100:>+6.2f}%")

        print(f"{'Directional (applied to all)':<35} {acc_directional_all*100:>6.2f}%     {len(y_test):<10} {(acc_directional_all - acc_baseline)*100:>+6.2f}%")

        # Breakdown by level type (in moderate range)
        if moderate_mask.sum() > 0:
            print(f"\n{'='*80}")
            print("ACCURACY BY LEVEL TYPE (Moderate Strength Only)")
            print(f"{'='*80}")

            print(f"\n{'Level Type':<20} {'Samples':<10} {'Baseline':<12} {'Directional':<12} {'Improvement'}")
            print("-" * 80)

            for level_type in ['support', 'resistance', 'neutral']:
                type_mask = moderate_mask & np.array([lt == level_type for lt in level_types_test])
                n = type_mask.sum()

                if n >= 5:  # Minimum samples
                    baseline_acc = accuracy_score(y_test[type_mask], y_pred_baseline[type_mask])
                    directional_acc = accuracy_score(y_test[type_mask], y_pred_directional[type_mask])
                    improvement = (directional_acc - baseline_acc) * 100

                    marker = " <--" if improvement > 5 else ""
                    print(f"{level_type.upper():<20} {n:<10} {baseline_acc*100:>6.2f}%      {directional_acc*100:>6.2f}%      {improvement:>+6.2f}%{marker}")

        # Detailed metrics for moderate + directional
        if moderate_mask.sum() > 0:
            print(f"\n{'='*80}")
            print("CLASSIFICATION REPORT - Moderate Strength + Directional Bias")
            print(f"{'='*80}")

            print(classification_report(
                y_test[moderate_mask],
                y_pred_directional[moderate_mask],
                target_names=['DOWN', 'UP'],
                digits=4
            ))

            print(f"Confusion Matrix:")
            cm = confusion_matrix(y_test[moderate_mask], y_pred_directional[moderate_mask])
            print(f"              Predicted")
            print(f"              DOWN    UP")
            for i, row_label in enumerate(['DOWN', 'UP']):
                print(f"  Actual {row_label:<8} {cm[i][0]:4d}  {cm[i][1]:4d}")

        return {
            'baseline_acc': acc_baseline,
            'moderate_baseline_acc': acc_moderate_baseline,
            'moderate_directional_acc': acc_moderate_directional,
            'directional_all_acc': acc_directional_all,
            'n_moderate': moderate_mask.sum(),
            'n_total': len(y_test)
        }


def analyze_both_symbols():
    """Analyze directional strategy for both symbols"""
    print("\n" + "="*80)
    print("DIRECTIONAL GREEK LEVELS STRATEGY")
    print("="*80)
    print("Strategy: At moderate strength (15-30):")
    print("  - SUPPORT levels -> Predict UP (dealer buying)")
    print("  - RESISTANCE levels -> Predict DOWN (dealer selling)")
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

            strategy = DirectionalGreekStrategy(symbol, str(model_path))

            # Load and prepare data
            X, y, feature_dict_list, prices = strategy.load_and_prepare_data(interval='1min')
            all_feature_names = list(feature_dict_list[0].keys())

            test_start_idx = int(len(X) * 0.8)

            # Evaluate directional strategy
            results = strategy.evaluate_directional_strategy(
                X, y, feature_dict_list, all_feature_names, test_start_idx,
                min_strength=15, max_strength=30
            )

            all_results[symbol] = results

        except Exception as e:
            print(f"\n[ERROR] Failed to analyze {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Symbol':<8} {'Baseline':<12} {'Moderate Only':<15} {'+ Directional':<15} {'Gain'}")
    print("-" * 80)

    for symbol, results in all_results.items():
        baseline = results['baseline_acc'] * 100
        moderate = results['moderate_baseline_acc'] * 100 if results['moderate_baseline_acc'] else 0
        directional = results['moderate_directional_acc'] * 100 if results['moderate_directional_acc'] else 0
        gain = directional - baseline

        print(f"{symbol:<8} {baseline:>6.2f}%      {moderate:>6.2f}%         {directional:>6.2f}%         {gain:>+6.2f}%")

    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    print("Directional bias at moderate levels:")
    print("  - Leverages dealer hedging mechanics")
    print("  - Support = buying pressure -> UP bias")
    print("  - Resistance = selling pressure -> DOWN bias")
    print("  - Should improve accuracy vs random predictions")
    print("="*80)


if __name__ == "__main__":
    analyze_both_symbols()
