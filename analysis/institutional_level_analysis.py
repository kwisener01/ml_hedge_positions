"""
Analyze Model Predictions Near Institutional Levels
Shows how SVM responds to HP, MHP, and HG support/resistance
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.svm_ensemble import SVMEnsemble
from data.tradier_client import TradierClient
from features.feature_matrix import FeatureMatrixBuilder
from data.cleaner import DataCleaner, EventWindowBuilder


class InstitutionalLevelAnalyzer:
    """Analyze model behavior near institutional levels"""

    def __init__(self, ensemble_path: str):
        self.ensemble = SVMEnsemble.load(ensemble_path)
        self.client = TradierClient()
        self.feature_builder = FeatureMatrixBuilder(self.client)

    def analyze_predictions_by_distance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Analyze how predictions change based on distance to institutional levels

        Args:
            X: Feature matrix
            feature_names: Names of features
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Analysis results
        """
        # Find institutional distance features
        inst_features = {
            'hp_support': None,
            'hp_resistance': None,
            'mhp_support': None,
            'mhp_resistance': None,
            'hg_above': None,
            'hg_below': None
        }

        for i, name in enumerate(feature_names):
            if 'hp_support_dist' in name and 'mhp' not in name:
                inst_features['hp_support'] = i
            elif 'hp_resist_dist' in name and 'mhp' not in name:
                inst_features['hp_resistance'] = i
            elif 'mhp_support_dist' in name:
                inst_features['mhp_support'] = i
            elif 'mhp_resist_dist' in name:
                inst_features['mhp_resistance'] = i
            elif 'hg_above_dist' in name:
                inst_features['hg_above'] = i
            elif 'hg_below_dist' in name:
                inst_features['hg_below'] = i

        print("\n" + "="*80)
        print("INSTITUTIONAL LEVEL ANALYSIS")
        print("="*80)

        print("\nFeature Indices Found:")
        for level, idx in inst_features.items():
            if idx is not None:
                print(f"  {level:20} -> Feature {idx:2} ({feature_names[idx]})")
            else:
                print(f"  {level:20} -> NOT FOUND")

        results = {}

        # Analyze each institutional level type
        for level_name, feat_idx in inst_features.items():
            if feat_idx is None:
                continue

            distances = X[:, feat_idx]

            # Create distance buckets
            buckets = {
                'very_close': distances < 0.001,   # Within 0.1%
                'close': (distances >= 0.001) & (distances < 0.005),  # 0.1% - 0.5%
                'near': (distances >= 0.005) & (distances < 0.01),    # 0.5% - 1%
                'medium': (distances >= 0.01) & (distances < 0.02),   # 1% - 2%
                'far': distances >= 0.02                              # > 2%
            }

            level_results = {}

            for bucket_name, mask in buckets.items():
                if not np.any(mask):
                    continue

                # Predictions in this bucket
                bucket_pred = y_pred[mask]
                bucket_true = y_true[mask]

                # Accuracy
                accuracy = np.mean(bucket_pred == bucket_true)

                # Prediction distribution
                unique, counts = np.unique(bucket_pred, return_counts=True)
                pred_dist = {int(k): int(v) for k, v in zip(unique, counts)}

                # Ensure all classes present
                for label in [-1, 0, 1]:
                    if label not in pred_dist:
                        pred_dist[label] = 0

                level_results[bucket_name] = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(accuracy),
                    'predictions': pred_dist,
                    'mean_distance': float(distances[mask].mean()),
                    'std_distance': float(distances[mask].std())
                }

            results[level_name] = level_results

        return results

    def print_analysis(self, results: Dict):
        """Print formatted analysis results"""

        for level_name, buckets in results.items():
            print("\n" + "="*80)
            print(f"LEVEL: {level_name.upper().replace('_', ' ')}")
            print("="*80)

            if not buckets:
                print("  No data available")
                continue

            print(f"\n{'Distance':15} {'Count':>8} {'Accuracy':>10} {'DOWN':>8} {'NEUTRAL':>8} {'UP':>8}")
            print("-" * 80)

            for bucket_name, data in buckets.items():
                count = data['count']
                acc = data['accuracy'] * 100
                preds = data['predictions']

                down_pct = preds.get(-1, 0) / count * 100 if count > 0 else 0
                neutral_pct = preds.get(0, 0) / count * 100 if count > 0 else 0
                up_pct = preds.get(1, 0) / count * 100 if count > 0 else 0

                print(f"{bucket_name:15} {count:8} {acc:9.2f}% {down_pct:7.1f}% {neutral_pct:7.1f}% {up_pct:7.1f}%")

    def analyze_support_resistance_behavior(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y_pred: np.ndarray
    ) -> Dict:
        """
        Analyze if model respects support/resistance concepts

        Near support: Should predict UP more often
        Near resistance: Should predict DOWN more often
        """
        print("\n" + "="*80)
        print("SUPPORT/RESISTANCE BEHAVIOR ANALYSIS")
        print("="*80)

        # Find support/resistance features
        hp_support_idx = None
        hp_resist_idx = None
        mhp_support_idx = None
        mhp_resist_idx = None

        for i, name in enumerate(feature_names):
            if 'hp_support_dist' in name and 'mhp' not in name:
                hp_support_idx = i
            elif 'hp_resist_dist' in name and 'mhp' not in name:
                hp_resist_idx = i
            elif 'mhp_support_dist' in name:
                mhp_support_idx = i
            elif 'mhp_resist_dist' in name:
                mhp_resist_idx = i

        results = {}

        # HP Support/Resistance
        if hp_support_idx is not None and hp_resist_idx is not None:
            # Near HP support (< 0.5% away)
            near_support = X[:, hp_support_idx] < 0.005
            support_preds = y_pred[near_support]

            if len(support_preds) > 0:
                up_pct = np.sum(support_preds == 1) / len(support_preds) * 100
                down_pct = np.sum(support_preds == -1) / len(support_preds) * 100

                results['hp_support'] = {
                    'count': int(np.sum(near_support)),
                    'up_bias': float(up_pct - down_pct),  # Positive = predicting bounces
                    'up_pct': float(up_pct),
                    'down_pct': float(down_pct)
                }

            # Near HP resistance (< 0.5% away)
            near_resistance = X[:, hp_resist_idx] < 0.005
            resist_preds = y_pred[near_resistance]

            if len(resist_preds) > 0:
                up_pct = np.sum(resist_preds == 1) / len(resist_preds) * 100
                down_pct = np.sum(resist_preds == -1) / len(resist_preds) * 100

                results['hp_resistance'] = {
                    'count': int(np.sum(near_resistance)),
                    'down_bias': float(down_pct - up_pct),  # Positive = predicting rejections
                    'up_pct': float(up_pct),
                    'down_pct': float(down_pct)
                }

        # MHP Support/Resistance
        if mhp_support_idx is not None and mhp_resist_idx is not None:
            # Near MHP support
            near_support = X[:, mhp_support_idx] < 0.005
            support_preds = y_pred[near_support]

            if len(support_preds) > 0:
                up_pct = np.sum(support_preds == 1) / len(support_preds) * 100
                down_pct = np.sum(support_preds == -1) / len(support_preds) * 100

                results['mhp_support'] = {
                    'count': int(np.sum(near_support)),
                    'up_bias': float(up_pct - down_pct),
                    'up_pct': float(up_pct),
                    'down_pct': float(down_pct)
                }

            # Near MHP resistance
            near_resistance = X[:, mhp_resist_idx] < 0.005
            resist_preds = y_pred[near_resistance]

            if len(resist_preds) > 0:
                up_pct = np.sum(resist_preds == 1) / len(resist_preds) * 100
                down_pct = np.sum(resist_preds == -1) / len(resist_preds) * 100

                results['mhp_resistance'] = {
                    'count': int(np.sum(near_resistance)),
                    'down_bias': float(down_pct - up_pct),
                    'up_pct': float(up_pct),
                    'down_pct': float(down_pct)
                }

        # Print results
        print("\nExpected behavior:")
        print("  - Near SUPPORT: Model should predict UP (bounce)")
        print("  - Near RESISTANCE: Model should predict DOWN (rejection)")
        print("\nActual behavior:\n")

        for level, data in results.items():
            level_type = "SUPPORT" if "support" in level else "RESISTANCE"
            level_name = level.replace("_", " ").upper()

            print(f"{level_name} (n={data['count']}):")
            print(f"  UP predictions:   {data['up_pct']:5.1f}%")
            print(f"  DOWN predictions: {data['down_pct']:5.1f}%")

            if level_type == "SUPPORT":
                bias = data['up_bias']
                if bias > 10:
                    verdict = "[OK] CORRECT - Predicting bounces!"
                elif bias > 0:
                    verdict = "[WEAK] Slight bounce bias"
                else:
                    verdict = "[BAD] NOT respecting support"
            else:
                bias = data['down_bias']
                if bias > 10:
                    verdict = "[OK] CORRECT - Predicting rejections!"
                elif bias > 0:
                    verdict = "[WEAK] Slight rejection bias"
                else:
                    verdict = "[BAD] NOT respecting resistance"

            print(f"  {verdict}\n")

        return results


def analyze_model(ensemble_path: str, data_path: str = None):
    """
    Run complete institutional level analysis

    Args:
        ensemble_path: Path to trained ensemble model
        data_path: Optional path to test data CSV
    """
    print("\n" + "="*80)
    print("INSTITUTIONAL LEVEL PREDICTION ANALYSIS")
    print("="*80)

    analyzer = InstitutionalLevelAnalyzer(ensemble_path)

    # Load model training data to analyze
    from training.train_ensemble import TrainingDataBuilder

    symbol = analyzer.ensemble.symbol
    print(f"\nAnalyzing: {symbol}")
    print(f"Model: {ensemble_path}")

    # Build dataset
    builder = TrainingDataBuilder(symbol)
    daily_data = builder.load_historical_data()
    quotes = builder.create_synthetic_quotes(daily_data)
    X, y_returns, feature_names = builder.build_feature_dataset(quotes)

    # Get model predictions
    print("\nGenerating predictions for all samples...")
    y_pred = analyzer.ensemble.predict_batch(X)
    y_true = analyzer.ensemble._compute_target(y_returns)

    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")

    # Run analyses
    print("\n" + "="*80)
    print("ANALYSIS 1: PREDICTIONS BY DISTANCE TO LEVELS")
    print("="*80)

    distance_results = analyzer.analyze_predictions_by_distance(
        X, feature_names, y_true, y_pred
    )

    analyzer.print_analysis(distance_results)

    print("\n" + "="*80)
    print("ANALYSIS 2: SUPPORT/RESISTANCE RESPECT")
    print("="*80)

    sr_results = analyzer.analyze_support_resistance_behavior(
        X, feature_names, y_pred
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_institutional_analysis.json"

    all_results = {
        'symbol': symbol,
        'model_path': str(ensemble_path),
        'total_samples': int(len(X)),
        'distance_analysis': distance_results,
        'support_resistance_analysis': sr_results
    }

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"[OK] Analysis saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze model behavior near institutional levels")
    parser.add_argument("--symbol", default="SPY", help="Symbol to analyze")
    parser.add_argument("--model", help="Path to model file (optional)")

    args = parser.parse_args()

    # Find model file
    if args.model:
        model_path = args.model
    else:
        model_path = Path(__file__).parent.parent / f"models/trained/{args.symbol}_ensemble.pkl"

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    analyze_model(str(model_path))
