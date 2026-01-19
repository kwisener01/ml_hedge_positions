"""
Model Improvement Pipeline
Systematic experimentation to improve SVM ensemble performance
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_ensemble import TrainingDataBuilder
from models.svm_ensemble import SVMEnsemble
from config.settings import model_config


class ModelImprover:
    """
    Systematic model improvement experiments

    Strategies:
    1. Alpha threshold tuning
    2. Hyperparameter optimization (C, degree)
    3. Feature selection
    4. Class balancing
    5. Ensemble size tuning
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.builder = TrainingDataBuilder(symbol)
        self.results_file = Path(__file__).parent.parent / f"training/improvement_results_{symbol}.json"
        self.results = []

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare training data"""
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {self.symbol}")
        print(f"{'='*80}")

        historical_data = self.builder.load_historical_data()
        quotes = self.builder.create_synthetic_quotes(historical_data, quotes_per_day=10)
        X, y_returns, feature_names = self.builder.build_feature_dataset(quotes, lookforward_bars=5)

        return X, y_returns, feature_names

    def experiment_alpha_threshold(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str]
    ):
        """
        Experiment 1: Alpha threshold tuning

        The alpha threshold determines what counts as UP/DOWN vs NEUTRAL.
        Current: 1e-05 (0.001%) - too small, almost no neutrals
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 1: ALPHA THRESHOLD TUNING")
        print(f"{'='*80}")

        # Test different thresholds
        alpha_values = [
            1e-05,   # Current (0.001%)
            1e-04,   # 0.01%
            5e-04,   # 0.05%
            1e-03,   # 0.1%
            2.5e-03, # 0.25%
            5e-03,   # 0.5%
            1e-02    # 1.0%
        ]

        best_acc = 0
        best_alpha = None

        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha} ({alpha*100:.3f}%)")

            # Train with this alpha
            ensemble = SVMEnsemble(
                ensemble_size=50,  # Use smaller ensemble for speed
                alpha_threshold=alpha
            )

            metrics = ensemble.train(
                X=X,
                y_returns=y_returns,
                feature_names=feature_names,
                symbol=self.symbol,
                test_size=0.2,
                verbose=False
            )

            # Check target distribution
            y = ensemble._compute_target(y_returns)
            unique, counts = np.unique(y, return_counts=True)
            distribution = dict(zip(unique, counts))

            print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")
            print(f"  Target Distribution:")
            for label in [-1, 0, 1]:
                count = distribution.get(label, 0)
                pct = count / len(y) * 100 if len(y) > 0 else 0
                print(f"    {label:2}: {count:4} ({pct:5.1f}%)")

            # Save result
            self.results.append({
                'experiment': 'alpha_threshold',
                'alpha': alpha,
                'test_accuracy': metrics.test_accuracy,
                'train_accuracy': metrics.train_accuracy,
                'distribution': {int(k): int(v) for k, v in distribution.items()},
                'timestamp': datetime.now().isoformat()
            })

            if metrics.test_accuracy > best_acc:
                best_acc = metrics.test_accuracy
                best_alpha = alpha

        print(f"\n{'='*60}")
        print(f"BEST ALPHA: {best_alpha} ({best_alpha*100:.3f}%) -> {best_acc*100:.2f}% accuracy")
        print(f"{'='*60}")

        return best_alpha, best_acc

    def experiment_hyperparameters(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str],
        alpha_threshold: float = 1e-03
    ):
        """
        Experiment 2: Hyperparameter optimization (C and degree)
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")

        # Test different combinations
        c_values = [0.1, 0.25, 0.5, 1.0]
        degree_values = [2, 3]

        best_acc = 0
        best_params = None

        for C in c_values:
            for degree in degree_values:
                print(f"\nTesting C={C}, degree={degree}")

                ensemble = SVMEnsemble(
                    ensemble_size=50,
                    polynomial_degree=degree,
                    constraint_param=C,
                    alpha_threshold=alpha_threshold
                )

                metrics = ensemble.train(
                    X=X,
                    y_returns=y_returns,
                    feature_names=feature_names,
                    symbol=self.symbol,
                    test_size=0.2,
                    verbose=False
                )

                print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")
                print(f"  Train Accuracy: {metrics.train_accuracy*100:.2f}%")

                self.results.append({
                    'experiment': 'hyperparameters',
                    'C': C,
                    'degree': degree,
                    'alpha': alpha_threshold,
                    'test_accuracy': metrics.test_accuracy,
                    'train_accuracy': metrics.train_accuracy,
                    'timestamp': datetime.now().isoformat()
                })

                if metrics.test_accuracy > best_acc:
                    best_acc = metrics.test_accuracy
                    best_params = {'C': C, 'degree': degree}

        print(f"\n{'='*60}")
        print(f"BEST PARAMS: C={best_params['C']}, degree={best_params['degree']} -> {best_acc*100:.2f}%")
        print(f"{'='*60}")

        return best_params, best_acc

    def experiment_feature_selection(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str],
        alpha_threshold: float = 1e-03
    ):
        """
        Experiment 3: Feature selection

        Reduce from 32 features to most important ones
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: FEATURE SELECTION")
        print(f"{'='*80}")

        # Get target labels
        ensemble = SVMEnsemble(alpha_threshold=alpha_threshold)
        y = ensemble._compute_target(y_returns)

        # Feature importance using mutual information
        print("\nCalculating feature importances...")
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Sort features by importance
        feature_importance = sorted(
            zip(feature_names, mi_scores),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop 15 Most Important Features:")
        for i, (name, score) in enumerate(feature_importance[:15]):
            print(f"  {i+1:2}. {name:30} | Score: {score:.4f}")

        # Test different feature counts
        k_values = [10, 15, 20, 25, 32]
        best_acc = 0
        best_k = None

        print("\nTesting different feature counts...")

        for k in k_values:
            print(f"\nTesting top {k} features...")

            # Select top k features
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

            # Train with selected features
            ensemble = SVMEnsemble(
                ensemble_size=50,
                alpha_threshold=alpha_threshold
            )

            metrics = ensemble.train(
                X=X_selected,
                y_returns=y_returns,
                feature_names=selected_features,
                symbol=self.symbol,
                test_size=0.2,
                verbose=False
            )

            print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")

            self.results.append({
                'experiment': 'feature_selection',
                'num_features': k,
                'features': selected_features,
                'test_accuracy': metrics.test_accuracy,
                'train_accuracy': metrics.train_accuracy,
                'timestamp': datetime.now().isoformat()
            })

            if metrics.test_accuracy > best_acc:
                best_acc = metrics.test_accuracy
                best_k = k

        print(f"\n{'='*60}")
        print(f"BEST K: {best_k} features -> {best_acc*100:.2f}% accuracy")
        print(f"{'='*60}")

        return best_k, best_acc, feature_importance

    def experiment_class_balancing(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str],
        alpha_threshold: float = 1e-03
    ):
        """
        Experiment 4: Class balancing strategies

        Current issue: Class imbalance (more UP than DOWN)
        Try: SMOTE (oversampling), undersampling, hybrid
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 4: CLASS BALANCING")
        print(f"{'='*80}")

        # Get targets
        ensemble = SVMEnsemble(alpha_threshold=alpha_threshold)
        y = ensemble._compute_target(y_returns)

        # Show current distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nOriginal Distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label:2}: {count:4} ({count/len(y)*100:.1f}%)")

        strategies = {
            'none': None,
            'smote': SMOTE(random_state=42),
            'undersample': RandomUnderSampler(random_state=42),
            'smote_undersample': SMOTE(random_state=42)
        }

        best_acc = 0
        best_strategy = None

        for strategy_name, sampler in strategies.items():
            print(f"\nTesting strategy: {strategy_name}")

            X_resampled = X
            y_resampled = y

            if sampler:
                try:
                    if strategy_name == 'smote_undersample':
                        # Hybrid: SMOTE then undersample
                        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
                        X_resampled, y_resampled = RandomUnderSampler(random_state=42).fit_resample(X_resampled, y_resampled)
                    else:
                        X_resampled, y_resampled = sampler.fit_resample(X, y)

                    # Show new distribution
                    unique, counts = np.unique(y_resampled, return_counts=True)
                    print(f"  Resampled Distribution:")
                    for label, count in zip(unique, counts):
                        print(f"    {label:2}: {count:4} ({count/len(y_resampled)*100:.1f}%)")

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

            # Train with resampled data
            ensemble = SVMEnsemble(
                ensemble_size=50,
                alpha_threshold=alpha_threshold
            )

            # Need to reconstruct y_returns for resampled data
            # Use dummy returns since we already have y labels
            y_returns_resampled = y_resampled * alpha_threshold * 2

            metrics = ensemble.train(
                X=X_resampled,
                y_returns=y_returns_resampled,
                feature_names=feature_names,
                symbol=self.symbol,
                test_size=0.2,
                verbose=False
            )

            print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")

            self.results.append({
                'experiment': 'class_balancing',
                'strategy': strategy_name,
                'samples_before': len(y),
                'samples_after': len(y_resampled),
                'test_accuracy': metrics.test_accuracy,
                'train_accuracy': metrics.train_accuracy,
                'timestamp': datetime.now().isoformat()
            })

            if metrics.test_accuracy > best_acc:
                best_acc = metrics.test_accuracy
                best_strategy = strategy_name

        print(f"\n{'='*60}")
        print(f"BEST STRATEGY: {best_strategy} -> {best_acc*100:.2f}% accuracy")
        print(f"{'='*60}")

        return best_strategy, best_acc

    def experiment_ensemble_size(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str],
        alpha_threshold: float = 1e-03
    ):
        """
        Experiment 5: Ensemble size optimization

        Current: 100 SVMs
        Test: 25, 50, 100, 200
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 5: ENSEMBLE SIZE")
        print(f"{'='*80}")

        sizes = [25, 50, 100, 200]
        best_acc = 0
        best_size = None

        for size in sizes:
            print(f"\nTesting ensemble size: {size}")

            ensemble = SVMEnsemble(
                ensemble_size=size,
                alpha_threshold=alpha_threshold
            )

            metrics = ensemble.train(
                X=X,
                y_returns=y_returns,
                feature_names=feature_names,
                symbol=self.symbol,
                test_size=0.2,
                verbose=False
            )

            print(f"  Test Accuracy: {metrics.test_accuracy*100:.2f}%")
            print(f"  Training Time: {metrics.training_time:.1f}s")

            self.results.append({
                'experiment': 'ensemble_size',
                'size': size,
                'test_accuracy': metrics.test_accuracy,
                'train_accuracy': metrics.train_accuracy,
                'training_time': metrics.training_time,
                'timestamp': datetime.now().isoformat()
            })

            if metrics.test_accuracy > best_acc:
                best_acc = metrics.test_accuracy
                best_size = size

        print(f"\n{'='*60}")
        print(f"BEST SIZE: {best_size} SVMs -> {best_acc*100:.2f}% accuracy")
        print(f"{'='*60}")

        return best_size, best_acc

    def save_results(self):
        """Save all experiment results to JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n[OK] Results saved to {self.results_file}")

    def run_all_experiments(self):
        """Run complete improvement pipeline"""
        print(f"\n{'='*80}")
        print(f"MODEL IMPROVEMENT PIPELINE: {self.symbol}")
        print(f"{'='*80}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Load data once
        X, y_returns, feature_names = self.load_training_data()

        # Experiment 1: Alpha threshold
        best_alpha, acc1 = self.experiment_alpha_threshold(X, y_returns, feature_names)

        # Experiment 2: Hyperparameters (using best alpha)
        best_params, acc2 = self.experiment_hyperparameters(X, y_returns, feature_names, best_alpha)

        # Experiment 3: Feature selection (using best alpha)
        best_k, acc3, feature_importance = self.experiment_feature_selection(X, y_returns, feature_names, best_alpha)

        # Experiment 4: Class balancing (using best alpha)
        best_strategy, acc4 = self.experiment_class_balancing(X, y_returns, feature_names, best_alpha)

        # Experiment 5: Ensemble size (using best alpha)
        best_size, acc5 = self.experiment_ensemble_size(X, y_returns, feature_names, best_alpha)

        # Final summary
        print(f"\n{'='*80}")
        print(f"IMPROVEMENT SUMMARY: {self.symbol}")
        print(f"{'='*80}")
        print(f"\nBest configurations found:")
        print(f"  Alpha Threshold: {best_alpha} ({best_alpha*100:.3f}%) -> {acc1*100:.2f}%")
        print(f"  Hyperparameters: C={best_params['C']}, degree={best_params['degree']} -> {acc2*100:.2f}%")
        print(f"  Feature Count: {best_k} features -> {acc3*100:.2f}%")
        print(f"  Class Balancing: {best_strategy} -> {acc4*100:.2f}%")
        print(f"  Ensemble Size: {best_size} SVMs -> {acc5*100:.2f}%")

        best_overall = max(acc1, acc2, acc3, acc4, acc5)
        print(f"\nBest Overall Accuracy: {best_overall*100:.2f}%")

        # Save all results
        self.save_results()

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        return {
            'alpha': best_alpha,
            'params': best_params,
            'k_features': best_k,
            'balancing': best_strategy,
            'ensemble_size': best_size,
            'best_accuracy': best_overall
        }


def main():
    """Run improvement experiments"""
    import argparse

    parser = argparse.ArgumentParser(description="Model Improvement Pipeline")
    parser.add_argument('--symbol', choices=['SPY', 'QQQ'], default='SPY',
                        help='Symbol to improve')
    parser.add_argument('--experiment', choices=[
        'all', 'alpha', 'hyperparams', 'features', 'balancing', 'ensemble'
    ], default='all', help='Which experiment to run')

    args = parser.parse_args()

    improver = ModelImprover(args.symbol)
    X, y_returns, feature_names = improver.load_training_data()

    if args.experiment == 'all':
        improver.run_all_experiments()
    elif args.experiment == 'alpha':
        improver.experiment_alpha_threshold(X, y_returns, feature_names)
        improver.save_results()
    elif args.experiment == 'hyperparams':
        improver.experiment_hyperparameters(X, y_returns, feature_names)
        improver.save_results()
    elif args.experiment == 'features':
        improver.experiment_feature_selection(X, y_returns, feature_names)
        improver.save_results()
    elif args.experiment == 'balancing':
        improver.experiment_class_balancing(X, y_returns, feature_names)
        improver.save_results()
    elif args.experiment == 'ensemble':
        improver.experiment_ensemble_size(X, y_returns, feature_names)
        improver.save_results()


if __name__ == "__main__":
    main()
