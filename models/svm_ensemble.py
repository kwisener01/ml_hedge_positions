"""
SVM Ensemble Module
Implements the 100-SVM polynomial kernel ensemble per specification
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from config.settings import model_config


@dataclass
class SVMPrediction:
    """Single ensemble prediction result"""
    symbol: str
    prediction: int  # -1 (down), 0 (neutral), 1 (up)
    confidence: float  # 0-1, based on voting agreement
    vote_distribution: Dict[int, int]  # {-1: count, 0: count, 1: count}
    timestamp: datetime
    feature_vector: np.ndarray


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    train_accuracy: float
    test_accuracy: float
    train_size: int
    test_size: int
    ensemble_size: int
    training_time: float
    individual_accuracies: List[float]

    @property
    def avg_individual_accuracy(self) -> float:
        return np.mean(self.individual_accuracies)

    @property
    def std_individual_accuracy(self) -> float:
        return np.std(self.individual_accuracies)


class SVMEnsemble:
    """
    Ensemble of 100 independent polynomial SVMs

    Each SVM:
    - Polynomial kernel degree 2 (d=2)
    - C=0.25 constraint parameter
    - Trained on random subset of data
    - Votes on final prediction
    """

    def __init__(
        self,
        ensemble_size: int = None,
        polynomial_degree: int = None,
        constraint_param: float = None,
        alpha_threshold: float = None
    ):
        self.ensemble_size = ensemble_size or model_config.ensemble_count
        self.polynomial_degree = polynomial_degree or model_config.polynomial_degree
        self.constraint_param = constraint_param or model_config.constraint_param
        self.alpha_threshold = alpha_threshold or model_config.alpha_threshold

        # Ensemble components
        self.models: List[SVC] = []
        self.scalers: List[StandardScaler] = []
        self.training_indices: List[np.ndarray] = []

        # Training metadata
        self.is_trained = False
        self.training_metrics: Optional[TrainingMetrics] = None
        self.feature_names: List[str] = []
        self.symbol: str = ""

    def _create_svm(self) -> SVC:
        """Create a single SVM with specified parameters"""
        return SVC(
            kernel='poly',
            degree=self.polynomial_degree,
            C=self.constraint_param,
            gamma='scale',  # Automatic scaling
            coef0=1.0,  # Independent term in polynomial
            class_weight='balanced',  # Handle class imbalance
            random_state=None  # Each SVM gets different randomization
        )

    def _compute_target(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute target labels from returns

        Args:
            returns: Array of forward returns

        Returns:
            Array of labels: -1 (down), 0 (neutral), 1 (up)
        """
        targets = np.zeros(len(returns), dtype=int)
        targets[returns > self.alpha_threshold] = 1
        targets[returns < -self.alpha_threshold] = -1
        return targets

    def train(
        self,
        X: np.ndarray,
        y_returns: np.ndarray,
        feature_names: List[str] = None,
        symbol: str = "",
        test_size: float = 0.2,
        subset_fraction: float = 0.8,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        Train the ensemble

        Args:
            X: Feature matrix (n_samples, n_features)
            y_returns: Forward returns for target generation
            feature_names: Names of features (for tracking)
            symbol: Symbol being trained
            test_size: Fraction for test set
            subset_fraction: Fraction of training data for each SVM
            verbose: Print progress

        Returns:
            Training metrics
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training SVM Ensemble: {symbol}")
            print(f"{'='*80}")
            print(f"Samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
            print(f"Ensemble size: {self.ensemble_size}")
            print(f"Polynomial degree: {self.polynomial_degree}")
            print(f"C parameter: {self.constraint_param}")
            print(f"Alpha threshold: {self.alpha_threshold}")

        start_time = datetime.now()

        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.symbol = symbol

        # Compute targets
        y = self._compute_target(y_returns)

        # Show target distribution
        if verbose:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\nTarget distribution:")
            for label, count in zip(unique, counts):
                direction = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(label, "UNKNOWN")
                print(f"  {direction:8} ({label:2}): {count:5} ({count/len(y)*100:5.1f}%)")

        # Train/test split
        # Check if we can stratify (need at least 2 samples per class)
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = all(counts >= 2)

        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        if verbose:
            print(f"\nTrain size: {len(X_train)}")
            print(f"Test size: {len(X_test)}")

        # Train ensemble
        self.models = []
        self.scalers = []
        self.training_indices = []
        individual_accuracies = []

        if verbose:
            print(f"\nTraining {self.ensemble_size} SVMs...")

        for i in range(self.ensemble_size):
            # Random subset of training data
            n_subset = int(len(X_train) * subset_fraction)
            indices = np.random.choice(len(X_train), n_subset, replace=False)

            X_subset = X_train[indices]
            y_subset = y_train[indices]

            # Scale features
            scaler = StandardScaler()
            X_subset_scaled = scaler.fit_transform(X_subset)

            # Train SVM
            svm = self._create_svm()
            svm.fit(X_subset_scaled, y_subset)

            # Track accuracy on training subset
            train_acc = svm.score(X_subset_scaled, y_subset)
            individual_accuracies.append(train_acc)

            # Store
            self.models.append(svm)
            self.scalers.append(scaler)
            self.training_indices.append(indices)

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{self.ensemble_size}] Trained | Acc: {train_acc:.3f}")

        # Mark as trained before evaluation
        self.is_trained = True

        # Evaluate ensemble
        if verbose:
            print(f"\nEvaluating ensemble...")

        train_preds = self.predict_batch(X_train, return_votes=False)
        test_preds = self.predict_batch(X_test, return_votes=False)

        train_accuracy = np.mean(train_preds == y_train)
        test_accuracy = np.mean(test_preds == y_test)

        training_time = (datetime.now() - start_time).total_seconds()

        # Store metrics
        self.training_metrics = TrainingMetrics(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_size=len(X_train),
            test_size=len(X_test),
            ensemble_size=self.ensemble_size,
            training_time=training_time,
            individual_accuracies=individual_accuracies
        )

        if verbose:
            print(f"\n{'='*80}")
            print("TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Training time: {training_time:.1f}s")
            print(f"Ensemble train accuracy: {train_accuracy*100:.2f}%")
            print(f"Ensemble test accuracy: {test_accuracy*100:.2f}%")
            print(f"Individual SVM avg accuracy: {self.training_metrics.avg_individual_accuracy*100:.2f}%")
            print(f"Individual SVM std accuracy: {self.training_metrics.std_individual_accuracy*100:.2f}%")

        return self.training_metrics

    def predict(self, X: np.ndarray) -> SVMPrediction:
        """
        Predict single sample using ensemble voting

        Args:
            X: Feature vector (n_features,) or (1, n_features)

        Returns:
            SVMPrediction with majority vote
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Reshape if needed
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect votes
        votes = []
        for svm, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = svm.predict(X_scaled)[0]
            votes.append(pred)

        # Count votes
        unique, counts = np.unique(votes, return_counts=True)
        vote_dist = dict(zip(unique.astype(int), counts.astype(int)))

        # Ensure all classes represented
        for label in [-1, 0, 1]:
            if label not in vote_dist:
                vote_dist[label] = 0

        # Majority vote
        majority_vote = max(vote_dist.items(), key=lambda x: x[1])[0]
        confidence = vote_dist[majority_vote] / self.ensemble_size

        return SVMPrediction(
            symbol=self.symbol,
            prediction=majority_vote,
            confidence=confidence,
            vote_distribution=vote_dist,
            timestamp=datetime.now(),
            feature_vector=X[0]
        )

    def predict_batch(self, X: np.ndarray, return_votes: bool = False) -> np.ndarray:
        """
        Predict batch of samples (faster than calling predict repeatedly)

        Args:
            X: Feature matrix (n_samples, n_features)
            return_votes: If True, return vote distribution instead of predictions

        Returns:
            Array of predictions or vote distributions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Collect all votes
        all_votes = np.zeros((len(X), self.ensemble_size), dtype=int)

        for i, (svm, scaler) in enumerate(zip(self.models, self.scalers)):
            X_scaled = scaler.transform(X)
            all_votes[:, i] = svm.predict(X_scaled)

        if return_votes:
            return all_votes

        # Majority vote for each sample
        predictions = np.apply_along_axis(
            lambda votes: np.bincount(votes + 1, minlength=3).argmax() - 1,
            axis=1,
            arr=all_votes
        )

        return predictions

    def save(self, filepath: str):
        """Save trained ensemble to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")

        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'training_indices': self.training_indices,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'symbol': self.symbol,
            'config': {
                'ensemble_size': self.ensemble_size,
                'polynomial_degree': self.polynomial_degree,
                'constraint_param': self.constraint_param,
                'alpha_threshold': self.alpha_threshold
            }
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[OK] Ensemble saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SVMEnsemble':
        """Load trained ensemble from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Create instance with saved config
        config = save_data['config']
        ensemble = cls(
            ensemble_size=config['ensemble_size'],
            polynomial_degree=config['polynomial_degree'],
            constraint_param=config['constraint_param'],
            alpha_threshold=config['alpha_threshold']
        )

        # Restore state
        ensemble.models = save_data['models']
        ensemble.scalers = save_data['scalers']
        ensemble.training_indices = save_data['training_indices']
        ensemble.training_metrics = save_data['training_metrics']
        ensemble.feature_names = save_data['feature_names']
        ensemble.symbol = save_data['symbol']
        ensemble.is_trained = True

        print(f"[OK] Ensemble loaded from {filepath}")
        print(f"  Symbol: {ensemble.symbol}")
        print(f"  Models: {len(ensemble.models)}")
        print(f"  Test accuracy: {ensemble.training_metrics.test_accuracy*100:.2f}%")

        return ensemble
