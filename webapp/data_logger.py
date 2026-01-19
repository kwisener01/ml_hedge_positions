"""
Data Logger for Continuous Learning
Logs predictions, features, and outcomes to CSV for model retraining
"""
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TrainingDataLogger:
    """
    Logs live predictions and features for model retraining
    """

    def __init__(self, data_dir: str = "../data_collected"):
        """
        Initialize data logger

        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.data_dir / "predictions.csv"
        self.features_file = self.data_dir / "features.csv"
        self.outcomes_file = self.data_dir / "outcomes.csv"

        # Initialize CSV files with headers if they don't exist
        self._initialize_files()

        logger.info(f"Data logger initialized: {self.data_dir}")

    def _initialize_files(self):
        """Create CSV files with headers if they don't exist"""

        # Predictions file
        if not self.predictions_file.exists():
            with open(self.predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'price', 'prediction',
                    'prob_up', 'confidence', 'strength_score',
                    'level_type', 'level_source', 'signal_fired'
                ])

        # Features file (stores complete feature vectors)
        if not self.features_file.exists():
            with open(self.features_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'features_json'])

        # Outcomes file (records actual price movements after predictions)
        if not self.outcomes_file.exists():
            with open(self.outcomes_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'prediction_timestamp', 'outcome_timestamp', 'symbol',
                    'entry_price', 'exit_price', 'return_pct',
                    'actual_direction', 'bars_held', 'outcome'
                ])

    def log_prediction(
        self,
        symbol: str,
        price: float,
        prediction: int,
        prob_up: float,
        confidence: float,
        strength_score: float,
        level_type: str,
        level_source: str,
        signal_fired: bool,
        feature_array: np.ndarray,
        feature_dict: Dict
    ):
        """
        Log a prediction with features

        Args:
            symbol: Trading symbol (QQQ, SPY)
            price: Current price
            prediction: Model prediction (0=down, 1=up)
            prob_up: Probability of up move
            confidence: Bayesian confidence score
            strength_score: Greek strength score
            level_type: support/resistance/none
            level_source: Which level (HP/MHP/HG)
            signal_fired: Whether entry signal was triggered
            feature_array: Complete feature vector
            feature_dict: Feature dictionary for reference
        """
        timestamp = datetime.now().isoformat()

        try:
            # Log prediction
            with open(self.predictions_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, symbol, price, prediction,
                    prob_up, confidence, strength_score,
                    level_type, level_source, signal_fired
                ])

            # Log features (as JSON for flexibility)
            with open(self.features_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # Convert numpy array to list for JSON serialization
                features_data = {
                    'feature_array': feature_array.tolist(),
                    'feature_dict': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                                    for k, v in feature_dict.items()}
                }
                writer.writerow([timestamp, symbol, json.dumps(features_data)])

            logger.debug(f"Logged prediction: {symbol} @ {price} = {prediction}")

        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def log_outcome(
        self,
        prediction_timestamp: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        bars_held: int
    ):
        """
        Log the outcome of a prediction after time has passed

        Args:
            prediction_timestamp: Original prediction timestamp
            symbol: Trading symbol
            entry_price: Price when prediction was made
            exit_price: Price after holding period
            bars_held: Number of bars held (e.g., 10 bars = 10 minutes)
        """
        outcome_timestamp = datetime.now().isoformat()
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        actual_direction = 1 if exit_price > entry_price else 0
        outcome = 'win' if return_pct > 0 else 'loss'

        try:
            with open(self.outcomes_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    prediction_timestamp, outcome_timestamp, symbol,
                    entry_price, exit_price, return_pct,
                    actual_direction, bars_held, outcome
                ])

            logger.debug(f"Logged outcome: {symbol} {outcome} ({return_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error logging outcome: {e}")

    def get_training_data_stats(self) -> Dict:
        """
        Get statistics about collected training data

        Returns:
            Dictionary with data collection statistics
        """
        stats = {
            'predictions_count': 0,
            'features_count': 0,
            'outcomes_count': 0,
            'signals_fired': 0,
            'data_size_mb': 0
        }

        try:
            # Count predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    stats['predictions_count'] = sum(1 for _ in f) - 1  # Exclude header

                    # Count signals
                    f.seek(0)
                    reader = csv.DictReader(f)
                    stats['signals_fired'] = sum(1 for row in reader if row['signal_fired'] == 'True')

            # Count features
            if self.features_file.exists():
                with open(self.features_file, 'r') as f:
                    stats['features_count'] = sum(1 for _ in f) - 1

            # Count outcomes
            if self.outcomes_file.exists():
                with open(self.outcomes_file, 'r') as f:
                    stats['outcomes_count'] = sum(1 for _ in f) - 1

            # Calculate total data size
            total_size = sum(
                f.stat().st_size for f in [
                    self.predictions_file,
                    self.features_file,
                    self.outcomes_file
                ] if f.exists()
            )
            stats['data_size_mb'] = round(total_size / (1024 * 1024), 2)

        except Exception as e:
            logger.error(f"Error getting training data stats: {e}")

        return stats

    def is_ready_for_retraining(self, min_samples: int = 100) -> bool:
        """
        Check if enough data has been collected for retraining

        Args:
            min_samples: Minimum number of predictions needed

        Returns:
            True if ready for retraining
        """
        stats = self.get_training_data_stats()
        return stats['predictions_count'] >= min_samples
