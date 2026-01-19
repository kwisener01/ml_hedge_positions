"""
Automated Retraining Script
Retrains model using collected live data combined with historical data
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_ensemble import TrainingDataBuilder
from models.svm_ensemble import SVMEnsemble
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveDataRetrainer:
    """
    Retrains models using live collected data
    """

    def __init__(self, symbol: str, data_dir: str = "../data_collected"):
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.model_dir = Path(__file__).parent.parent / "models/trained"

    def load_live_data(self) -> tuple:
        """
        Load collected live data from CSV files

        Returns:
            (features_df, predictions_df) tuple
        """
        features_file = self.data_dir / "features.csv"
        predictions_file = self.data_dir / "predictions.csv"

        if not features_file.exists() or not predictions_file.exists():
            raise FileNotFoundError("Live data files not found")

        # Load predictions
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])

        # Load features
        features_df = pd.read_csv(features_file)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

        logger.info(f"Loaded {len(features_df)} feature vectors from live data")

        return features_df, predictions_df

    def parse_features(self, features_df: pd.DataFrame) -> tuple:
        """
        Parse JSON feature data into numpy arrays and dicts

        Returns:
            (feature_arrays, feature_dicts) tuple
        """
        feature_arrays = []
        feature_dicts = []

        for idx, row in features_df.iterrows():
            try:
                features_data = json.loads(row['features_json'])
                feature_arrays.append(np.array(features_data['feature_array']))
                feature_dicts.append(features_data['feature_dict'])
            except Exception as e:
                logger.warning(f"Failed to parse features at index {idx}: {e}")
                continue

        return np.array(feature_arrays), feature_dicts

    def create_labels_from_outcomes(self, predictions_df: pd.DataFrame) -> np.ndarray:
        """
        Create training labels from predictions
        For now, use the actual outcomes if available, otherwise use model predictions

        In production, you would:
        1. Wait for actual price movements (10 bars)
        2. Calculate if prediction was correct
        3. Use that as the label

        Args:
            predictions_df: Predictions dataframe

        Returns:
            Labels array (1 = up, 0 = down)
        """
        # Use model predictions as labels for now
        # In production, replace with actual outcomes
        labels = predictions_df['prediction'].values

        logger.info(f"Created {len(labels)} labels")
        logger.info(f"Label distribution: {np.bincount(labels)}")

        return labels

    def combine_with_historical(self, live_X: np.ndarray, live_y: np.ndarray) -> tuple:
        """
        Combine live data with historical training data

        Args:
            live_X: Live feature arrays
            live_y: Live labels

        Returns:
            (combined_X, combined_y) tuple
        """
        # Load historical training data if available
        historical_file = self.model_dir.parent.parent / f"data_local/training_{self.symbol}.pkl"

        if historical_file.exists():
            with open(historical_file, 'rb') as f:
                historical_data = pickle.load(f)
                hist_X = historical_data['X']
                hist_y = historical_data['y']

            logger.info(f"Loaded {len(hist_X)} historical samples")

            # Combine
            combined_X = np.vstack([hist_X, live_X])
            combined_y = np.hstack([hist_y, live_y])

            logger.info(f"Combined dataset: {len(combined_X)} samples")
        else:
            logger.info("No historical data found, using only live data")
            combined_X = live_X
            combined_y = live_y

        return combined_X, combined_y

    def retrain_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "xgboost"):
        """
        Retrain the model with new data

        Args:
            X: Feature matrix
            y: Labels
            model_type: Type of model to train ("xgboost" or "ensemble")
        """
        logger.info(f"Starting {model_type} model retraining")
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Feature dimensions: {X.shape[1]}")

        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if model_type == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='binary:logistic',
                eval_metric='logloss'
            )

            logger.info("Training XGBoost model...")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            logger.info(f"Test Accuracy: {accuracy:.3f}")
            logger.info(f"Test Precision: {precision:.3f}")
            logger.info(f"Test Recall: {recall:.3f}")

            # Save model
            model_filename = self.model_dir / f"{self.symbol}_xgboost_optimized.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'selected_features': None,
                    'training_date': datetime.now().isoformat(),
                    'training_samples': len(X),
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall
                }, f)

            logger.info(f"Model saved: {model_filename}")

        elif model_type == "ensemble":
            # Train SVM ensemble
            ensemble = SVMEnsemble()

            logger.info("Training SVM ensemble...")
            ensemble.train(X_train, y_train)

            # Evaluate
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Test Accuracy: {accuracy:.3f}")

            # Save model
            model_filename = self.model_dir / f"{self.symbol}_ensemble.pkl"
            ensemble.save(str(model_filename))

            logger.info(f"Model saved: {model_filename}")

    def run_retraining(self, min_samples: int = 100, model_type: str = "xgboost"):
        """
        Complete retraining pipeline

        Args:
            min_samples: Minimum live samples required
            model_type: Type of model to train
        """
        logger.info("="*60)
        logger.info(f"Starting retraining for {self.symbol}")
        logger.info("="*60)

        # 1. Load live data
        features_df, predictions_df = self.load_live_data()

        if len(features_df) < min_samples:
            logger.warning(f"Not enough data: {len(features_df)} < {min_samples}")
            logger.warning("Retraining aborted")
            return False

        # 2. Parse features
        X_live, feature_dicts = self.parse_features(features_df)
        y_live = self.create_labels_from_outcomes(predictions_df)

        if len(X_live) != len(y_live):
            logger.error("Feature and label counts don't match!")
            return False

        # 3. Combine with historical
        X_combined, y_combined = self.combine_with_historical(X_live, y_live)

        # 4. Retrain model
        self.retrain_model(X_combined, y_combined, model_type=model_type)

        logger.info("="*60)
        logger.info("Retraining complete!")
        logger.info("="*60)

        return True


def main():
    """
    Main retraining entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Retrain model from live data')
    parser.add_argument('--symbol', default='QQQ', help='Trading symbol')
    parser.add_argument('--min-samples', type=int, default=100, help='Minimum samples required')
    parser.add_argument('--model-type', default='xgboost', choices=['xgboost', 'ensemble'])
    args = parser.parse_args()

    retrainer = LiveDataRetrainer(args.symbol)

    try:
        success = retrainer.run_retraining(
            min_samples=args.min_samples,
            model_type=args.model_type
        )

        if success:
            logger.info("SUCCESS: Model retrained successfully")
            sys.exit(0)
        else:
            logger.warning("SKIPPED: Retraining skipped (not enough data)")
            sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR: Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
