"""
Live Prediction Engine
Makes real-time predictions using trained SVM ensemble
"""
import sys
import time
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

sys.path.append('..')
from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder
from models.svm_ensemble import SVMEnsemble, SVMPrediction


class LivePredictor:
    """
    Real-time prediction engine using trained ensemble

    Manages:
    - Quote collection and windowing
    - Feature extraction
    - Ensemble prediction
    - Signal generation
    """

    def __init__(self, symbol: str, model_path: Optional[str] = None):
        self.symbol = symbol
        self.model_path = model_path or f"models/trained/{symbol}_ensemble.pkl"

        # Load trained ensemble
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.ensemble = SVMEnsemble.load(self.model_path)

        # Initialize data pipeline
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.window_builder = EventWindowBuilder()
        self.feature_builder = FeatureMatrixBuilder(self.client)

        print(f"[OK] LivePredictor initialized for {symbol}")

    def collect_window(self, timeout: int = 60) -> bool:
        """
        Collect quotes until a complete window is formed

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if window collected, False if timeout
        """
        print(f"\nCollecting window for {self.symbol}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Get quote
            quote = self.client.get_quote(self.symbol)

            if quote:
                # Clean
                cleaned = self.cleaner.clean_quotes([quote])

                if cleaned:
                    # Try to form window
                    window = self.window_builder.add_event(cleaned[0])

                    if window:
                        self.current_window = window
                        self.current_quote = cleaned[0]
                        print(f"[OK] Window complete: {len(window)} events")
                        return True

            time.sleep(1)

        print(f"[TIMEOUT] Failed to collect window in {timeout}s")
        return False

    def predict(self) -> SVMPrediction:
        """
        Make prediction using current window

        Returns:
            SVMPrediction result
        """
        if not hasattr(self, 'current_window'):
            raise ValueError("No window available. Call collect_window() first.")

        # Build features
        feature_array, feature_dict = self.feature_builder.build_feature_vector(
            self.symbol,
            self.current_window,
            self.current_quote
        )

        # Get prediction
        prediction = self.ensemble.predict(feature_array)

        return prediction

    def predict_live(self, window_timeout: int = 60) -> Optional[SVMPrediction]:
        """
        Convenience method: collect window and predict

        Args:
            window_timeout: Max seconds to wait for window

        Returns:
            SVMPrediction or None if failed
        """
        if self.collect_window(timeout=window_timeout):
            return self.predict()
        return None

    def format_prediction(self, pred: SVMPrediction) -> str:
        """Format prediction for display"""
        direction_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
        direction = direction_map[pred.prediction]

        output = f"\n{'='*60}\n"
        output += f"PREDICTION: {pred.symbol}\n"
        output += f"{'='*60}\n"
        output += f"Timestamp: {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Direction: {direction} ({pred.prediction:+d})\n"
        output += f"Confidence: {pred.confidence*100:.1f}%\n"
        output += f"\nVote Distribution:\n"
        output += f"  DOWN    ({-1:2}): {pred.vote_distribution[-1]:3} votes\n"
        output += f"  NEUTRAL ({ 0:2}): {pred.vote_distribution[0]:3} votes\n"
        output += f"  UP      ({+1:2}): {pred.vote_distribution[1]:3} votes\n"
        output += f"{'='*60}\n"

        return output


class MultiSymbolPredictor:
    """Manage predictions for multiple symbols"""

    def __init__(self, symbols: list = None):
        self.symbols = symbols or ["SPY", "QQQ"]
        self.predictors: Dict[str, LivePredictor] = {}

        # Initialize predictors
        for symbol in self.symbols:
            try:
                self.predictors[symbol] = LivePredictor(symbol)
            except FileNotFoundError:
                print(f"[WARNING] No model found for {symbol}, skipping")

    def predict_all(self, window_timeout: int = 60) -> Dict[str, SVMPrediction]:
        """
        Get predictions for all symbols

        Args:
            window_timeout: Max seconds to wait per symbol

        Returns:
            Dictionary of {symbol: prediction}
        """
        predictions = {}

        for symbol, predictor in self.predictors.items():
            print(f"\nProcessing {symbol}...")
            pred = predictor.predict_live(window_timeout=window_timeout)

            if pred:
                predictions[symbol] = pred
                print(predictor.format_prediction(pred))
            else:
                print(f"[FAILED] Could not generate prediction for {symbol}")

        return predictions


def demo_live_prediction():
    """Demonstration of live prediction"""
    print("\n" + "="*80)
    print("LIVE PREDICTION DEMO")
    print("="*80)

    # Check API key
    if not tradier_config.api_key:
        print("[ERROR] TRADIER_API_KEY not set")
        return

    # Predict for both symbols
    multi_predictor = MultiSymbolPredictor(["SPY", "QQQ"])
    predictions = multi_predictor.predict_all(window_timeout=60)

    # Summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    for symbol, pred in predictions.items():
        direction_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
        print(f"{symbol}: {direction_map[pred.prediction]} ({pred.confidence*100:.0f}% confidence)")

    print()


if __name__ == "__main__":
    demo_live_prediction()
