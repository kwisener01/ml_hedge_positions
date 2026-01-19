"""
Signal Integration Module
Combines SVM ensemble predictions with institutional layer signals
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder
from institutional.hedge_pressure import HedgePressureCalculator
from institutional.monthly_hp import MonthlyHPCalculator
from institutional.half_gap import HalfGapCalculator
from models.svm_ensemble import SVMEnsemble, SVMPrediction


class SignalType(Enum):
    """Trading signal types"""
    NO_SIGNAL = 0
    LONG = 1
    SHORT = -1


class SignalStrength(Enum):
    """Signal confidence levels"""
    WEAK = 1
    MEDIUM = 2
    STRONG = 3


@dataclass
class TradingSignal:
    """Complete trading signal with all context"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    signal_strength: SignalStrength

    # Price context
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float

    # SVM prediction
    svm_prediction: int  # -1, 0, 1
    svm_confidence: float

    # Institutional context
    hp_score: float
    mhp_score: float
    key_support: Optional[float]
    key_resistance: Optional[float]
    nearest_hg_above: Optional[float]
    nearest_hg_below: Optional[float]

    # Signal reasoning
    confluence_score: float  # 0-1, how well SVM + institutional agree
    reasoning: str

    def __str__(self) -> str:
        """Format signal for display"""
        direction = {
            SignalType.LONG: "LONG",
            SignalType.SHORT: "SHORT",
            SignalType.NO_SIGNAL: "NO SIGNAL"
        }[self.signal_type]

        strength = self.signal_strength.name

        output = f"\n{'='*80}\n"
        output += f"TRADING SIGNAL: {self.symbol}\n"
        output += f"{'='*80}\n"
        output += f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Signal: {direction} ({strength})\n"
        output += f"Confluence: {self.confluence_score*100:.1f}%\n\n"

        output += f"PRICE LEVELS:\n"
        output += f"  Current:     ${self.current_price:.2f}\n"
        output += f"  Entry:       ${self.entry_price:.2f}\n"
        output += f"  Stop Loss:   ${self.stop_loss:.2f} ({abs(self.stop_loss-self.entry_price)/self.entry_price*100:.2f}%)\n"
        output += f"  Take Profit: ${self.take_profit:.2f} ({abs(self.take_profit-self.entry_price)/self.entry_price*100:.2f}%)\n\n"

        output += f"SVM PREDICTION:\n"
        output += f"  Direction: {self.svm_prediction:+d}\n"
        output += f"  Confidence: {self.svm_confidence*100:.1f}%\n\n"

        output += f"INSTITUTIONAL CONTEXT:\n"
        output += f"  HP Score: {self.hp_score:+.3f}\n"
        output += f"  MHP Score: {self.mhp_score:+.3f}\n"
        if self.key_support:
            output += f"  Key Support: ${self.key_support:.2f}\n"
        if self.key_resistance:
            output += f"  Key Resistance: ${self.key_resistance:.2f}\n"
        if self.nearest_hg_below:
            output += f"  HG Below: ${self.nearest_hg_below:.2f}\n"
        if self.nearest_hg_above:
            output += f"  HG Above: ${self.nearest_hg_above:.2f}\n"

        output += f"\nREASONING:\n{self.reasoning}\n"
        output += f"{'='*80}\n"

        return output


class SignalGenerator:
    """
    Generates trading signals by combining:
    1. SVM ensemble predictions
    2. Institutional levels (HP/MHP/HG)
    3. Risk management rules
    """

    def __init__(
        self,
        symbol: str,
        model_path: Optional[str] = None,
        min_svm_confidence: float = 0.6,
        min_confluence: float = 0.5,
        risk_percent: float = 0.01,  # 1% risk per trade
        reward_risk_ratio: float = 2.0  # 2:1 reward:risk
    ):
        self.symbol = symbol
        self.min_svm_confidence = min_svm_confidence
        self.min_confluence = min_confluence
        self.risk_percent = risk_percent
        self.reward_risk_ratio = reward_risk_ratio

        # Load trained ensemble
        if model_path is None:
            model_path = Path(__file__).parent.parent / f"models/trained/{symbol}_ensemble.pkl"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.ensemble = SVMEnsemble.load(str(model_path))

        # Initialize components
        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.window_builder = EventWindowBuilder()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.hp_calc = HedgePressureCalculator()
        self.mhp_calc = MonthlyHPCalculator()
        self.hg_calc = HalfGapCalculator()

        print(f"[OK] SignalGenerator initialized for {symbol}")
        print(f"     Min SVM Confidence: {min_svm_confidence*100:.0f}%")
        print(f"     Min Confluence: {min_confluence*100:.0f}%")
        print(f"     Risk per trade: {risk_percent*100:.1f}%")
        print(f"     Reward:Risk: {reward_risk_ratio}:1")

    def collect_market_data(self) -> Tuple[float, Dict]:
        """
        Collect current market data and institutional context

        Returns:
            (current_price, institutional_data)
        """
        # Get current quote
        quote = self.client.get_quote(self.symbol)
        if not quote:
            raise ValueError(f"Failed to get quote for {self.symbol}")

        spot = quote.mid_price

        # Get institutional data
        institutional = {}

        # HP
        chains = self.client.get_multiple_chains(self.symbol, expiration_count=1)
        if chains:
            exp, chain = list(chains.items())[0]
            hp_result = self.hp_calc.calculate(chain, spot, self.symbol)
            institutional['hp'] = hp_result
        else:
            institutional['hp'] = None

        # MHP
        chains_multi = self.client.get_multiple_chains(self.symbol, expiration_count=4)
        if chains_multi:
            mhp_result = self.mhp_calc.calculate(chains_multi, spot, self.symbol)
            institutional['mhp'] = mhp_result
        else:
            institutional['mhp'] = None

        # HG
        history = self.client.get_history(self.symbol, interval="daily")
        if not history.empty:
            hg_result = self.hg_calc.calculate(history, spot, self.symbol)
            institutional['hg'] = hg_result
        else:
            institutional['hg'] = None

        return spot, institutional

    def get_svm_prediction(self) -> Optional[SVMPrediction]:
        """
        Collect window and get SVM prediction

        Returns:
            SVMPrediction or None if failed
        """
        print(f"\nCollecting window for {self.symbol}...")

        # Collect quotes for window
        quotes = []
        for _ in range(model_config.window_size):
            quote = self.client.get_quote(self.symbol)
            if quote:
                cleaned = self.cleaner.clean_quotes([quote])
                if cleaned:
                    quotes.append(cleaned[0])
            import time
            time.sleep(1)

        if len(quotes) < model_config.window_size:
            print(f"[FAILED] Only collected {len(quotes)}/{model_config.window_size} quotes")
            return None

        # Build window
        window = None
        for quote in quotes:
            window = self.window_builder.add_event(quote)
            if window:
                break

        if not window:
            print("[FAILED] Could not form window")
            return None

        # Build features
        feature_array, _ = self.feature_builder.build_feature_vector(
            self.symbol, window, quotes[-1]
        )

        # Get prediction
        prediction = self.ensemble.predict(feature_array)
        print(f"[OK] SVM Prediction: {prediction.prediction:+d} ({prediction.confidence*100:.0f}% confidence)")

        return prediction

    def calculate_confluence(
        self,
        svm_pred: SVMPrediction,
        institutional: Dict
    ) -> Tuple[float, str]:
        """
        Calculate how well SVM and institutional signals align

        Returns:
            (confluence_score, reasoning)
        """
        reasoning_parts = []
        agreement_scores = []

        # Check SVM direction vs HP
        if institutional['hp']:
            hp = institutional['hp']
            hp_bullish = hp.net_hp > 0
            svm_bullish = svm_pred.prediction > 0

            if hp_bullish == svm_bullish:
                agreement_scores.append(1.0)
                reasoning_parts.append(f"[OK] SVM and HP agree ({hp.dominant_direction})")
            else:
                agreement_scores.append(0.0)
                reasoning_parts.append(f"[X] SVM and HP disagree (HP: {hp.dominant_direction})")

        # Check SVM direction vs MHP
        if institutional['mhp']:
            mhp = institutional['mhp']
            mhp_bullish = mhp.mhp_score > 0
            svm_bullish = svm_pred.prediction > 0

            if mhp_bullish == svm_bullish:
                agreement_scores.append(1.0)
                reasoning_parts.append(f"[OK] SVM and MHP agree (MHP: {mhp.mhp_score:+.3f})")
            else:
                agreement_scores.append(0.0)
                reasoning_parts.append(f"[X] SVM and MHP disagree (MHP: {mhp.mhp_score:+.3f})")

        # Check proximity to institutional levels
        if institutional['hp']:
            hp = institutional['hp']
            current_price = svm_pred.feature_vector[0]  # V1 is current mid price

            # If going long, check distance to support
            if svm_pred.prediction > 0 and hp.key_support:
                distance_pct = abs(current_price - hp.key_support) / current_price
                if distance_pct < 0.01:  # Within 1%
                    agreement_scores.append(1.0)
                    reasoning_parts.append(f"[OK] Near HP support level (${hp.key_support:.2f})")
                elif distance_pct < 0.02:  # Within 2%
                    agreement_scores.append(0.5)
                    reasoning_parts.append(f"[~] Moderately near HP support (${hp.key_support:.2f})")

            # If going short, check distance to resistance
            elif svm_pred.prediction < 0 and hp.key_resistance:
                distance_pct = abs(current_price - hp.key_resistance) / current_price
                if distance_pct < 0.01:
                    agreement_scores.append(1.0)
                    reasoning_parts.append(f"[OK] Near HP resistance level (${hp.key_resistance:.2f})")
                elif distance_pct < 0.02:
                    agreement_scores.append(0.5)
                    reasoning_parts.append(f"[~] Moderately near HP resistance (${hp.key_resistance:.2f})")

        # Calculate final score
        if agreement_scores:
            confluence = sum(agreement_scores) / len(agreement_scores)
        else:
            confluence = 0.5  # Neutral if no institutional data
            reasoning_parts.append("! No institutional data available")

        reasoning = "\n  ".join(reasoning_parts)

        return confluence, reasoning

    def calculate_risk_levels(
        self,
        entry_price: float,
        signal_direction: int,
        institutional: Dict
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Entry price
            signal_direction: 1 for long, -1 for short
            institutional: Institutional data

        Returns:
            (stop_loss, take_profit)
        """
        # Default risk based on percentage
        risk_amount = entry_price * self.risk_percent

        # Adjust stop based on institutional levels
        if signal_direction > 0:  # LONG
            # Stop below support or default
            stop_loss = entry_price - risk_amount

            if institutional['hp'] and institutional['hp'].key_support:
                support = institutional['hp'].key_support
                if support < entry_price:
                    # Stop slightly below support
                    stop_loss = max(stop_loss, support * 0.995)

            # Take profit above resistance or default
            take_profit = entry_price + (risk_amount * self.reward_risk_ratio)

            if institutional['hp'] and institutional['hp'].key_resistance:
                resistance = institutional['hp'].key_resistance
                if resistance > entry_price:
                    # Target slightly below resistance
                    take_profit = min(take_profit, resistance * 0.995)

        else:  # SHORT
            # Stop above resistance or default
            stop_loss = entry_price + risk_amount

            if institutional['hp'] and institutional['hp'].key_resistance:
                resistance = institutional['hp'].key_resistance
                if resistance > entry_price:
                    # Stop slightly above resistance
                    stop_loss = min(stop_loss, resistance * 1.005)

            # Take profit below support or default
            take_profit = entry_price - (risk_amount * self.reward_risk_ratio)

            if institutional['hp'] and institutional['hp'].key_support:
                support = institutional['hp'].key_support
                if support < entry_price:
                    # Target slightly above support
                    take_profit = max(take_profit, support * 1.005)

        return stop_loss, take_profit

    def generate_signal(self) -> Optional[TradingSignal]:
        """
        Generate complete trading signal

        Returns:
            TradingSignal or None if no signal
        """
        print(f"\n{'='*80}")
        print(f"GENERATING SIGNAL: {self.symbol}")
        print(f"{'='*80}")

        # Get market data
        try:
            current_price, institutional = self.collect_market_data()
            print(f"[OK] Current price: ${current_price:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to collect market data: {e}")
            return None

        # Get SVM prediction
        svm_pred = self.get_svm_prediction()
        if not svm_pred:
            return None

        # Check minimum SVM confidence
        if svm_pred.confidence < self.min_svm_confidence:
            print(f"[NO SIGNAL] SVM confidence {svm_pred.confidence*100:.0f}% below minimum {self.min_svm_confidence*100:.0f}%")
            return None

        # Check for neutral prediction
        if svm_pred.prediction == 0:
            print("[NO SIGNAL] SVM predicts neutral movement")
            return None

        # Calculate confluence
        confluence, reasoning = self.calculate_confluence(svm_pred, institutional)
        print(f"[OK] Confluence score: {confluence*100:.0f}%")

        # Check minimum confluence
        if confluence < self.min_confluence:
            print(f"[NO SIGNAL] Confluence {confluence*100:.0f}% below minimum {self.min_confluence*100:.0f}%")
            return None

        # Determine signal type
        signal_type = SignalType.LONG if svm_pred.prediction > 0 else SignalType.SHORT

        # Determine signal strength
        if confluence >= 0.8 and svm_pred.confidence >= 0.8:
            signal_strength = SignalStrength.STRONG
        elif confluence >= 0.65 and svm_pred.confidence >= 0.65:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK

        # Calculate risk levels
        stop_loss, take_profit = self.calculate_risk_levels(
            current_price, svm_pred.prediction, institutional
        )

        # Build signal
        signal = TradingSignal(
            symbol=self.symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            signal_strength=signal_strength,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            svm_prediction=svm_pred.prediction,
            svm_confidence=svm_pred.confidence,
            hp_score=institutional['hp'].net_hp if institutional['hp'] else 0.0,
            mhp_score=institutional['mhp'].mhp_score if institutional['mhp'] else 0.0,
            key_support=institutional['hp'].key_support if institutional['hp'] else None,
            key_resistance=institutional['hp'].key_resistance if institutional['hp'] else None,
            nearest_hg_above=institutional['hg'].nearest_hg_above if institutional['hg'] else None,
            nearest_hg_below=institutional['hg'].nearest_hg_below if institutional['hg'] else None,
            confluence_score=confluence,
            reasoning=reasoning
        )

        print(f"[SIGNAL GENERATED] {signal_type.name} ({signal_strength.name})")

        return signal


def demo():
    """Demonstrate signal generation"""
    print("\n" + "="*80)
    print("SIGNAL GENERATION DEMO")
    print("="*80)

    if not tradier_config.api_key:
        print("[ERROR] TRADIER_API_KEY not set")
        return

    symbols = ["SPY", "QQQ"]

    for symbol in symbols:
        try:
            generator = SignalGenerator(
                symbol=symbol,
                min_svm_confidence=0.55,  # Relaxed for demo
                min_confluence=0.4,  # Relaxed for demo
                risk_percent=0.01,
                reward_risk_ratio=2.0
            )

            signal = generator.generate_signal()

            if signal:
                print(signal)
            else:
                print(f"\n[NO SIGNAL] No trading signal generated for {symbol}\n")

        except FileNotFoundError as e:
            print(f"\n[ERROR] {symbol}: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {symbol}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()
