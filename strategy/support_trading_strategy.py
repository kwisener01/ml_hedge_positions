"""
Production Support Trading Strategy

Components:
1. Entry: Moderate strength (15-30) + Support level + UP bias
2. Exit: Multiple exit types (profit target, stop loss, time decay)
3. Bayesian Confidence: Adjust position size based on prediction confidence
4. Risk Management: Max loss per trade, position sizing

Strategy Logic:
- ONLY trade at support levels (not resistance)
- Filter for moderate Greek strength (15-30)
- Use Bayesian updating to quantify confidence
- Dynamic position sizing based on confidence
- Multiple exit conditions
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import model_config
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' only for support strategy
    size: float  # Position size (shares or contracts)
    confidence: float  # Bayesian confidence (0-1)
    strength_score: float  # Greek level strength

    # Exit conditions
    profit_target: float  # Price level
    stop_loss: float  # Price level
    max_bars: int  # Time-based exit

    # Results (filled on exit)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    bars_held: Optional[int] = None

    def is_open(self) -> bool:
        return self.exit_time is None

    def check_exit(self, current_price: float, current_time: datetime, bars_held: int) -> Tuple[bool, str]:
        """
        Check if any exit condition is met

        Returns: (should_exit, reason)
        """
        if current_price >= self.profit_target:
            return True, 'PROFIT_TARGET'

        if current_price <= self.stop_loss:
            return True, 'STOP_LOSS'

        if bars_held >= self.max_bars:
            return True, 'TIME_DECAY'

        return False, ''

    def close_trade(self, exit_price: float, exit_time: datetime, bars_held: int, reason: str):
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.bars_held = bars_held

        # Calculate P&L (LONG only)
        self.pnl = (exit_price - self.entry_price) * self.size


class BayesianConfidenceScorer:
    """
    Bayesian confidence scoring for predictions

    Uses prior probabilities and updates with evidence:
    - Model prediction probability
    - Level strength (stronger = higher confidence)
    - Historical accuracy at similar setups
    """

    def __init__(self):
        # Prior probabilities (from historical data)
        self.prior_support_up = 0.6875  # 68.75% from our analysis
        self.prior_baseline = 0.50  # Random baseline

    def calculate_confidence(
        self,
        model_probability: float,
        strength_score: float,
        level_type: str,
        historical_accuracy: float = None
    ) -> float:
        """
        Calculate Bayesian confidence score

        Args:
            model_probability: Model's probability for UP (0-1)
            strength_score: Greek level strength (0-100)
            level_type: 'support' or 'resistance'
            historical_accuracy: Optional historical win rate at similar setups

        Returns:
            Confidence score (0-1)
        """
        if level_type != 'support':
            # Only high confidence at support
            return min(model_probability, 0.5)

        # Start with prior (support UP bias)
        confidence = self.prior_support_up

        # Update with model probability (Bayesian updating)
        # P(UP|evidence) ∝ P(evidence|UP) * P(UP)
        likelihood = model_probability
        confidence = (likelihood * confidence) / \
                    ((likelihood * confidence) + ((1 - likelihood) * (1 - confidence)))

        # Adjust for strength (moderate strength = higher confidence)
        # Strength 15-30 is optimal, scale accordingly
        if 15 <= strength_score <= 30:
            strength_factor = 1.0 + (25 - abs(strength_score - 22.5)) / 25 * 0.2
        else:
            strength_factor = 0.8  # Lower confidence outside optimal range

        confidence *= strength_factor

        # Incorporate historical accuracy if available
        if historical_accuracy:
            confidence = (confidence + historical_accuracy) / 2

        # Cap between reasonable bounds
        return np.clip(confidence, 0.3, 0.95)


class SupportTradingStrategy:
    """
    Complete support-only trading strategy with entry/exit rules
    """

    def __init__(
        self,
        symbol: str,
        model_path: str,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        profit_target_pct: float = 0.015,  # 1.5% profit target
        stop_loss_pct: float = 0.01,  # 1% stop loss
        max_holding_bars: int = 10  # Exit after 10 bars if no target hit
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars

        self.client = TradierClient()
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureMatrixBuilder(self.client)
        self.confidence_scorer = BayesianConfidenceScorer()

        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.selected_features = model_data.get('selected_features', None)

        # Track trades
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None

    def calculate_level_strength(self, feature_dict: Dict) -> float:
        """Calculate Greek level strength score"""
        score = 0

        # HP proximity (0-10)
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        if hp_support_dist < 0.005:
            score += (0.005 - hp_support_dist) / 0.005 * 10

        # MHP proximity (0-20, weighted higher)
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        if mhp_support_dist < 0.005:
            score += (0.005 - mhp_support_dist) / 0.005 * 20

        return score

    def identify_level_type(self, feature_dict: Dict) -> str:
        """Identify if at support or resistance"""
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))

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

        if support_score > resist_score * 1.2:
            return 'support'
        elif resist_score > support_score * 1.2:
            return 'resistance'
        else:
            return 'neutral'

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """
        Calculate position size based on risk management and confidence

        Uses:
        1. Fixed % risk per trade
        2. Scaled by Bayesian confidence
        """
        # Risk amount in dollars
        risk_amount = self.current_capital * self.risk_per_trade

        # Scale by confidence (higher confidence = larger position)
        # Confidence 0.5 = 50% of normal size
        # Confidence 1.0 = 100% of normal size
        confidence_scalar = (confidence - 0.5) * 2  # Map 0.5-1.0 to 0-1
        confidence_scalar = np.clip(confidence_scalar, 0.3, 1.0)

        risk_amount *= confidence_scalar

        # Calculate position size
        # Risk per share = entry - stop_loss
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return 0

        position_size = risk_amount / risk_per_share

        # Round down to whole shares
        return int(position_size)

    def should_enter(
        self,
        feature_dict: Dict,
        feature_array: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Determine if should enter trade

        Returns: (should_enter, confidence, strength_score)
        """
        # Calculate strength
        strength = self.calculate_level_strength(feature_dict)

        # Filter: Moderate strength only (15-30)
        if not (15 <= strength <= 30):
            return False, 0, strength

        # Filter: Support only
        level_type = self.identify_level_type(feature_dict)
        if level_type != 'support':
            return False, 0, strength

        # Get model probability
        # XGBoost predict_proba returns [prob_down, prob_up]
        probs = self.model.predict_proba(feature_array.reshape(1, -1))[0]
        prob_up = probs[1]

        # Calculate Bayesian confidence
        confidence = self.confidence_scorer.calculate_confidence(
            model_probability=prob_up,
            strength_score=strength,
            level_type=level_type
        )

        # Minimum confidence threshold
        if confidence < 0.55:
            return False, confidence, strength

        return True, confidence, strength

    def enter_trade(
        self,
        entry_price: float,
        entry_time: datetime,
        confidence: float,
        strength_score: float
    ):
        """Enter a new LONG trade at support"""
        if self.open_trade is not None:
            return  # Already in a trade

        # Calculate exit levels
        profit_target = entry_price * (1 + self.profit_target_pct)
        stop_loss = entry_price * (1 - self.stop_loss_pct)

        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss, confidence)

        if position_size == 0:
            return  # Can't size position

        # Create trade
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction='LONG',
            size=position_size,
            confidence=confidence,
            strength_score=strength_score,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_bars=self.max_holding_bars
        )

        self.open_trade = trade

    def check_and_exit(
        self,
        current_price: float,
        current_time: datetime,
        bars_held: int
    ):
        """Check if should exit current trade"""
        if self.open_trade is None:
            return

        should_exit, reason = self.open_trade.check_exit(
            current_price, current_time, bars_held
        )

        if should_exit:
            self.open_trade.close_trade(current_price, current_time, bars_held, reason)

            # Update capital
            self.current_capital += self.open_trade.pnl

            # Move to closed trades
            self.trades.append(self.open_trade)
            self.open_trade = None

    def backtest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_dict_list: List[Dict],
        prices: List[float],
        timestamps: List[datetime],
        all_feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Backtest the strategy

        Returns: DataFrame with trade results
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING SUPPORT STRATEGY: {self.symbol}")
        print(f"{'='*80}")

        # Filter features if needed
        if self.selected_features:
            selected_indices = [all_feature_names.index(f) for f in self.selected_features if f in all_feature_names]
            X = X[:, selected_indices]

        # Simulate trading
        print(f"\nSimulating trades on {len(X)} bars...")

        bars_in_trade = 0

        for i in range(len(X)):
            current_price = prices[i]
            current_time = timestamps[i] if i < len(timestamps) else datetime.now()
            feature_dict = feature_dict_list[i]
            feature_array = X[i]

            # Check exit conditions for open trade
            if self.open_trade:
                bars_in_trade += 1
                self.check_and_exit(current_price, current_time, bars_in_trade)

            # Check entry conditions (only if not in trade)
            if self.open_trade is None:
                bars_in_trade = 0
                should_enter, confidence, strength = self.should_enter(
                    feature_dict, feature_array
                )

                if should_enter:
                    self.enter_trade(current_price, current_time, confidence, strength)

        # Close any remaining open trade
        if self.open_trade:
            final_price = prices[-1]
            final_time = timestamps[-1] if timestamps else datetime.now()
            self.open_trade.close_trade(final_price, final_time, bars_in_trade, 'END_OF_DATA')
            self.current_capital += self.open_trade.pnl
            self.trades.append(self.open_trade)
            self.open_trade = None

        # Generate results
        return self.generate_results()

    def generate_results(self) -> pd.DataFrame:
        """Generate backtest results and statistics"""
        if not self.trades:
            print("\n[WARNING] No trades executed")
            return pd.DataFrame()

        # Convert trades to DataFrame
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'size': trade.size,
                'confidence': trade.confidence,
                'strength': trade.strength_score,
                'bars_held': trade.bars_held,
                'pnl': trade.pnl,
                'return_pct': (trade.exit_price - trade.entry_price) / trade.entry_price * 100
            })

        df = pd.DataFrame(trades_data)

        # Calculate statistics
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}")

        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100

        print(f"\nOverall Performance:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Winning Trades: {winning_trades} ({win_rate*100:.2f}%)")
        print(f"  Losing Trades: {losing_trades}")
        print(f"  Win Rate: {win_rate*100:.2f}%")
        print(f"\n  Total P&L: ${total_pnl:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Avg Win: ${avg_win:,.2f}")
        print(f"  Avg Loss: ${avg_loss:,.2f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"\n  Starting Capital: ${self.initial_capital:,.2f}")
        print(f"  Ending Capital: ${self.current_capital:,.2f}")

        # Exit reason breakdown
        print(f"\nExit Reasons:")
        for reason, count in df['exit_reason'].value_counts().items():
            pct = count / total_trades * 100
            avg_pnl = df[df['exit_reason'] == reason]['pnl'].mean()
            print(f"  {reason:<20} {count:>3} ({pct:>5.1f}%)  Avg P&L: ${avg_pnl:>8,.2f}")

        # Confidence analysis
        print(f"\nConfidence Analysis:")
        print(f"  Avg Confidence: {df['confidence'].mean():.3f}")
        print(f"  Min Confidence: {df['confidence'].min():.3f}")
        print(f"  Max Confidence: {df['confidence'].max():.3f}")

        # High vs low confidence
        high_conf = df[df['confidence'] > 0.7]
        low_conf = df[df['confidence'] <= 0.7]

        if len(high_conf) > 0:
            print(f"\n  High Confidence (>0.7): {len(high_conf)} trades, {(high_conf['pnl'] > 0).sum() / len(high_conf) * 100:.1f}% win rate")
        if len(low_conf) > 0:
            print(f"  Low Confidence (<=0.7): {len(low_conf)} trades, {(low_conf['pnl'] > 0).sum() / len(low_conf) * 100:.1f}% win rate")

        return df


def load_and_prepare_data(symbol: str, interval: str = '1min'):
    """Load intraday data and prepare for backtesting"""
    client = TradierClient()
    cleaner = DataCleaner()
    feature_builder = FeatureMatrixBuilder(client)

    # Load data
    intraday_dir = Path(__file__).parent.parent / "data_local/intraday"
    pattern = f"{symbol}_{interval}_*.csv"
    files = list(intraday_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {interval} data found for {symbol}")

    filepath = sorted(files)[-1]
    df = pd.read_csv(filepath, parse_dates=['time'])

    # Convert to quotes
    quotes = []
    for idx, row in df.iterrows():
        price = row['close']
        bar_range = row['high'] - row['low']
        spread = max(bar_range / 4, price * 0.0002)

        quote = QuoteData(
            symbol=symbol,
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
    cleaned_quotes = cleaner.clean_quotes(quotes)
    window_builder = EventWindowBuilder()
    windows = []
    window_end_quotes = []

    for quote in cleaned_quotes:
        window = window_builder.add_event(quote)
        if window:
            windows.append(window)
            window_end_quotes.append(quote)

    # Build feature dataset
    X_list = []
    feature_dict_list = []
    prices = []
    timestamps = []

    for i in range(len(windows)):
        try:
            current_quote = window_end_quotes[i]
            feature_array, feature_dict = feature_builder.build_feature_vector(
                symbol, windows[i], current_quote
            )

            X_list.append(feature_array)
            feature_dict_list.append(feature_dict)
            prices.append(current_quote.mid_price)
            timestamps.append(current_quote.timestamp)

        except Exception as e:
            continue

    X = np.vstack(X_list)
    all_feature_names = list(feature_dict_list[0].keys())

    return X, feature_dict_list, prices, timestamps, all_feature_names


def run_strategy(symbol: str = 'QQQ'):
    """Run the complete support strategy"""
    print("\n" + "="*80)
    print("SUPPORT-ONLY TRADING STRATEGY WITH EXIT RULES")
    print("="*80)

    # Load model
    model_path = Path(__file__).parent.parent / f"models/trained/{symbol}_xgboost_optimized.pkl"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Load data
    X, feature_dict_list, prices, timestamps, all_feature_names = load_and_prepare_data(symbol)

    # Create dummy y for compatibility (not used in backtesting)
    y = np.zeros(len(X))

    # Initialize strategy
    strategy = SupportTradingStrategy(
        symbol=symbol,
        model_path=str(model_path),
        initial_capital=100000,
        risk_per_trade=0.02,  # 2% risk per trade
        profit_target_pct=0.015,  # 1.5% profit target
        stop_loss_pct=0.01,  # 1% stop loss
        max_holding_bars=10  # Max 10 bars (50 minutes at 5-bar windows)
    )

    # Run backtest
    results_df = strategy.backtest(
        X, y, feature_dict_list, prices, timestamps, all_feature_names
    )

    # Save results
    if not results_df.empty:
        output_path = Path(__file__).parent.parent / f"backtest_results_{symbol}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")


if __name__ == "__main__":
    run_strategy('QQQ')
