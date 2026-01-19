"""
Backtesting Framework
Test signal generator on historical data
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.signal_generator import TradingSignal, SignalType


@dataclass
class Trade:
    """Executed trade record"""
    symbol: str
    entry_time: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    stop_loss: float
    take_profit: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""  # "stop", "target", "signal_reversal", "end_of_data"

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl(self) -> float:
        """Profit/Loss in dollars"""
        if not self.exit_price:
            return 0.0
        return (self.exit_price - self.entry_price) * self.direction * self.size

    @property
    def pnl_percent(self) -> float:
        """Profit/Loss as percentage"""
        if not self.exit_price:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * self.direction * 100

    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds"""
        if not self.exit_time:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass
class BacktestResult:
    """Complete backtest performance metrics"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl < 0])

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if not t.is_open)

    @property
    def total_return_pct(self) -> float:
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if not t.is_open and t.pnl > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if not t.is_open and t.pnl < 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profits / Gross losses"""
        gross_profit = sum(t.pnl for t in self.trades if not t.is_open and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if not t.is_open and t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown in dollars"""
        if not self.equity_curve:
            return 0.0

        peak = self.initial_capital
        max_dd = 0.0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage"""
        return (self.max_drawdown / self.initial_capital) * 100

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming risk-free rate = 0)"""
        if not self.trades:
            return 0.0

        returns = [t.pnl_percent for t in self.trades if not t.is_open]
        if not returns:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming ~252 trading days)
        return (avg_return / std_return) * np.sqrt(252)

    def summary(self) -> str:
        """Format performance summary"""
        output = f"\n{'='*80}\n"
        output += f"BACKTEST RESULTS: {self.symbol}\n"
        output += f"{'='*80}\n\n"

        output += f"Period: {self.start_date.date()} to {self.end_date.date()}\n"
        output += f"Initial Capital: ${self.initial_capital:,.2f}\n"
        output += f"Final Capital: ${self.initial_capital + self.total_pnl:,.2f}\n\n"

        output += f"PERFORMANCE:\n"
        output += f"  Total Return: ${self.total_pnl:,.2f} ({self.total_return_pct:+.2f}%)\n"
        output += f"  Total Trades: {self.total_trades}\n"
        output += f"  Win Rate: {self.win_rate*100:.1f}% ({self.winning_trades}W / {self.losing_trades}L)\n"
        output += f"  Avg Win: ${self.avg_win:,.2f}\n"
        output += f"  Avg Loss: ${self.avg_loss:,.2f}\n"
        output += f"  Profit Factor: {self.profit_factor:.2f}\n"
        output += f"  Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)\n"
        output += f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n\n"

        # Recent trades
        if self.trades:
            output += f"RECENT TRADES (last 5):\n"
            for trade in self.trades[-5:]:
                direction = "LONG" if trade.direction > 0 else "SHORT"
                output += f"  {trade.entry_time.date()} {direction} @ ${trade.entry_price:.2f} → "
                if trade.exit_time:
                    output += f"${trade.exit_price:.2f} = {trade.pnl:+.2f} ({trade.pnl_percent:+.2f}%) [{trade.exit_reason}]\n"
                else:
                    output += f"OPEN\n"

        output += f"\n{'='*80}\n"

        return output


class Backtester:
    """
    Backtest signal generator on historical data

    Simulates trading based on generated signals
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.1,  # 10% of capital per trade
        max_positions: int = 1  # Only one position at a time
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

    def run_backtest(
        self,
        symbol: str,
        signals: List[TradingSignal],
        price_data: pd.DataFrame
    ) -> BacktestResult:
        """
        Run backtest on list of signals

        Args:
            symbol: Trading symbol
            signals: List of TradingSignal objects
            price_data: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            BacktestResult with performance metrics
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST: {symbol}")
        print(f"{'='*80}")
        print(f"Signals to test: {len(signals)}")
        print(f"Price data bars: {len(price_data)}")
        print(f"Initial capital: ${self.initial_capital:,.2f}\n")

        result = BacktestResult(
            symbol=symbol,
            start_date=price_data.index.min(),
            end_date=price_data.index.max(),
            initial_capital=self.initial_capital
        )

        current_capital = self.initial_capital
        open_trades: List[Trade] = []

        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.timestamp)

        # Process each signal
        for i, signal in enumerate(signals):
            signal_time = signal.timestamp

            # Check for exits on open trades first
            for trade in open_trades[:]:
                # Find next bars after trade entry
                future_bars = price_data[price_data.index > trade.entry_time]

                if future_bars.empty:
                    continue

                # Check each bar for stop/target hits
                for bar_time, bar in future_bars.iterrows():
                    # Check stop loss
                    if trade.direction > 0:  # Long
                        if bar['low'] <= trade.stop_loss:
                            trade.exit_time = bar_time
                            trade.exit_price = trade.stop_loss
                            trade.exit_reason = "stop"
                            current_capital += trade.pnl
                            open_trades.remove(trade)
                            result.trades.append(trade)
                            break

                        # Check take profit
                        if bar['high'] >= trade.take_profit:
                            trade.exit_time = bar_time
                            trade.exit_price = trade.take_profit
                            trade.exit_reason = "target"
                            current_capital += trade.pnl
                            open_trades.remove(trade)
                            result.trades.append(trade)
                            break

                    else:  # Short
                        if bar['high'] >= trade.stop_loss:
                            trade.exit_time = bar_time
                            trade.exit_price = trade.stop_loss
                            trade.exit_reason = "stop"
                            current_capital += trade.pnl
                            open_trades.remove(trade)
                            result.trades.append(trade)
                            break

                        # Check take profit
                        if bar['low'] <= trade.take_profit:
                            trade.exit_time = bar_time
                            trade.exit_price = trade.take_profit
                            trade.exit_reason = "target"
                            current_capital += trade.pnl
                            open_trades.remove(trade)
                            result.trades.append(trade)
                            break

            # Check if we can take new signal
            if signal.signal_type == SignalType.NO_SIGNAL:
                continue

            if len(open_trades) >= self.max_positions:
                continue

            # Calculate position size
            position_value = current_capital * self.position_size_pct
            position_size = position_value / signal.entry_price

            # Create trade
            direction = 1 if signal.signal_type == SignalType.LONG else -1

            trade = Trade(
                symbol=symbol,
                entry_time=signal.timestamp,
                entry_price=signal.entry_price,
                direction=direction,
                size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            open_trades.append(trade)

            # Record equity
            result.equity_curve.append((signal.timestamp, current_capital))

            print(f"[{i+1}/{len(signals)}] {'LONG' if direction > 0 else 'SHORT'} @ ${signal.entry_price:.2f}")

        # Close any remaining open trades at last price
        if open_trades and not price_data.empty:
            last_price = price_data.iloc[-1]['close']
            last_time = price_data.index[-1]

            for trade in open_trades:
                trade.exit_time = last_time
                trade.exit_price = last_price
                trade.exit_reason = "end_of_data"
                current_capital += trade.pnl
                result.trades.append(trade)

        # Final equity
        result.equity_curve.append((price_data.index[-1], current_capital))

        print(f"\n{result.summary()}")

        return result


def demo():
    """Demonstrate backtesting (simplified example)"""
    print("\n" + "="*80)
    print("BACKTESTING DEMO")
    print("="*80)
    print("\nNOTE: This is a simplified demo.")
    print("For full backtest, generate signals on historical data first.\n")

    # For demo, just show how to use the backtester
    # In reality, you'd generate signals from historical data

    print("To run a real backtest:")
    print("1. Generate signals on historical price data")
    print("2. Load the corresponding OHLC price data")
    print("3. Pass both to Backtester.run_backtest()")
    print("\nExample code:")
    print("""
    # Load price data
    price_data = pd.read_csv('data_local/price_history/SPY_daily.csv',
                             index_col='date', parse_dates=True)

    # Generate signals (you'd loop through historical data)
    signals = [...]  # List of TradingSignal objects

    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run_backtest('SPY', signals, price_data)
    print(result.summary())
    """)


if __name__ == "__main__":
    demo()
