"""
Live Signal Monitor
Continuously monitors markets and generates trading signals
"""
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import tradier_config
from signals.signal_generator import SignalGenerator, TradingSignal, SignalType


class SignalMonitor:
    """
    Continuously monitor markets for trading signals

    Features:
    - Multi-symbol monitoring
    - Configurable scan intervals
    - Signal logging
    - Alert system
    """

    def __init__(
        self,
        symbols: List[str] = None,
        scan_interval_minutes: int = 15,
        min_svm_confidence: float = 0.6,
        min_confluence: float = 0.5,
        log_file: str = "signals/signal_log.json"
    ):
        self.symbols = symbols or ["SPY", "QQQ"]
        self.scan_interval = scan_interval_minutes * 60  # Convert to seconds
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize generators
        self.generators: Dict[str, SignalGenerator] = {}

        for symbol in self.symbols:
            try:
                self.generators[symbol] = SignalGenerator(
                    symbol=symbol,
                    min_svm_confidence=min_svm_confidence,
                    min_confluence=min_confluence
                )
            except FileNotFoundError as e:
                print(f"[WARNING] Skipping {symbol}: {e}")

        self.signal_history: List[TradingSignal] = []
        self.last_scan_time: Optional[datetime] = None

        print(f"\n{'='*80}")
        print(f"SIGNAL MONITOR INITIALIZED")
        print(f"{'='*80}")
        print(f"Symbols: {list(self.generators.keys())}")
        print(f"Scan Interval: {scan_interval_minutes} minutes")
        print(f"Min SVM Confidence: {min_svm_confidence*100:.0f}%")
        print(f"Min Confluence: {min_confluence*100:.0f}%")
        print(f"Log File: {self.log_file}")
        print(f"{'='*80}\n")

    def scan_markets(self) -> List[TradingSignal]:
        """
        Scan all symbols for signals

        Returns:
            List of generated signals
        """
        print(f"\n{'='*80}")
        print(f"MARKET SCAN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        signals = []

        for symbol, generator in self.generators.items():
            try:
                print(f"\nScanning {symbol}...")
                signal = generator.generate_signal()

                if signal:
                    signals.append(signal)
                    self.signal_history.append(signal)
                    self._log_signal(signal)
                    print(f"[SIGNAL] {symbol}: {signal.signal_type.name} ({signal.signal_strength.name})")
                else:
                    print(f"[NO SIGNAL] {symbol}")

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        self.last_scan_time = datetime.now()

        print(f"\n{'='*80}")
        print(f"Scan complete. Signals generated: {len(signals)}")
        print(f"{'='*80}\n")

        return signals

    def _log_signal(self, signal: TradingSignal):
        """Log signal to JSON file"""
        log_entry = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.name,
            'signal_strength': signal.signal_strength.name,
            'current_price': float(signal.current_price),
            'entry_price': float(signal.entry_price),
            'stop_loss': float(signal.stop_loss),
            'take_profit': float(signal.take_profit),
            'svm_prediction': int(signal.svm_prediction),
            'svm_confidence': float(signal.svm_confidence),
            'confluence_score': float(signal.confluence_score),
            'hp_score': float(signal.hp_score),
            'mhp_score': float(signal.mhp_score),
            'reasoning': signal.reasoning
        }

        # Append to log file
        logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(log_entry)

        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_active_signals(self, hours: int = 24) -> List[TradingSignal]:
        """
        Get signals generated in last N hours

        Args:
            hours: Look back period

        Returns:
            List of recent signals
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self.signal_history if s.timestamp >= cutoff]

    def run_continuous(self, duration_hours: Optional[int] = None):
        """
        Run continuous monitoring

        Args:
            duration_hours: How long to run (None = run forever)
        """
        print(f"\n{'='*80}")
        print(f"STARTING CONTINUOUS MONITORING")
        print(f"{'='*80}")
        print(f"Duration: {'Indefinite' if duration_hours is None else f'{duration_hours} hours'}")
        print(f"Scan Interval: {self.scan_interval/60:.0f} minutes")
        print(f"Press Ctrl+C to stop\n")

        start_time = datetime.now()

        try:
            while True:
                # Run scan
                signals = self.scan_markets()

                # Display active signals
                if signals:
                    for signal in signals:
                        print(signal)

                # Check duration
                if duration_hours:
                    elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed_hours >= duration_hours:
                        print(f"\nDuration limit reached ({duration_hours} hours)")
                        break

                # Wait for next scan
                next_scan = datetime.now() + timedelta(seconds=self.scan_interval)
                print(f"\nNext scan at: {next_scan.strftime('%H:%M:%S')}")
                print(f"Waiting {self.scan_interval/60:.0f} minutes...")

                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Monitoring stopped by user")

        # Final summary
        self._print_summary()

    def run_single_scan(self):
        """Run a single market scan"""
        signals = self.scan_markets()

        if signals:
            for signal in signals:
                print(signal)
        else:
            print("\nNo trading signals generated.\n")

        return signals

    def _print_summary(self):
        """Print session summary"""
        print(f"\n{'='*80}")
        print(f"SESSION SUMMARY")
        print(f"{'='*80}")

        if self.signal_history:
            print(f"Total Signals Generated: {len(self.signal_history)}\n")

            # By symbol
            symbol_counts = {}
            for signal in self.signal_history:
                symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1

            print("Signals by Symbol:")
            for symbol, count in symbol_counts.items():
                print(f"  {symbol}: {count}")

            # By type
            long_signals = len([s for s in self.signal_history if s.signal_type == SignalType.LONG])
            short_signals = len([s for s in self.signal_history if s.signal_type == SignalType.SHORT])

            print(f"\nSignals by Type:")
            print(f"  LONG: {long_signals}")
            print(f"  SHORT: {short_signals}")

            # By strength
            strong = len([s for s in self.signal_history if s.signal_strength.name == "STRONG"])
            medium = len([s for s in self.signal_history if s.signal_strength.name == "MEDIUM"])
            weak = len([s for s in self.signal_history if s.signal_strength.name == "WEAK"])

            print(f"\nSignals by Strength:")
            print(f"  STRONG: {strong}")
            print(f"  MEDIUM: {medium}")
            print(f"  WEAK: {weak}")

        else:
            print("No signals generated during session.")

        print(f"\nLog file: {self.log_file}")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Live Trading Signal Monitor")
    parser.add_argument(
        '--mode',
        choices=['single', 'continuous'],
        default='single',
        help='Run mode: single scan or continuous monitoring'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in hours for continuous mode (default: run forever)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=15,
        help='Scan interval in minutes for continuous mode (default: 15)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.55,
        help='Minimum SVM confidence (0-1, default: 0.55)'
    )
    parser.add_argument(
        '--confluence',
        type=float,
        default=0.4,
        help='Minimum confluence score (0-1, default: 0.4)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SPY', 'QQQ'],
        help='Symbols to monitor (default: SPY QQQ)'
    )

    args = parser.parse_args()

    # Check API key
    if not tradier_config.api_key:
        print("[ERROR] TRADIER_API_KEY not set")
        return

    # Initialize monitor
    monitor = SignalMonitor(
        symbols=args.symbols,
        scan_interval_minutes=args.interval,
        min_svm_confidence=args.confidence,
        min_confluence=args.confluence
    )

    # Run
    if args.mode == 'single':
        monitor.run_single_scan()
    else:
        monitor.run_continuous(duration_hours=args.duration)


if __name__ == "__main__":
    main()
