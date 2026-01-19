"""
Resilience and Arbitrage Indicators (LOB-based)
Implements V3 Crossing Return and Order Flow Resilience
Based on Limit Order Book microstructure analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tradier_client import TradierClient, QuoteData


@dataclass
class ResilienceMetrics:
    """
    Order flow resilience indicators

    Measures how quickly market corrects imbalances
    High resilience = strong internal buying support
    """
    order_flow_resilience: float   # Speed of order replacement (0-1)
    bid_ask_resilience: float      # How quickly spread tightens after trades
    volume_resilience: float       # Volume consistency (stability metric)
    price_impact_decay: float      # How fast price impact fades (0-1)
    immediacy_ratio: float         # Fill rate / Average Daily Volume proxy


@dataclass
class ArbitrageIndicators:
    """
    Arbitrage opportunity indicators (LOB-based)

    V3 Crossing Return: Profit from buying at first ask, selling at last bid
    """
    v3_crossing_return: float      # (Last_Bid - First_Ask) / First_Ask
    bid_ask_spread_pct: float      # Current bid-ask spread %
    spread_z_score: float          # Z-score of spread (tight/wide)
    effective_spread: float        # Effective spread after trades
    microstructure_edge: float     # Combined arb signal (normalized)


class ResilienceCalculator:
    """
    Calculate order flow resilience metrics

    Resilience = speed at which new orders correct market imbalances
    High resilience = strong institutional support (immediacy + resiliency)
    """

    def calculate_order_flow_resilience(self, window: List[QuoteData]) -> float:
        """
        Order flow resilience: how quickly spread recovers after price moves

        Measures the number of quote updates (new orders) after mid-price changes
        High = many orders, market quickly corrects
        Low = few orders, market stays imbalanced
        """
        if len(window) < 10:
            return 0.5

        mid_prices = np.array([q.mid_price for q in window])
        spreads = np.array([q.ask - q.bid for q in window])

        # Find mid-price changes (trades implied)
        price_changes = np.abs(np.diff(mid_prices))
        threshold = np.std(price_changes) if np.std(price_changes) > 0 else 0.01

        recovery_speeds = []

        for i in range(len(price_changes) - 5):
            if price_changes[i] > threshold:  # Significant price move
                # Measure spread recovery over next 5 events
                initial_spread = spreads[i]
                if initial_spread == 0:
                    continue

                # How quickly does spread tighten back?
                future_spreads = spreads[i+1:i+6]
                if len(future_spreads) > 0:
                    avg_future_spread = np.mean(future_spreads)
                    recovery = 1 - (avg_future_spread / initial_spread)
                    recovery_speeds.append(max(0, recovery))

        if not recovery_speeds:
            return 0.5

        return np.mean(recovery_speeds)

    def calculate_bid_ask_resilience(self, window: List[QuoteData]) -> float:
        """
        Bid-ask resilience: spread stability after volatility

        Low = spread widens and stays wide (illiquid)
        High = spread quickly returns to tight (liquid, resilient)
        """
        if len(window) < 10:
            return 0.5

        spreads = np.array([q.ask - q.bid for q in window])
        mid_prices = np.array([q.mid_price for q in window])

        # Calculate spread as % of mid
        spread_pct = spreads / mid_prices * 100

        # Find volatility spikes (large price moves)
        returns = np.abs(np.diff(mid_prices) / mid_prices[:-1])
        vol_threshold = np.percentile(returns, 75)

        spread_recovery = []

        for i in range(len(returns) - 5):
            if returns[i] > vol_threshold:  # Volatility spike
                spike_spread = spread_pct[i]
                if spike_spread == 0:
                    continue

                # Measure spread tightening over next 5 bars
                future_spreads = spread_pct[i+1:i+6]
                if len(future_spreads) > 0:
                    avg_future = np.mean(future_spreads)
                    tightening = (spike_spread - avg_future) / spike_spread
                    spread_recovery.append(max(0, tightening))

        if not spread_recovery:
            return 0.5

        return np.mean(spread_recovery)

    def calculate_volume_resilience(self, window: List[QuoteData]) -> float:
        """
        Volume resilience: consistency of order flow

        Measures volume stability (low variance = resilient order flow)
        """
        if len(window) < 5:
            return 0.5

        volumes = np.array([q.volume for q in window])

        # Remove zeros
        volumes = volumes[volumes > 0]
        if len(volumes) < 3:
            return 0.5

        # Coefficient of variation (inverse = resilience)
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        if mean_vol == 0:
            return 0.5

        cv = std_vol / mean_vol
        # Lower CV = higher resilience
        # Normalize: CV of 0.5 = resilience 0.5
        resilience = 1 / (1 + cv)

        return resilience

    def calculate_price_impact_decay(self, window: List[QuoteData]) -> float:
        """
        Price impact decay: how quickly price reverts after large moves

        High decay = moves are temporary (resilient, mean-reverting)
        Low decay = moves persist (trending, low resilience)
        """
        if len(window) < 10:
            return 0.5

        mid_prices = np.array([q.mid_price for q in window])
        returns = np.diff(mid_prices) / mid_prices[:-1]

        # Find large moves (>1 std)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.5

        decay_rates = []

        for i in range(len(returns) - 5):
            if abs(returns[i]) > std_ret:  # Large move
                initial_move = returns[i]
                # Measure reversion over next 5 bars
                next_5 = returns[i+1:i+6]
                if len(next_5) > 0:
                    # If move was up, expect negative returns (reversion)
                    # If move was down, expect positive returns
                    expected_reversion = -np.sign(initial_move) * np.mean(next_5)
                    decay = max(0, expected_reversion / abs(initial_move))
                    decay_rates.append(min(decay, 1.0))

        if not decay_rates:
            return 0.5

        return np.mean(decay_rates)

    def calculate_immediacy_ratio(self, window: List[QuoteData]) -> float:
        """
        Immediacy ratio: fill rate relative to volume

        High = many quotes per volume (tight market, high immediacy)
        Low = few quotes per volume (wide market, low immediacy)
        """
        if len(window) < 5:
            return 0.5

        total_volume = sum([q.volume for q in window if q.volume > 0])
        num_quotes = len(window)

        if total_volume == 0:
            return 0.5

        # Quotes per million volume
        immediacy = (num_quotes / total_volume) * 1_000_000

        # Normalize to 0-1 (assume 1000 quotes/million vol = perfect)
        normalized = immediacy / 1000
        return min(normalized, 1.0)

    def calculate(self, window: List[QuoteData]) -> ResilienceMetrics:
        """Calculate all resilience metrics from quote window"""

        order_flow_res = self.calculate_order_flow_resilience(window)
        bid_ask_res = self.calculate_bid_ask_resilience(window)
        volume_res = self.calculate_volume_resilience(window)
        impact_decay = self.calculate_price_impact_decay(window)
        immediacy = self.calculate_immediacy_ratio(window)

        return ResilienceMetrics(
            order_flow_resilience=order_flow_res,
            bid_ask_resilience=bid_ask_res,
            volume_resilience=volume_res,
            price_impact_decay=impact_decay,
            immediacy_ratio=immediacy
        )


class ArbitrageCalculator:
    """
    Calculate LOB-based arbitrage indicators

    V3 Crossing Return: Profit from buying at first ask, selling at last bid
    Based on Strategy I: Recovering Information in Data Thinning
    """

    def __init__(self, k: int = 5):
        """
        Args:
            k: Event window size for V3 calculation (default 5)
        """
        self.k = k

    def calculate_v3_crossing_return(self, window: List[QuoteData]) -> float:
        """
        V3 Crossing Return: (Last_Best_Bid - First_Best_Ask) / First_Best_Ask

        Detects arbitrage profit from buying at window start (ask)
        and selling at window end (bid)

        Positive V3 = profitable arb opportunity
        Negative V3 = no arb (would lose money)
        """
        if len(window) < self.k:
            return 0.0

        # Use last k events
        window_k = window[-self.k:]

        first_ask = window_k[0].ask
        last_bid = window_k[-1].bid

        if first_ask == 0 or last_bid == 0:
            return 0.0

        v3 = (last_bid - first_ask) / first_ask

        return v3

    def calculate_bid_ask_spread_pct(self, quote: QuoteData) -> float:
        """
        Current bid-ask spread as percentage

        Lower = tighter market, less arb friction
        Higher = wider market, more arb potential but harder to execute
        """
        if quote.mid_price == 0:
            return 0.0

        spread = quote.ask - quote.bid
        spread_pct = (spread / quote.mid_price) * 100

        return spread_pct

    def calculate_spread_z_score(self, window: List[QuoteData]) -> float:
        """
        Z-score of current spread relative to recent history

        Positive = spread wider than normal (opportunity or risk)
        Negative = spread tighter than normal (efficient market)
        """
        if len(window) < 10:
            return 0.0

        current_spread = window[-1].ask - window[-1].bid
        current_mid = window[-1].mid_price

        if current_mid == 0:
            return 0.0

        current_spread_pct = current_spread / current_mid

        # Historical spreads
        spreads_pct = []
        for q in window[:-1]:
            if q.mid_price > 0:
                spread_pct = (q.ask - q.bid) / q.mid_price
                spreads_pct.append(spread_pct)

        if len(spreads_pct) < 5:
            return 0.0

        mean_spread = np.mean(spreads_pct)
        std_spread = np.std(spreads_pct)

        if std_spread == 0:
            return 0.0

        z_score = (current_spread_pct - mean_spread) / std_spread

        return z_score

    def calculate_effective_spread(self, window: List[QuoteData]) -> float:
        """
        Effective spread: actual cost of trading

        Measures difference between mid-price and trade prices
        Lower = less slippage, more efficient execution
        """
        if len(window) < 2:
            return 0.0

        effective_spreads = []

        for i in range(1, len(window)):
            prev_mid = window[i-1].mid_price
            curr_last = window[i].last
            curr_mid = window[i].mid_price

            if prev_mid > 0 and curr_last > 0 and curr_mid > 0:
                # Effective spread = 2 * |trade_price - mid_price| / mid_price
                eff_spread = 2 * abs(curr_last - curr_mid) / curr_mid
                effective_spreads.append(eff_spread)

        if not effective_spreads:
            return 0.0

        return np.mean(effective_spreads)

    def calculate_microstructure_edge(
        self,
        v3: float,
        spread_pct: float,
        spread_z: float,
        eff_spread: float
    ) -> float:
        """
        Combined microstructure edge signal

        Normalized score combining:
        - V3 crossing return (arb potential)
        - Spread tightness (execution quality)
        - Spread z-score (anomaly detection)
        - Effective spread (slippage)

        Range: -1 to 1
        Positive = favorable microstructure for trading
        Negative = unfavorable microstructure
        """
        # Normalize V3 (typical range -0.005 to 0.005)
        v3_norm = np.clip(v3 / 0.005, -1, 1)

        # Normalize spread_pct (typical 0-0.5%)
        # Lower is better, so invert
        spread_norm = -np.clip(spread_pct / 0.5, 0, 1)

        # Normalize spread_z (typical -3 to 3)
        # Negative is better (tight spread)
        z_norm = -np.clip(spread_z / 3, -1, 1)

        # Normalize effective spread (typical 0-1%)
        # Lower is better
        eff_norm = -np.clip(eff_spread / 0.01, 0, 1)

        # Weighted combination
        # V3 is most important (60%), others support (40%)
        edge = (
            0.60 * v3_norm +
            0.15 * spread_norm +
            0.15 * z_norm +
            0.10 * eff_norm
        )

        return edge

    def calculate(self, window: List[QuoteData]) -> ArbitrageIndicators:
        """Calculate all arbitrage indicators from quote window"""

        v3 = self.calculate_v3_crossing_return(window)
        spread_pct = self.calculate_bid_ask_spread_pct(window[-1])
        spread_z = self.calculate_spread_z_score(window)
        eff_spread = self.calculate_effective_spread(window)

        micro_edge = self.calculate_microstructure_edge(
            v3, spread_pct, spread_z, eff_spread
        )

        return ArbitrageIndicators(
            v3_crossing_return=v3,
            bid_ask_spread_pct=spread_pct,
            spread_z_score=spread_z,
            effective_spread=eff_spread,
            microstructure_edge=micro_edge
        )


if __name__ == "__main__":
    """Test resilience and arb calculators with QQQ"""

    client = TradierClient()
    res_calc = ResilienceCalculator()
    arb_calc = ArbitrageCalculator(k=5)

    print("\n" + "="*80)
    print("LOB RESILIENCE & ARBITRAGE INDICATORS: QQQ")
    print("="*80)

    try:
        # Get recent quotes
        quote = client.get_quote("QQQ")
        print(f"\nCurrent QQQ Price: ${quote.last:.2f}")
        print(f"Bid: ${quote.bid:.2f} | Ask: ${quote.ask:.2f} | Spread: ${quote.ask - quote.bid:.2f}")

        # Simulate a window of quotes (in production, this comes from live feed)
        # For testing, create synthetic window from current quote
        window = [quote] * 10  # Placeholder

        print("\n" + "-"*80)
        print("RESILIENCE INDICATORS (Order Flow)")
        print("-"*80)

        res = res_calc.calculate(window)

        print(f"\nOrder Flow Resilience:  {res.order_flow_resilience:.3f}")
        print(f"  {'[HIGH - Quick order replacement]' if res.order_flow_resilience > 0.7 else '[LOW - Slow order flow]'}")

        print(f"\nBid-Ask Resilience:     {res.bid_ask_resilience:.3f}")
        print(f"  {'[TIGHT - Spread recovers quickly]' if res.bid_ask_resilience > 0.7 else '[WIDE - Spread stays elevated]'}")

        print(f"\nVolume Resilience:      {res.volume_resilience:.3f}")
        print(f"  {'[STABLE - Consistent flow]' if res.volume_resilience > 0.6 else '[ERRATIC - Inconsistent flow]'}")

        print(f"\nPrice Impact Decay:     {res.price_impact_decay:.3f}")
        print(f"  {'[MEAN-REVERTING]' if res.price_impact_decay > 0.6 else '[TRENDING]'}")

        print(f"\nImmediacy Ratio:        {res.immediacy_ratio:.3f}")
        print(f"  {'[HIGH LIQUIDITY]' if res.immediacy_ratio > 0.5 else '[LOW LIQUIDITY]'}")

        print("\n" + "-"*80)
        print("ARBITRAGE INDICATORS (V3 Crossing Return)")
        print("-"*80)

        arb = arb_calc.calculate(window)

        print(f"\nV3 Crossing Return:     {arb.v3_crossing_return:+.4%}")
        print(f"  {'[ARB OPPORTUNITY]' if arb.v3_crossing_return > 1e-5 else '[NO ARB]' if arb.v3_crossing_return > -1e-5 else '[NEGATIVE ARB]'}")
        print(f"  (Buy @ First Ask, Sell @ Last Bid over k={arb_calc.k} events)")

        print(f"\nBid-Ask Spread:         {arb.bid_ask_spread_pct:.3f}%")
        print(f"  {'[TIGHT]' if arb.bid_ask_spread_pct < 0.1 else '[NORMAL]' if arb.bid_ask_spread_pct < 0.3 else '[WIDE]'}")

        print(f"\nSpread Z-Score:         {arb.spread_z_score:+.2f}")
        print(f"  {'[ANOMALY - Abnormally wide]' if arb.spread_z_score > 2 else '[ANOMALY - Abnormally tight]' if arb.spread_z_score < -2 else '[NORMAL]'}")

        print(f"\nEffective Spread:       {arb.effective_spread:.4%}")
        print(f"  (Average execution slippage)")

        print(f"\nMicrostructure Edge:    {arb.microstructure_edge:+.3f}")
        print(f"  {'[FAVORABLE FOR TRADING]' if arb.microstructure_edge > 0.2 else '[UNFAVORABLE]' if arb.microstructure_edge < -0.2 else '[NEUTRAL]'}")

        print("\n" + "="*80)
        print("INTEGRATION NOTES")
        print("="*80)
        print("\n• V3 Crossing Return uses k=5 event windows (as specified)")
        print("• Alpha threshold: 1e-5 for significance testing")
        print("• Resilience indicators measure market internal strength")
        print("• Combine with HP/MHP/HG levels for 'Golden Setups'")
        print("• Filter with LIXI (Liquidity Index) to avoid thin markets")
        print("\n" + "="*80)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
