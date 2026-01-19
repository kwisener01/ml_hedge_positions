"""
Half Gap (HG) Calculation
Identifies significant gap fills and institutional reference levels
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import sys
sys.path.append('..')
from config.settings import institutional_config

class GapType(Enum):
    """Classification of price gaps"""
    COMMON = "common"           # Typically fills quickly
    BREAKAWAY = "breakaway"     # Start of trend, may not fill
    RUNAWAY = "runaway"         # Continuation, partial fills
    EXHAUSTION = "exhaustion"   # End of trend, likely fills

@dataclass
class GapLevel:
    """Single gap with metadata"""
    gap_date: datetime
    gap_type: GapType
    gap_high: float         # Top of gap
    gap_low: float          # Bottom of gap
    half_gap: float         # Midpoint
    gap_size: float         # Absolute size
    gap_size_pct: float     # Percentage size
    direction: str          # 'up' or 'down'
    filled: bool            # Whether gap has been filled
    fill_date: Optional[datetime]
    
    @property
    def is_unfilled(self) -> bool:
        return not self.filled

@dataclass
class HalfGapResult:
    """Complete HG analysis"""
    symbol: str
    spot_price: float
    timestamp: datetime
    gaps: List[GapLevel]
    active_half_gaps: List[float]   # Unfilled HG levels
    nearest_hg_above: Optional[float]
    nearest_hg_below: Optional[float]
    gap_magnet_zone: Optional[Tuple[float, float]]

class HalfGapCalculator:
    """
    Calculates Half Gap levels from price history
    
    Half Gap Theory:
    - Price gaps often act as magnets, pulling price back
    - The midpoint (half gap) is a key institutional reference
    - Unfilled gaps represent imbalance that market tends to correct
    
    Usage in trading:
    - HG below spot = potential support during pullbacks
    - HG above spot = potential resistance/target
    - Multiple unfilled HGs create "gap magnet zones"
    """
    
    def __init__(
        self,
        min_gap_pct: float = 0.002,     # 0.2% minimum gap
        lookback_days: int = 60,         # How far back to scan
        gap_fill_threshold: float = 0.9  # 90% fill = considered filled
    ):
        self.min_gap_pct = min_gap_pct
        self.lookback_days = lookback_days
        self.gap_fill_threshold = gap_fill_threshold
    
    def calculate(
        self,
        price_history: pd.DataFrame,
        spot_price: float,
        symbol: str = "UNKNOWN"
    ) -> HalfGapResult:
        """
        Calculate HG levels from price history
        
        Args:
            price_history: DataFrame with columns: open, high, low, close
                          Index should be datetime
            spot_price: Current price
            symbol: Ticker symbol
        """
        if price_history.empty or len(price_history) < 2:
            return self._empty_result(symbol, spot_price)
        
        # Detect all gaps
        gaps = self._detect_gaps(price_history)
        
        # Check fill status
        gaps = self._check_gap_fills(gaps, price_history, spot_price)
        
        # Get active (unfilled) half gaps
        active_hgs = [g.half_gap for g in gaps if g.is_unfilled]
        
        # Find nearest levels
        hg_above = self._find_nearest_above(active_hgs, spot_price)
        hg_below = self._find_nearest_below(active_hgs, spot_price)
        
        # Identify gap magnet zone (cluster of unfilled gaps)
        magnet_zone = self._find_magnet_zone(gaps, spot_price)
        
        return HalfGapResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            gaps=gaps,
            active_half_gaps=active_hgs,
            nearest_hg_above=hg_above,
            nearest_hg_below=hg_below,
            gap_magnet_zone=magnet_zone
        )
    
    def _detect_gaps(self, df: pd.DataFrame) -> List[GapLevel]:
        """Detect all price gaps in history"""
        gaps = []
        
        df = df.sort_index()
        
        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            prev_high = prev_row["high"]
            prev_low = prev_row["low"]
            prev_close = prev_row["close"]
            
            curr_open = curr_row["open"]
            curr_high = curr_row["high"]
            curr_low = curr_row["low"]
            
            # Gap up: current low > previous high
            if curr_low > prev_high:
                gap_size = curr_low - prev_high
                gap_size_pct = gap_size / prev_close
                
                if gap_size_pct >= self.min_gap_pct:
                    gap = GapLevel(
                        gap_date=df.index[i],
                        gap_type=self._classify_gap(df, i, "up"),
                        gap_high=curr_low,
                        gap_low=prev_high,
                        half_gap=(curr_low + prev_high) / 2,
                        gap_size=gap_size,
                        gap_size_pct=gap_size_pct,
                        direction="up",
                        filled=False,
                        fill_date=None
                    )
                    gaps.append(gap)
            
            # Gap down: current high < previous low
            elif curr_high < prev_low:
                gap_size = prev_low - curr_high
                gap_size_pct = gap_size / prev_close
                
                if gap_size_pct >= self.min_gap_pct:
                    gap = GapLevel(
                        gap_date=df.index[i],
                        gap_type=self._classify_gap(df, i, "down"),
                        gap_high=prev_low,
                        gap_low=curr_high,
                        half_gap=(prev_low + curr_high) / 2,
                        gap_size=gap_size,
                        gap_size_pct=gap_size_pct,
                        direction="down",
                        filled=False,
                        fill_date=None
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _classify_gap(
        self,
        df: pd.DataFrame,
        gap_index: int,
        direction: str
    ) -> GapType:
        """
        Classify gap type based on context
        
        - Common: Low volume, small gap, likely fills
        - Breakaway: High volume, breaks key level, may not fill
        - Runaway: Mid-trend continuation
        - Exhaustion: End of extended move
        """
        # Simplified classification based on volume and trend
        lookback = min(20, gap_index)
        
        if lookback < 5:
            return GapType.COMMON
        
        # Get volume context
        avg_volume = df.iloc[gap_index - lookback:gap_index]["volume"].mean()
        gap_volume = df.iloc[gap_index]["volume"]
        volume_ratio = gap_volume / avg_volume if avg_volume > 0 else 1
        
        # Get trend context
        prices = df.iloc[gap_index - lookback:gap_index]["close"]
        trend_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        # High volume gap at start of move = breakaway
        if volume_ratio > 1.5 and abs(trend_return) < 0.02:
            return GapType.BREAKAWAY
        
        # Gap after extended move = potential exhaustion
        if abs(trend_return) > 0.05:
            return GapType.EXHAUSTION
        
        # Gap in direction of trend = runaway
        if (direction == "up" and trend_return > 0) or \
           (direction == "down" and trend_return < 0):
            if volume_ratio > 1.0:
                return GapType.RUNAWAY
        
        return GapType.COMMON
    
    def _check_gap_fills(
        self,
        gaps: List[GapLevel],
        df: pd.DataFrame,
        spot_price: float
    ) -> List[GapLevel]:
        """Check which gaps have been filled"""
        
        for gap in gaps:
            # Get price action after gap
            gap_date = gap.gap_date
            subsequent_data = df[df.index > gap_date]
            
            if subsequent_data.empty:
                # Check current price
                if gap.direction == "up":
                    fill_level = gap.gap_low + (gap.gap_size * self.gap_fill_threshold)
                    if spot_price <= gap.half_gap:
                        gap.filled = True
                else:
                    fill_level = gap.gap_high - (gap.gap_size * self.gap_fill_threshold)
                    if spot_price >= gap.half_gap:
                        gap.filled = True
            else:
                # Check historical fills
                if gap.direction == "up":
                    # Gap up fills when price drops below half gap
                    min_low = subsequent_data["low"].min()
                    if min_low <= gap.half_gap:
                        gap.filled = True
                        fill_idx = subsequent_data[subsequent_data["low"] <= gap.half_gap].index[0]
                        gap.fill_date = fill_idx
                else:
                    # Gap down fills when price rises above half gap
                    max_high = subsequent_data["high"].max()
                    if max_high >= gap.half_gap:
                        gap.filled = True
                        fill_idx = subsequent_data[subsequent_data["high"] >= gap.half_gap].index[0]
                        gap.fill_date = fill_idx
        
        return gaps
    
    def _find_nearest_above(
        self,
        levels: List[float],
        spot: float
    ) -> Optional[float]:
        """Find nearest unfilled HG above spot"""
        above = [l for l in levels if l > spot]
        return min(above) if above else None
    
    def _find_nearest_below(
        self,
        levels: List[float],
        spot: float
    ) -> Optional[float]:
        """Find nearest unfilled HG below spot"""
        below = [l for l in levels if l < spot]
        return max(below) if below else None
    
    def _find_magnet_zone(
        self,
        gaps: List[GapLevel],
        spot_price: float
    ) -> Optional[Tuple[float, float]]:
        """
        Identify zone with multiple unfilled gaps (strong magnet)
        """
        unfilled = [g for g in gaps if g.is_unfilled]
        
        if len(unfilled) < 2:
            return None
        
        # Cluster unfilled gaps within 2% of each other
        half_gaps = sorted([g.half_gap for g in unfilled])
        
        # Find largest cluster
        best_cluster = []
        current_cluster = [half_gaps[0]]
        
        for i in range(1, len(half_gaps)):
            if (half_gaps[i] - current_cluster[-1]) / current_cluster[-1] < 0.02:
                current_cluster.append(half_gaps[i])
            else:
                if len(current_cluster) > len(best_cluster):
                    best_cluster = current_cluster
                current_cluster = [half_gaps[i]]
        
        if len(current_cluster) > len(best_cluster):
            best_cluster = current_cluster
        
        if len(best_cluster) >= 2:
            return (min(best_cluster), max(best_cluster))
        
        return None
    
    def _empty_result(self, symbol: str, spot_price: float) -> HalfGapResult:
        """Return empty result"""
        return HalfGapResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            gaps=[],
            active_half_gaps=[],
            nearest_hg_above=None,
            nearest_hg_below=None,
            gap_magnet_zone=None
        )