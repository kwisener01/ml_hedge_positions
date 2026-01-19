"""
Hedge Pressure (HP) Calculation
Identifies institutional positioning from options flow
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.append('..')
from config.settings import institutional_config

@dataclass
class HedgePressureLevel:
    """Single HP level with metadata"""
    strike: float
    pressure_score: float
    direction: str  # 'call_wall', 'put_wall', 'neutral'
    open_interest: int
    volume: int
    net_delta: float
    confidence: float  # 0-1 strength indicator
    
    @property
    def is_support(self) -> bool:
        return self.direction == 'put_wall'
    
    @property
    def is_resistance(self) -> bool:
        return self.direction == 'call_wall'

@dataclass  
class HedgePressureResult:
    """Complete HP analysis result"""
    symbol: str
    spot_price: float
    timestamp: datetime
    levels: List[HedgePressureLevel]
    net_hp: float  # Positive = bullish positioning, Negative = bearish
    dominant_direction: str
    key_support: Optional[float]
    key_resistance: Optional[float]
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "timestamp": self.timestamp.isoformat(),
            "net_hp": self.net_hp,
            "dominant_direction": self.dominant_direction,
            "key_support": self.key_support,
            "key_resistance": self.key_resistance,
            "level_count": len(self.levels)
        }

class HedgePressureCalculator:
    """
    Calculates Hedge Pressure from options chain data
    
    HP identifies strikes where market makers have significant 
    gamma exposure, creating natural support/resistance levels.
    
    Formula:
    HP(K) = OI_weight * OI(K) + Vol_weight * Vol(K)
    
    Directional adjustment:
    - Call OI at strike K -> Resistance (dealers short calls = sell rallies)
    - Put OI at strike K -> Support (dealers short puts = buy dips)
    """
    
    def __init__(self, config=None):
        self.config = config or institutional_config
        self.oi_weight = self.config.oi_weight
        self.vol_weight = self.config.volume_weight
        
    def calculate(
        self,
        options_chain: pd.DataFrame,
        spot_price: float,
        symbol: str = "UNKNOWN"
    ) -> HedgePressureResult:
        """
        Calculate HP levels from options chain
        
        Args:
            options_chain: DataFrame with strike, type, open_interest, volume
            spot_price: Current underlying price
            symbol: Ticker symbol
        """
        if options_chain.empty:
            return self._empty_result(symbol, spot_price)
        
        # Filter to relevant strike range
        min_strike = spot_price * (1 - self.config.strike_range_pct)
        max_strike = spot_price * (1 + self.config.strike_range_pct)
        
        df = options_chain[
            (options_chain["strike"] >= min_strike) &
            (options_chain["strike"] <= max_strike)
        ].copy()
        
        if df.empty:
            return self._empty_result(symbol, spot_price)
        
        # Separate calls and puts
        calls = df[df["option_type"] == "call"].copy()
        puts = df[df["option_type"] == "put"].copy()
        
        # Calculate raw pressure at each strike
        strike_pressure = self._aggregate_by_strike(calls, puts)
        
        # Build HP levels
        levels = self._build_levels(strike_pressure, spot_price)
        
        # Calculate net HP
        net_hp = self._calculate_net_hp(levels, spot_price)
        
        # Identify key levels
        key_support, key_resistance = self._find_key_levels(levels, spot_price)
        
        # Determine dominant direction
        dominant = "bullish" if net_hp > 0 else "bearish" if net_hp < 0 else "neutral"
        
        return HedgePressureResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            levels=levels,
            net_hp=net_hp,
            dominant_direction=dominant,
            key_support=key_support,
            key_resistance=key_resistance
        )
    
    def _aggregate_by_strike(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate OI and volume by strike"""
        
        # Call pressure (resistance)
        call_agg = calls.groupby("strike").agg({
            "open_interest": "sum",
            "volume": "sum"
        }).rename(columns={
            "open_interest": "call_oi",
            "volume": "call_vol"
        })
        
        # Put pressure (support)
        put_agg = puts.groupby("strike").agg({
            "open_interest": "sum",
            "volume": "sum"
        }).rename(columns={
            "open_interest": "put_oi",
            "volume": "put_vol"
        })
        
        # Merge
        combined = call_agg.join(put_agg, how="outer").fillna(0)
        
        # Calculate pressure scores
        combined["call_pressure"] = (
            self.oi_weight * combined["call_oi"] +
            self.vol_weight * combined["call_vol"]
        )
        combined["put_pressure"] = (
            self.oi_weight * combined["put_oi"] +
            self.vol_weight * combined["put_vol"]
        )
        combined["net_pressure"] = combined["put_pressure"] - combined["call_pressure"]
        combined["total_pressure"] = combined["call_pressure"] + combined["put_pressure"]
        
        return combined.reset_index()
    
    def _build_levels(
        self,
        strike_pressure: pd.DataFrame,
        spot_price: float
    ) -> List[HedgePressureLevel]:
        """Convert pressure data to HP levels"""
        
        if strike_pressure.empty:
            return []
        
        # Normalize pressure for confidence calculation
        max_pressure = strike_pressure["total_pressure"].max()
        if max_pressure == 0:
            max_pressure = 1
        
        levels = []
        
        for _, row in strike_pressure.iterrows():
            # Determine direction
            if row["call_pressure"] > row["put_pressure"] * 1.5:
                direction = "call_wall"
            elif row["put_pressure"] > row["call_pressure"] * 1.5:
                direction = "put_wall"
            else:
                direction = "neutral"
            
            # Calculate delta proxy (simplified)
            distance_pct = (row["strike"] - spot_price) / spot_price
            net_delta = -row["call_oi"] * max(0, 0.5 - distance_pct) + \
                        row["put_oi"] * max(0, 0.5 + distance_pct)
            
            level = HedgePressureLevel(
                strike=row["strike"],
                pressure_score=row["total_pressure"],
                direction=direction,
                open_interest=int(row["call_oi"] + row["put_oi"]),
                volume=int(row["call_vol"] + row["put_vol"]),
                net_delta=net_delta,
                confidence=row["total_pressure"] / max_pressure
            )
            levels.append(level)
        
        # Sort by pressure score descending
        levels.sort(key=lambda x: x.pressure_score, reverse=True)
        
        return levels
    
    def _calculate_net_hp(
        self,
        levels: List[HedgePressureLevel],
        spot_price: float
    ) -> float:
        """
        Calculate net hedge pressure
        
        Positive = More put support below = bullish
        Negative = More call resistance above = bearish
        """
        if not levels:
            return 0.0
        
        bullish_pressure = 0.0
        bearish_pressure = 0.0
        
        for level in levels:
            if level.strike < spot_price and level.is_support:
                bullish_pressure += level.pressure_score * level.confidence
            elif level.strike > spot_price and level.is_resistance:
                bearish_pressure += level.pressure_score * level.confidence
        
        total = bullish_pressure + bearish_pressure
        if total == 0:
            return 0.0
            
        return (bullish_pressure - bearish_pressure) / total
    
    def _find_key_levels(
        self,
        levels: List[HedgePressureLevel],
        spot_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find nearest significant support and resistance"""
        
        supports = [l for l in levels if l.is_support and l.strike < spot_price]
        resistances = [l for l in levels if l.is_resistance and l.strike > spot_price]
        
        # Get highest confidence below spot
        key_support = None
        if supports:
            best_support = max(supports, key=lambda x: x.confidence)
            if best_support.confidence > 0.3:  # Minimum threshold
                key_support = best_support.strike
        
        # Get highest confidence above spot
        key_resistance = None
        if resistances:
            best_resistance = max(resistances, key=lambda x: x.confidence)
            if best_resistance.confidence > 0.3:
                key_resistance = best_resistance.strike
        
        return key_support, key_resistance
    
    def _empty_result(self, symbol: str, spot_price: float) -> HedgePressureResult:
        """Return empty result when no data available"""
        return HedgePressureResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            levels=[],
            net_hp=0.0,
            dominant_direction="neutral",
            key_support=None,
            key_resistance=None
        )