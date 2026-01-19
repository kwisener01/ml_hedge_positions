"""
Monthly Hedge Pressure (MHP) Calculation
Identifies significant institutional levels at monthly expirations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from config.settings import institutional_config
from institutional.hedge_pressure import (
    HedgePressureCalculator,
    HedgePressureLevel,
    HedgePressureResult
)

@dataclass
class MonthlyHPResult:
    """MHP analysis across multiple expirations"""
    symbol: str
    spot_price: float
    timestamp: datetime
    expiration_results: Dict[str, HedgePressureResult]
    aggregated_levels: List[HedgePressureLevel]
    mhp_score: float  # Weighted composite score
    primary_support: Optional[float]
    primary_resistance: Optional[float]
    gamma_flip_zone: Optional[Tuple[float, float]]  # Price range where dealers flip
    
    def get_nearest_support(self) -> Optional[float]:
        """Get nearest support below spot"""
        supports = [l.strike for l in self.aggregated_levels 
                   if l.is_support and l.strike < self.spot_price]
        return max(supports) if supports else None
    
    def get_nearest_resistance(self) -> Optional[float]:
        """Get nearest resistance above spot"""
        resistances = [l.strike for l in self.aggregated_levels
                      if l.is_resistance and l.strike > self.spot_price]
        return min(resistances) if resistances else None

class MonthlyHPCalculator:
    """
    Calculates Monthly Hedge Pressure across multiple expirations
    
    MHP weights nearer expirations more heavily and identifies
    strikes with consistent institutional positioning across time.
    
    Key concept: Strikes that show HP across multiple months
    represent stronger institutional conviction levels.
    """
    
    def __init__(self, config=None):
        self.config = config or institutional_config
        self.hp_calc = HedgePressureCalculator(config)
        
    def calculate(
        self,
        chains_by_expiration: Dict[str, pd.DataFrame],
        spot_price: float,
        symbol: str = "UNKNOWN"
    ) -> MonthlyHPResult:
        """
        Calculate MHP from multiple expiration chains
        
        Args:
            chains_by_expiration: Dict of expiration date -> options chain DataFrame
            spot_price: Current underlying price
            symbol: Ticker symbol
        """
        if not chains_by_expiration:
            return self._empty_result(symbol, spot_price)
        
        # Calculate HP for each expiration
        expiration_results: Dict[str, HedgePressureResult] = {}
        
        for exp_date, chain in chains_by_expiration.items():
            hp_result = self.hp_calc.calculate(chain, spot_price, symbol)
            expiration_results[exp_date] = hp_result
        
        # Weight and aggregate
        aggregated = self._aggregate_across_expirations(
            expiration_results, spot_price
        )
        
        # Calculate composite MHP score
        mhp_score = self._calculate_mhp_score(aggregated, spot_price)
        
        # Find primary levels (highest multi-expiration confirmation)
        primary_support, primary_resistance = self._find_primary_levels(
            aggregated, spot_price
        )
        
        # Identify gamma flip zone
        gamma_flip = self._find_gamma_flip_zone(aggregated, spot_price)
        
        return MonthlyHPResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            expiration_results=expiration_results,
            aggregated_levels=aggregated,
            mhp_score=mhp_score,
            primary_support=primary_support,
            primary_resistance=primary_resistance,
            gamma_flip_zone=gamma_flip
        )
    
    def _aggregate_across_expirations(
        self,
        results: Dict[str, HedgePressureResult],
        spot_price: float
    ) -> List[HedgePressureLevel]:
        """
        Aggregate HP levels across expirations with time-decay weighting
        """
        # Calculate weights based on days to expiration
        today = datetime.now().date()
        weights = {}
        
        for exp_str in results.keys():
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                # Exponential decay: nearer = higher weight
                weights[exp_str] = np.exp(-dte / 30)  # 30-day half-life
            except:
                weights[exp_str] = 0.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate by strike
        strike_aggregation: Dict[float, Dict] = {}
        
        for exp_str, hp_result in results.items():
            w = weights.get(exp_str, 0.5)
            
            for level in hp_result.levels:
                strike = level.strike
                
                if strike not in strike_aggregation:
                    strike_aggregation[strike] = {
                        "weighted_pressure": 0.0,
                        "total_oi": 0,
                        "total_vol": 0,
                        "net_delta": 0.0,
                        "directions": [],
                        "confirmations": 0
                    }
                
                agg = strike_aggregation[strike]
                agg["weighted_pressure"] += level.pressure_score * w
                agg["total_oi"] += level.open_interest
                agg["total_vol"] += level.volume
                agg["net_delta"] += level.net_delta * w
                agg["directions"].append(level.direction)
                agg["confirmations"] += 1
        
        # Build aggregated levels
        aggregated_levels = []
        max_pressure = max(
            (a["weighted_pressure"] for a in strike_aggregation.values()),
            default=1
        )
        
        for strike, agg in strike_aggregation.items():
            # Determine consensus direction
            direction_counts = {}
            for d in agg["directions"]:
                direction_counts[d] = direction_counts.get(d, 0) + 1
            
            consensus_direction = max(direction_counts, key=direction_counts.get)
            
            # Multi-expiration confirmation bonus
            confirmation_bonus = min(agg["confirmations"] / len(results), 1.0)
            
            level = HedgePressureLevel(
                strike=strike,
                pressure_score=agg["weighted_pressure"],
                direction=consensus_direction,
                open_interest=agg["total_oi"],
                volume=agg["total_vol"],
                net_delta=agg["net_delta"],
                confidence=(agg["weighted_pressure"] / max_pressure) * (0.7 + 0.3 * confirmation_bonus)
            )
            aggregated_levels.append(level)
        
        # Sort by weighted pressure
        aggregated_levels.sort(key=lambda x: x.pressure_score, reverse=True)
        
        return aggregated_levels
    
    def _calculate_mhp_score(
        self,
        levels: List[HedgePressureLevel],
        spot_price: float
    ) -> float:
        """
        Calculate composite MHP score
        
        Range: -1 (extremely bearish) to +1 (extremely bullish)
        """
        if not levels:
            return 0.0
        
        support_weight = 0.0
        resistance_weight = 0.0
        
        for level in levels:
            # Weight by proximity to spot
            distance = abs(level.strike - spot_price) / spot_price
            proximity_weight = np.exp(-distance * 10)  # Decay with distance
            
            weighted_score = level.pressure_score * level.confidence * proximity_weight
            
            if level.is_support and level.strike < spot_price:
                support_weight += weighted_score
            elif level.is_resistance and level.strike > spot_price:
                resistance_weight += weighted_score
        
        total = support_weight + resistance_weight
        if total == 0:
            return 0.0
        
        return (support_weight - resistance_weight) / total
    
    def _find_primary_levels(
        self,
        levels: List[HedgePressureLevel],
        spot_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find strongest support and resistance with multi-expiration confirmation"""
        
        # Filter to high-confidence levels
        strong_levels = [l for l in levels if l.confidence > 0.5]
        
        supports = [l for l in strong_levels 
                   if l.is_support and l.strike < spot_price]
        resistances = [l for l in strong_levels
                      if l.is_resistance and l.strike > spot_price]
        
        primary_support = None
        if supports:
            # Nearest strong support
            supports.sort(key=lambda x: x.strike, reverse=True)
            primary_support = supports[0].strike
        
        primary_resistance = None
        if resistances:
            # Nearest strong resistance
            resistances.sort(key=lambda x: x.strike)
            primary_resistance = resistances[0].strike
        
        return primary_support, primary_resistance
    
    def _find_gamma_flip_zone(
        self,
        levels: List[HedgePressureLevel],
        spot_price: float
    ) -> Optional[Tuple[float, float]]:
        """
        Identify the gamma flip zone where dealer positioning changes
        
        This is where the market transitions from support to resistance
        """
        if len(levels) < 3:
            return None
        
        # Find strikes near spot with balanced pressure
        near_spot = [l for l in levels 
                    if abs(l.strike - spot_price) / spot_price < 0.02]  # Within 2%
        
        if len(near_spot) < 2:
            return None
        
        # Look for direction changes
        strikes = sorted([l.strike for l in near_spot])
        
        if len(strikes) >= 2:
            return (strikes[0], strikes[-1])
        
        return None
    
    def _empty_result(self, symbol: str, spot_price: float) -> MonthlyHPResult:
        """Return empty result when no data"""
        return MonthlyHPResult(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=datetime.now(),
            expiration_results={},
            aggregated_levels=[],
            mhp_score=0.0,
            primary_support=None,
            primary_resistance=None,
            gamma_flip_zone=None
        )