"""
Within-Window Features (V1-V10)
Recovers information lost during data thinning
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

import sys
sys.path.append('..')
from data.tradier_client import QuoteData

@dataclass
class WindowFeatures:
    """V1-V10 feature set for a single window"""
    v1_open_mid: float          # Opening mid-price
    v2_close_mid: float         # Closing mid-price
    v3_crossing_return: float   # Return from open to close
    v4_high_mid: float          # Highest mid-price
    v5_low_mid: float           # Lowest mid-price
    v6_range: float             # High - Low
    v7_avg_spread: float        # Average bid-ask spread
    v8_spread_volatility: float # Std dev of spreads
    v9_mid_volatility: float    # Std dev of mid-prices
    v10_intensity: float        # Events per unit time (arrival rate)
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.v1_open_mid,
            self.v2_close_mid,
            self.v3_crossing_return,
            self.v4_high_mid,
            self.v5_low_mid,
            self.v6_range,
            self.v7_avg_spread,
            self.v8_spread_volatility,
            self.v9_mid_volatility,
            self.v10_intensity
        ])
    
    def to_dict(self) -> Dict:
        return {
            "V1_open_mid": self.v1_open_mid,
            "V2_close_mid": self.v2_close_mid,
            "V3_crossing_return": self.v3_crossing_return,
            "V4_high_mid": self.v4_high_mid,
            "V5_low_mid": self.v5_low_mid,
            "V6_range": self.v6_range,
            "V7_avg_spread": self.v7_avg_spread,
            "V8_spread_volatility": self.v8_spread_volatility,
            "V9_mid_volatility": self.v9_mid_volatility,
            "V10_intensity": self.v10_intensity
        }

class WindowFeatureExtractor:
    """
    Extracts V1-V10 features from event windows
    
    These features capture within-window dynamics that would be
    lost if using simple OHLC aggregation.
    """
    
    def extract(self, window: List[QuoteData]) -> WindowFeatures:
        """Extract all within-window features from a single window"""
        
        if not window:
            return self._empty_features()
        
        # Extract raw values
        mid_prices = [q.mid_price for q in window]
        spreads = [q.spread for q in window]
        timestamps = [q.timestamp for q in window]
        
        # V1: Opening mid-price
        v1 = mid_prices[0]
        
        # V2: Closing mid-price
        v2 = mid_prices[-1]
        
        # V3: Crossing return (within-window return)
        v3 = (v2 - v1) / v1 if v1 != 0 else 0
        
        # V4: Highest mid-price
        v4 = max(mid_prices)
        
        # V5: Lowest mid-price
        v5 = min(mid_prices)
        
        # V6: Range
        v6 = v4 - v5
        
        # V7: Average spread
        v7 = np.mean(spreads)
        
        # V8: Spread volatility
        v8 = np.std(spreads) if len(spreads) > 1 else 0
        
        # V9: Mid-price volatility
        v9 = np.std(mid_prices) if len(mid_prices) > 1 else 0
        
        # V10: Intensity (events per second)
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            v10 = len(window) / time_span if time_span > 0 else 0
        else:
            v10 = 0
        
        return WindowFeatures(
            v1_open_mid=v1,
            v2_close_mid=v2,
            v3_crossing_return=v3,
            v4_high_mid=v4,
            v5_low_mid=v5,
            v6_range=v6,
            v7_avg_spread=v7,
            v8_spread_volatility=v8,
            v9_mid_volatility=v9,
            v10_intensity=v10
        )
    
    def extract_batch(self, windows: List[List[QuoteData]]) -> pd.DataFrame:
        """Extract features from multiple windows"""
        features = [self.extract(w).to_dict() for w in windows]
        return pd.DataFrame(features)
    
    def _empty_features(self) -> WindowFeatures:
        return WindowFeatures(
            v1_open_mid=0, v2_close_mid=0, v3_crossing_return=0,
            v4_high_mid=0, v5_low_mid=0, v6_range=0,
            v7_avg_spread=0, v8_spread_volatility=0,
            v9_mid_volatility=0, v10_intensity=0
        )