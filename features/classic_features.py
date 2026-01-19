"""
Standard/Classic Features (V11-V22)
Captures immediate state of the limit order book
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass

import sys
sys.path.append('..')
from data.tradier_client import QuoteData
from config.settings import model_config

@dataclass
class ClassicFeatures:
    """V11-V22 feature set"""
    v11_bid: float              # Current bid
    v12_ask: float              # Current ask
    v13_mid_price: float        # Current mid-price
    v14_spread: float           # Current spread
    v15_spread_pct: float       # Spread as % of mid
    v16_spread_return: float    # Change in spread
    v17_bid_size: int           # Bid depth
    v18_ask_size: int           # Ask depth
    v19_imbalance: float        # Order book imbalance
    v20_volume: int             # Recent volume
    v21_vwap: float             # Volume-weighted avg price
    v22_arrival_rate: float     # Quote arrival rate
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.v11_bid, self.v12_ask, self.v13_mid_price,
            self.v14_spread, self.v15_spread_pct, self.v16_spread_return,
            self.v17_bid_size, self.v18_ask_size, self.v19_imbalance,
            self.v20_volume, self.v21_vwap, self.v22_arrival_rate
        ])
    
    def to_dict(self) -> Dict:
        return {
            "V11_bid": self.v11_bid,
            "V12_ask": self.v12_ask,
            "V13_mid_price": self.v13_mid_price,
            "V14_spread": self.v14_spread,
            "V15_spread_pct": self.v15_spread_pct,
            "V16_spread_return": self.v16_spread_return,
            "V17_bid_size": self.v17_bid_size,
            "V18_ask_size": self.v18_ask_size,
            "V19_imbalance": self.v19_imbalance,
            "V20_volume": self.v20_volume,
            "V21_vwap": self.v21_vwap,
            "V22_arrival_rate": self.v22_arrival_rate
        }

class ClassicFeatureExtractor:
    """
    Extracts V11-V22 features from current market state
    
    These capture the immediate LOB conditions at window end.
    """
    
    def __init__(self):
        self.previous_spread: Optional[float] = None
        self.quote_times: List[float] = []
        
    def extract(
        self,
        current_quote: QuoteData,
        window: List[QuoteData]
    ) -> ClassicFeatures:
        """
        Extract classic features from current quote and recent window
        
        Args:
            current_quote: Most recent quote
            window: Recent quotes for context
        """
        q = current_quote
        
        # V11-V13: Basic prices
        v11 = q.bid
        v12 = q.ask
        v13 = q.mid_price
        
        # V14-V15: Spread metrics
        v14 = q.spread
        v15 = q.spread_pct
        
        # V16: Spread return
        if self.previous_spread is not None and self.previous_spread > 0:
            v16 = (v14 - self.previous_spread) / self.previous_spread
        else:
            v16 = 0
        self.previous_spread = v14
        
        # V17-V18: Depth
        v17 = q.bid_size
        v18 = q.ask_size
        
        # V19: Order imbalance
        total_size = v17 + v18
        v19 = (v17 - v18) / total_size if total_size > 0 else 0
        
        # V20: Volume
        v20 = q.volume
        
        # V21: VWAP approximation from window
        if window:
            prices = [w.last for w in window if w.last > 0]
            volumes = [w.volume for w in window if w.volume > 0]
            
            if volumes and sum(volumes) > 0:
                v21 = np.average(prices, weights=volumes)
            else:
                v21 = v13
        else:
            v21 = v13
        
        # V22: Arrival rate
        self.quote_times.append(q.timestamp.timestamp())
        
        # Keep last 100 timestamps
        if len(self.quote_times) > 100:
            self.quote_times = self.quote_times[-100:]
        
        if len(self.quote_times) > 1:
            time_diffs = np.diff(self.quote_times)
            avg_interval = np.mean(time_diffs)
            v22 = 1 / avg_interval if avg_interval > 0 else 0
        else:
            v22 = 0
        
        return ClassicFeatures(
            v11_bid=v11, v12_ask=v12, v13_mid_price=v13,
            v14_spread=v14, v15_spread_pct=v15, v16_spread_return=v16,
            v17_bid_size=v17, v18_ask_size=v18, v19_imbalance=v19,
            v20_volume=v20, v21_vwap=v21, v22_arrival_rate=v22
        )
    
    def reset(self):
        """Reset state between sessions"""
        self.previous_spread = None
        self.quote_times = []