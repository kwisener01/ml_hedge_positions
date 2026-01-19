"""
Data Cleaning Module
Implements the clean-up protocol from specification
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.append('..')
from config.settings import model_config
from data.tradier_client import QuoteData

@dataclass
class CleaningStats:
    """Track data quality metrics"""
    total_records: int = 0
    negative_price_removed: int = 0
    zero_quantity_removed: int = 0
    wide_spread_removed: int = 0
    valid_records: int = 0
    
    @property
    def rejection_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return 1 - (self.valid_records / self.total_records)

class DataCleaner:
    """
    Cleans raw market data per specification:
    - Remove negative prices
    - Remove zero quantities
    - Remove spreads > 25% of midpoint
    """
    
    def __init__(self, spread_max_pct: float = None):
        self.spread_max_pct = spread_max_pct or model_config.spread_max_pct
        self.stats = CleaningStats()
        
    def clean_quote(self, quote: QuoteData) -> Tuple[bool, Optional[str]]:
        """
        Validate a single quote
        
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check negative prices
        if quote.bid < 0 or quote.ask < 0 or quote.last < 0:
            return False, "negative_price"
        
        # Check zero quantities
        if quote.bid_size <= 0 or quote.ask_size <= 0:
            return False, "zero_quantity"
        
        # Check spread threshold
        if quote.spread_pct > self.spread_max_pct:
            return False, "wide_spread"
        
        # Check crossed market (ask < bid)
        if quote.ask < quote.bid:
            return False, "crossed_market"
            
        return True, None
    
    def clean_quotes(self, quotes: List[QuoteData]) -> List[QuoteData]:
        """Clean a batch of quotes, tracking statistics"""
        self.stats = CleaningStats(total_records=len(quotes))
        cleaned = []
        
        for quote in quotes:
            is_valid, reason = self.clean_quote(quote)
            
            if is_valid:
                cleaned.append(quote)
            else:
                if reason == "negative_price":
                    self.stats.negative_price_removed += 1
                elif reason == "zero_quantity":
                    self.stats.zero_quantity_removed += 1
                elif reason == "wide_spread":
                    self.stats.wide_spread_removed += 1
        
        self.stats.valid_records = len(cleaned)
        return cleaned
    
    def clean_options_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean options chain DataFrame
        
        Removes:
        - Zero open interest
        - Zero volume (optional, configurable)
        - Invalid Greeks
        - Extreme bid-ask spreads
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove zero/negative prices
        df = df[df["bid"] >= 0]
        df = df[df["ask"] > 0]
        
        # Remove crossed markets
        df = df[df["ask"] >= df["bid"]]
        
        # Calculate spread percentage
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread_pct"] = np.where(
            df["mid"] > 0,
            (df["ask"] - df["bid"]) / df["mid"],
            float('inf')
        )
        
        # Filter wide spreads
        df = df[df["spread_pct"] <= self.spread_max_pct]
        
        # Remove zero open interest (no institutional interest)
        if "open_interest" in df.columns:
            df = df[df["open_interest"] > 0]
        
        # Validate Greeks if present
        greek_cols = [c for c in df.columns if c.startswith("greek_")]
        for col in greek_cols:
            df = df[df[col].notna()]
            df = df[~np.isinf(df[col])]
        
        print(f"Options chain cleaned: {original_len} -> {len(df)} records")
        
        return df.reset_index(drop=True)


class EventWindowBuilder:
    """
    Groups events into k-sized windows for feature calculation
    Implements data thinning per specification
    """
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or model_config.window_size
        self.buffer: List[QuoteData] = []
        self.windows: List[List[QuoteData]] = []
        
    def add_event(self, quote: QuoteData) -> Optional[List[QuoteData]]:
        """
        Add event to buffer, return completed window if ready
        """
        self.buffer.append(quote)
        
        if len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size]
            self.buffer = self.buffer[self.window_size:]
            self.windows.append(window)
            return window
            
        return None
    
    def get_windows_dataframe(self) -> pd.DataFrame:
        """
        Convert windows to DataFrame with aggregated metrics
        """
        if not self.windows:
            return pd.DataFrame()
        
        records = []
        
        for i, window in enumerate(self.windows):
            mid_prices = [q.mid_price for q in window]
            spreads = [q.spread for q in window]
            volumes = [q.volume for q in window]
            
            records.append({
                "window_id": i,
                "start_time": window[0].timestamp,
                "end_time": window[-1].timestamp,
                "open_mid": mid_prices[0],
                "close_mid": mid_prices[-1],
                "high_mid": max(mid_prices),
                "low_mid": min(mid_prices),
                "avg_mid": np.mean(mid_prices),
                "avg_spread": np.mean(spreads),
                "total_volume": sum(volumes),
                "event_count": len(window),
                "price_change": mid_prices[-1] - mid_prices[0],
                "price_change_pct": (mid_prices[-1] - mid_prices[0]) / mid_prices[0] if mid_prices[0] != 0 else 0
            })
        
        return pd.DataFrame(records)
    
    def reset(self):
        """Clear all buffers and windows"""
        self.buffer = []
        self.windows = []