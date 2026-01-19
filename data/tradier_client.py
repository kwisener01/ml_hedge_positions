"""
Tradier API Client
Handles all data fetching with rate limiting and error handling
"""
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
from functools import wraps

import sys
sys.path.append('..')
from config.settings import tradier_config, model_config, TradierConfig

def rate_limit(calls_per_minute: int = 60):
    """Decorator for API rate limiting"""
    min_interval = 60.0 / calls_per_minute
    last_call = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class QuoteData:
    """Structured quote response"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    volume: int
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price == 0:
            return float('inf')
        return self.spread / self.mid_price

class TradierClient:
    """
    Tradier API client with built-in caching and rate limiting
    """
    
    def __init__(self, config: Optional[TradierConfig] = None):
        self.config = config or tradier_config
        self._options_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)
        
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Base request method with error handling"""
        url = f"{self.config.active_url}{endpoint}"
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.config.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self._request(endpoint, params)
            raise Exception(f"HTTP Error: {e}")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    @rate_limit(calls_per_minute=120)
    def get_quote(self, symbol: str) -> Optional[QuoteData]:
        """Fetch current quote for a symbol"""
        data = self._request("/markets/quotes", {"symbols": symbol})
        
        if "quotes" not in data or "quote" not in data["quotes"]:
            return None
            
        q = data["quotes"]["quote"]
        
        return QuoteData(
            symbol=q.get("symbol", symbol),
            bid=float(q.get("bid", 0)),
            ask=float(q.get("ask", 0)),
            bid_size=int(q.get("bidsize", 0)),
            ask_size=int(q.get("asksize", 0)),
            last=float(q.get("last", 0)),
            volume=int(q.get("volume", 0)),
            timestamp=datetime.now()
        )
    
    @rate_limit(calls_per_minute=60)
    def get_options_chain(
        self, 
        symbol: str, 
        expiration: Optional[str] = None,
        greeks: bool = True
    ) -> pd.DataFrame:
        """
        Fetch options chain with optional Greeks
        
        Args:
            symbol: Underlying ticker
            expiration: YYYY-MM-DD format, None for nearest
            greeks: Include Greeks data
        """
        cache_key = f"{symbol}_{expiration}_{greeks}"
        
        # Check cache
        if cache_key in self._options_cache:
            if datetime.now() - self._cache_timestamp[cache_key] < self._cache_ttl:
                return self._options_cache[cache_key]
        
        # Get expiration if not specified
        if expiration is None:
            expirations = self.get_expirations(symbol)
            if not expirations:
                return pd.DataFrame()
            expiration = expirations[0]
        
        params = {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": str(greeks).lower()
        }
        
        data = self._request("/markets/options/chains", params)
        
        if "options" not in data or data["options"] is None:
            return pd.DataFrame()
            
        options = data["options"].get("option", [])
        
        if not options:
            return pd.DataFrame()
        
        df = pd.DataFrame(options)
        
        # Flatten Greeks if present
        if greeks and "greeks" in df.columns:
            greeks_df = pd.json_normalize(df["greeks"])
            greeks_df.columns = [f"greek_{col}" for col in greeks_df.columns]
            df = pd.concat([df.drop("greeks", axis=1), greeks_df], axis=1)
        
        # Cache result
        self._options_cache[cache_key] = df
        self._cache_timestamp[cache_key] = datetime.now()
        
        return df
    
    @rate_limit(calls_per_minute=120)
    def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates for options"""
        data = self._request("/markets/options/expirations", {"symbol": symbol})
        
        if "expirations" not in data or data["expirations"] is None:
            return []
            
        return data["expirations"].get("date", [])
    
    @rate_limit(calls_per_minute=60)
    def get_history(
        self, 
        symbol: str, 
        interval: str = "daily",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            symbol: Ticker symbol
            interval: daily, weekly, monthly
            start: YYYY-MM-DD
            end: YYYY-MM-DD
        """
        params = {"symbol": symbol, "interval": interval}
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        data = self._request("/markets/history", params)
        
        if "history" not in data or data["history"] is None:
            return pd.DataFrame()
            
        days = data["history"].get("day", [])
        
        if not days:
            return pd.DataFrame()
            
        df = pd.DataFrame(days)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        return df
    
    def get_multiple_chains(
        self,
        symbol: str,
        expiration_count: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple expiration chains for MHP calculation
        
        Args:
            symbol: Underlying ticker
            expiration_count: Number of expirations to fetch
        """
        expirations = self.get_expirations(symbol)[:expiration_count]
        chains = {}
        
        for exp in expirations:
            chains[exp] = self.get_options_chain(symbol, exp)
            
        return chains
    
    def stream_quotes(
        self,
        symbols: List[str],
        callback: callable,
        duration_seconds: int = 300
    ):
        """
        Simulated quote streaming via polling
        Real streaming requires websocket upgrade
        
        Args:
            symbols: List of tickers
            callback: Function to call with each quote batch
            duration_seconds: How long to stream
        """
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            quotes = []
            for symbol in symbols:
                quote = self.get_quote(symbol)
                if quote:
                    quotes.append(quote)
            
            if quotes:
                callback(quotes)
            
            time.sleep(1)  # 1 second polling interval