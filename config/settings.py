"""
Configuration management for SVM Trading System
Tradier API credentials and system parameters
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TradierConfig:
    """Tradier API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("TRADIER_API_KEY", ""))
    base_url: str = "https://api.tradier.com/v1"
    sandbox_url: str = "https://sandbox.tradier.com/v1"
    use_sandbox: bool = False
    
    @property
    def active_url(self) -> str:
        return self.sandbox_url if self.use_sandbox else self.base_url
    
    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

@dataclass
class ModelConfig:
    """SVM ensemble configuration"""
    tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    window_size: int = 5                    # k=5 events per window
    polynomial_degree: int = 3              # d=3 kernel (optimized from experiments)
    constraint_param: float = 0.1           # C=0.1 (optimized from experiments)
    ensemble_count: int = 100               # 100 independent SVMs
    alpha_threshold: float = 1e-5           # Movement threshold
    spread_max_pct: float = 0.25            # 25% max spread filter
    
@dataclass
class InstitutionalConfig:
    """Hedge pressure calculation parameters"""
    hp_lookback_days: int = 5               # Daily HP lookback
    mhp_expiration_range: int = 30          # Monthly expiration window
    oi_weight: float = 0.6                  # Open interest weight
    volume_weight: float = 0.4              # Volume weight
    strike_range_pct: float = 0.10          # +/- 10% from spot for analysis

# Global instances
tradier_config = TradierConfig()
model_config = ModelConfig()
institutional_config = InstitutionalConfig()