"""
Configuration loader for web monitoring app
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AppConfig:
    """Application configuration"""

    # Tradier API
    tradier_api_key: str
    tradier_use_sandbox: bool

    # Model
    model_path: str

    # Monitoring
    polling_interval: int  # seconds
    cache_ttl: int  # seconds
    min_confidence: float
    min_strength: float
    max_strength: float

    # Alerts
    alert_cooldown: int  # seconds

    # Logging
    log_level: str
    log_file: str

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        # Default model path - resolves to absolute path
        default_model = str(Path(__file__).parent.parent / 'models/trained/QQQ_xgboost_optimized.pkl')

        return cls(
            tradier_api_key=os.getenv('TRADIER_API_KEY', ''),
            tradier_use_sandbox=os.getenv('TRADIER_USE_SANDBOX', 'false').lower() == 'true',
            model_path=os.getenv('MODEL_PATH', default_model),
            polling_interval=int(os.getenv('POLLING_INTERVAL', '60')),
            cache_ttl=int(os.getenv('CACHE_TTL', '300')),
            min_confidence=float(os.getenv('MIN_CONFIDENCE', '0.55')),
            min_strength=float(os.getenv('MIN_STRENGTH', '15')),
            max_strength=float(os.getenv('MAX_STRENGTH', '30')),
            alert_cooldown=int(os.getenv('ALERT_COOLDOWN', '300')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'logs/monitor.log')
        )

    def validate(self):
        """Validate configuration"""
        if not self.tradier_api_key:
            raise ValueError("TRADIER_API_KEY not set")

        model_path = Path(self.model_path)
        if not model_path.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Model not found: {self.model_path}")
            logger.warning("App will start but predictions will fail until model is uploaded")
            logger.warning("See RAILWAY_DEPLOYMENT.md for instructions on uploading models")

        if self.polling_interval < 10:
            raise ValueError("POLLING_INTERVAL must be >= 10 seconds")

        if not (0 < self.min_confidence < 1):
            raise ValueError("MIN_CONFIDENCE must be between 0 and 1")

# Global config instance
config = AppConfig.from_env()
