"""
Background Monitoring Service
Polls QQQ every 60 seconds and checks for tradeable setups
"""
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tradier_client import TradierClient, QuoteData
from features.feature_matrix import FeatureMatrixBuilder
from webapp_config import AppConfig
from state import app_state, QuoteState, GreekLevels, StrengthData, SignalAlert
from strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)


class MonitorService:
    """
    Background monitoring service for QQQ
    Runs in asyncio background task
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False

        # Initialize clients
        self.tradier_client = TradierClient()
        self.feature_builder = FeatureMatrixBuilder(self.tradier_client)

        # Initialize strategy engine
        self.strategy = StrategyEngine(config.model_path)

        logger.info("Monitor service initialized")

    async def start(self):
        """Start monitoring loop"""
        if self.running:
            logger.warning("Monitor already running")
            return

        self.running = True
        await app_state.set_monitoring_status(True)
        logger.info("Starting monitor loop")

        try:
            await self.monitor_loop()
        except Exception as e:
            logger.error(f"Monitor loop crashed: {e}", exc_info=True)
        finally:
            self.running = False
            await app_state.set_monitoring_status(False)

    async def stop(self):
        """Stop monitoring loop"""
        self.running = False
        await app_state.set_monitoring_status(False)
        logger.info("Monitor stopped")

    async def monitor_loop(self):
        """
        Main monitoring loop
        Polls every POLLING_INTERVAL seconds
        """
        while self.running:
            try:
                await self._monitor_iteration()
            except Exception as e:
                logger.error(f"Monitor iteration error: {e}", exc_info=True)

            # Wait for next iteration
            await asyncio.sleep(self.config.polling_interval)

    async def _monitor_iteration(self):
        """
        Single monitoring iteration
        1. Fetch quote
        2. Build features (uses cached institutional data)
        3. Calculate strength
        4. Run model
        5. Check entry conditions
        6. Update state
        7. Emit alerts if conditions met
        """
        symbol = 'QQQ'

        # 1. Fetch quote (async wrapper for sync call)
        quote = await asyncio.to_thread(
            self.tradier_client.get_quote,
            symbol
        )

        if not quote:
            logger.warning("Failed to get quote")
            return

        # Update quote state
        quote_state = QuoteState(
            symbol=quote.symbol,
            price=quote.mid_price,
            bid=quote.bid,
            ask=quote.ask,
            spread=quote.spread,
            timestamp=quote.timestamp.isoformat()
        )
        await app_state.update_quote(quote_state)

        logger.debug(f"Quote: {quote.symbol} @ ${quote.mid_price:.2f}")

        # 2. Build feature vector (uses 5-min cached institutional data)
        try:
            feature_array, feature_dict = await asyncio.to_thread(
                self.feature_builder.build_feature_vector,
                symbol,
                [quote],  # Simplified window for real-time
                quote
            )
        except Exception as e:
            logger.error(f"Feature building error: {e}", exc_info=True)
            return

        # Extract Greek levels from features
        greeks = self._extract_greek_levels(feature_dict)
        await app_state.update_greeks(greeks)

        # 3. Calculate strength score
        strength_score, components = self.strategy.calculate_strength_score(feature_dict)
        level_type = self.strategy.identify_level_type(feature_dict)
        level_source = self.strategy.identify_level_source(feature_dict)

        strength_data = StrengthData(
            total=strength_score,
            components=components,
            level_type=level_type,
            level_source=level_source
        )

        # 4. Run model prediction
        proba = await asyncio.to_thread(
            self.strategy.model.predict_proba,
            feature_array.reshape(1, -1)
        )
        prob_up = proba[0][1]

        # 5. Calculate Bayesian confidence
        confidence = self.strategy.bayesian_scorer.calculate_confidence(
            model_probability=prob_up,
            strength_score=strength_score,
            level_type=level_type
        )

        # Update strength and confidence state
        await app_state.update_strength(strength_data, confidence, prob_up)

        logger.debug(
            f"Strength: {strength_score:.1f} | "
            f"Level: {level_type} | "
            f"Confidence: {confidence:.3f} | "
            f"Model P(UP): {prob_up:.3f}"
        )

        # 6. Check entry conditions
        should_enter, conf, strength, ltype = self.strategy.check_entry_conditions(
            feature_array,
            feature_dict,
            self.config.min_strength,
            self.config.max_strength,
            self.config.min_confidence
        )

        # 7. Emit alert if conditions met
        if should_enter:
            # Check alert cooldown
            if not app_state.can_emit_alert(self.config.alert_cooldown):
                logger.info("Alert cooldown active, skipping")
                return

            logger.info(
                f"SIGNAL DETECTED! "
                f"Entry: ${quote.mid_price:.2f} | "
                f"Strength: {strength:.1f} | "
                f"Confidence: {conf:.3f}"
            )

            # Create alert
            alert = SignalAlert(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                signal_type='LONG',
                entry_price=quote.mid_price,
                strength_score=strength,
                confidence=conf,
                level_type=ltype,
                level_source=level_source,
                greek_levels=greeks.to_dict(),
                components=components,
                suggested_stop=quote.mid_price * 0.99,  # -1%
                suggested_target=quote.mid_price * 1.015  # +1.5%
            )

            await app_state.add_signal(alert)

    def _extract_greek_levels(self, feature_dict: dict) -> GreekLevels:
        """
        Extract Greek level prices from feature dictionary

        Note: Features contain distances, not absolute prices
        Need to calculate actual price levels
        """
        # This is a simplified extraction
        # In production, would need to store actual level prices
        # For now, return None for levels that aren't directly available

        return GreekLevels(
            hp_support=None,
            hp_resistance=None,
            mhp_support=None,
            mhp_resistance=None,
            hg_below=None,
            hg_above=None,
            gamma_flip=None
        )


async def run_monitor(config: AppConfig):
    """
    Run monitor service
    Called by FastAPI on startup
    """
    service = MonitorService(config)
    await service.start()
