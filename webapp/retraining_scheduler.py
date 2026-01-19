"""
Retraining Scheduler Service
Runs periodic model retraining in the background
"""
import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime, time
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_logger import TrainingDataLogger

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Schedules and executes model retraining at specified intervals
    """

    def __init__(
        self,
        symbol: str = "QQQ",
        retrain_hour: int = 2,  # 2 AM by default
        min_samples: int = 100,
        check_interval: int = 3600  # Check every hour
    ):
        """
        Initialize retraining scheduler

        Args:
            symbol: Trading symbol
            retrain_hour: Hour of day (0-23) to run retraining
            min_samples: Minimum samples needed for retraining
            check_interval: Seconds between readiness checks
        """
        self.symbol = symbol
        self.retrain_hour = retrain_hour
        self.min_samples = min_samples
        self.check_interval = check_interval
        self.running = False
        self.last_retrain = None

        self.data_logger = TrainingDataLogger()
        self.retrain_script = Path(__file__).parent.parent / "training/retrain_from_live_data.py"

        logger.info(f"Retraining scheduler initialized (daily at {retrain_hour}:00)")

    async def start(self):
        """Start the retraining scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        logger.info("Retraining scheduler started")

        try:
            await self.scheduler_loop()
        except Exception as e:
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
        finally:
            self.running = False

    async def stop(self):
        """Stop the retraining scheduler"""
        self.running = False
        logger.info("Retraining scheduler stopped")

    async def scheduler_loop(self):
        """
        Main scheduler loop
        Checks periodically if it's time to retrain
        """
        while self.running:
            try:
                await self._check_and_retrain()
            except Exception as e:
                logger.error(f"Retraining check error: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _check_and_retrain(self):
        """Check if retraining should run and execute if needed"""
        current_hour = datetime.now().hour

        # Check if it's the right hour for retraining
        if current_hour != self.retrain_hour:
            return

        # Check if already retrained today
        if self.last_retrain and self.last_retrain.date() == datetime.now().date():
            logger.debug("Already retrained today, skipping")
            return

        # Check if enough data collected
        stats = self.data_logger.get_training_data_stats()
        logger.info(f"Training data stats: {stats}")

        if stats['predictions_count'] < self.min_samples:
            logger.info(
                f"Not enough data for retraining: "
                f"{stats['predictions_count']}/{self.min_samples}"
            )
            return

        # Execute retraining
        logger.info("Starting model retraining...")
        success = await self._run_retraining()

        if success:
            self.last_retrain = datetime.now()
            logger.info(f"Retraining completed successfully at {self.last_retrain}")
        else:
            logger.warning("Retraining failed or was skipped")

    async def _run_retraining(self) -> bool:
        """
        Execute the retraining script

        Returns:
            True if successful, False otherwise
        """
        try:
            # Run retraining script as subprocess
            cmd = [
                sys.executable,
                str(self.retrain_script),
                '--symbol', self.symbol,
                '--min-samples', str(self.min_samples),
                '--model-type', 'xgboost'
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            # Run async
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Log output
            if stdout:
                logger.info(f"Retraining output: {stdout.decode()}")
            if stderr:
                logger.warning(f"Retraining errors: {stderr.decode()}")

            # Check exit code
            if process.returncode == 0:
                logger.info("Retraining subprocess completed successfully")
                return True
            else:
                logger.warning(f"Retraining subprocess exited with code {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"Failed to run retraining: {e}", exc_info=True)
            return False

    def force_retrain(self) -> bool:
        """
        Force immediate retraining (synchronous)
        Useful for manual triggering

        Returns:
            True if successful
        """
        logger.info("Force retraining triggered")

        try:
            cmd = [
                sys.executable,
                str(self.retrain_script),
                '--symbol', self.symbol,
                '--min-samples', str(self.min_samples),
                '--model-type', 'xgboost'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)

            if result.returncode == 0:
                self.last_retrain = datetime.now()
                logger.info("Force retraining completed successfully")
                return True
            else:
                logger.warning(f"Force retraining failed with code {result.returncode}")
                return False

        except Exception as e:
            logger.error(f"Force retraining error: {e}", exc_info=True)
            return False


# Global scheduler instance
retraining_scheduler: Optional[RetrainingScheduler] = None


def get_scheduler() -> RetrainingScheduler:
    """Get global scheduler instance"""
    global retraining_scheduler
    if retraining_scheduler is None:
        retraining_scheduler = RetrainingScheduler()
    return retraining_scheduler
