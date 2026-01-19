"""
QQQ Support Trading Monitor - FastAPI Application
Real-time web dashboard for monitoring tradeable setups
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from webapp_config import config
from routes import router
from monitor_service import MonitorService
from retraining_scheduler import RetrainingScheduler

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global services
monitor_service = None
monitor_task = None
retraining_scheduler = None
scheduler_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Starts/stops background monitor and retraining scheduler
    """
    global monitor_service, monitor_task, retraining_scheduler, scheduler_task

    # Startup
    logger.info("Starting QQQ Monitor Application")

    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Initialize monitor service
    monitor_service = MonitorService(config)

    # Start monitor in background task
    monitor_task = asyncio.create_task(monitor_service.start())
    logger.info("Background monitor started")

    # Initialize retraining scheduler
    retraining_scheduler = RetrainingScheduler(
        symbol='QQQ',
        retrain_hour=2,  # 2 AM daily
        min_samples=100
    )

    # Start scheduler in background task
    scheduler_task = asyncio.create_task(retraining_scheduler.start())
    logger.info("Retraining scheduler started")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop monitor
    if monitor_service:
        await monitor_service.stop()

    # Stop scheduler
    if retraining_scheduler:
        await retraining_scheduler.stop()

    # Cancel background tasks
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="QQQ Support Trading Monitor",
    description="Real-time monitoring dashboard for QQQ support trading strategy",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve main dashboard"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    """Health check endpoint (for Docker)"""
    return {
        "status": "healthy",
        "monitoring_active": monitor_service.running if monitor_service else False
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level=config.log_level.lower()
    )
