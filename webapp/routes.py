"""
API Routes
REST endpoints and SSE streaming
"""
import asyncio
import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict

from state import app_state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/status")
async def get_status() -> Dict:
    """
    Health check endpoint

    Returns:
        status: ok/error
        monitoring_active: bool
        last_update: timestamp
    """
    try:
        state = await app_state.get_current_state()
        return {
            "status": "ok",
            "monitoring_active": state['monitoring_active'],
            "last_update": state['last_update']
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/api/current")
async def get_current() -> Dict:
    """
    Get complete current state

    Returns:
        quote: Current QQQ quote
        greeks: Greek levels
        strength: Strength data
        confidence: Bayesian confidence
        model_probability: Model probability
    """
    try:
        state = await app_state.get_current_state()
        return {
            "status": "ok",
            "data": state
        }
    except Exception as e:
        logger.error(f"Current state error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/api/strength")
async def get_strength() -> Dict:
    """
    Get strength score breakdown

    Returns:
        total: Total strength (0-100)
        components: Component breakdown
        level_type: support/resistance/none
        level_source: Which level (HP, MHP, HG)
    """
    try:
        state = await app_state.get_current_state()
        return {
            "status": "ok",
            "strength": state.get('strength')
        }
    except Exception as e:
        logger.error(f"Strength error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/api/signals")
async def get_signals(limit: int = 50) -> Dict:
    """
    Get recent signals

    Args:
        limit: Maximum number of signals to return

    Returns:
        signals: List of recent signals
    """
    try:
        signals = await app_state.get_signals(limit)
        return {
            "status": "ok",
            "count": len(signals),
            "signals": signals
        }
    except Exception as e:
        logger.error(f"Signals error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/api/events")
async def event_stream(request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time updates

    Event types:
    - quote_update: New quote data
    - greek_update: Updated Greek levels
    - strength_update: Updated strength/confidence
    - signal_alert: New trading signal
    - monitoring_status: Monitoring active/inactive
    """
    async def generate():
        # Register this client
        client_queue = await app_state.register_sse_client()

        try:
            # Send initial state
            state = await app_state.get_current_state()
            yield f"event: initial_state\n"
            yield f"data: {json.dumps(state)}\n\n"

            # Stream events
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        client_queue.get(),
                        timeout=30.0
                    )

                    # Send event
                    event_type = event['type']
                    event_data = event['data']

                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event_data)}\n\n"

                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f": heartbeat\n\n"

        except asyncio.CancelledError:
            logger.info("SSE client cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
        finally:
            # Unregister client
            await app_state.unregister_sse_client(client_queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Control endpoints (optional, for future use)

@router.post("/api/control/start")
async def start_monitoring() -> Dict:
    """Start monitoring (if stopped)"""
    # This would require managing the background task
    # For now, monitoring starts automatically on app startup
    return {
        "status": "ok",
        "message": "Monitoring starts automatically on app startup"
    }


@router.post("/api/control/stop")
async def stop_monitoring() -> Dict:
    """Stop monitoring"""
    # This would require managing the background task
    # For now, monitoring runs continuously
    return {
        "status": "ok",
        "message": "Monitoring runs continuously (restart app to stop)"
    }
