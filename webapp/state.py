"""
Shared application state manager
Thread-safe state for monitoring service and API routes
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class QuoteState:
    """Current quote state"""
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GreekLevels:
    """Greek level state"""
    hp_support: Optional[float] = None
    hp_resistance: Optional[float] = None
    mhp_support: Optional[float] = None
    mhp_resistance: Optional[float] = None
    hg_below: Optional[float] = None
    hg_above: Optional[float] = None
    gamma_flip: Optional[float] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StrengthData:
    """Strength score and components"""
    total: float
    components: Dict[str, float]
    level_type: str  # 'support', 'resistance', or 'none'
    level_source: str  # 'HP Support', 'MHP Support', etc.

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SignalAlert:
    """Trading signal alert"""
    timestamp: str
    symbol: str
    signal_type: str  # 'LONG'
    entry_price: float
    strength_score: float
    confidence: float
    level_type: str
    level_source: str
    greek_levels: Dict
    components: Dict
    suggested_stop: float
    suggested_target: float

    def to_dict(self) -> Dict:
        return asdict(self)


class AppState:
    """
    Application state manager
    Thread-safe access to shared state
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Current state
        self.quote: Optional[QuoteState] = None
        self.greeks: Optional[GreekLevels] = None
        self.strength: Optional[StrengthData] = None
        self.confidence: float = 0.0
        self.model_probability: float = 0.5

        # Monitoring status
        self.monitoring_active: bool = False
        self.last_update: Optional[datetime] = None
        self.last_alert_time: Optional[datetime] = None

        # Signal history (last 24 hours)
        self.signals: List[SignalAlert] = []
        self.max_signals = 100

        # SSE clients
        self.sse_clients: List[asyncio.Queue] = []

    async def update_quote(self, quote: QuoteState):
        """Update current quote"""
        async with self._lock:
            self.quote = quote
            self.last_update = datetime.now()
            await self._broadcast_event('quote_update', quote.to_dict())

    async def update_greeks(self, greeks: GreekLevels):
        """Update Greek levels"""
        async with self._lock:
            self.greeks = greeks
            await self._broadcast_event('greek_update', greeks.to_dict())

    async def update_strength(self, strength: StrengthData, confidence: float, model_prob: float):
        """Update strength and confidence"""
        async with self._lock:
            self.strength = strength
            self.confidence = confidence
            self.model_probability = model_prob
            await self._broadcast_event('strength_update', {
                'strength': strength.to_dict(),
                'confidence': confidence,
                'model_probability': model_prob
            })

    async def add_signal(self, signal: SignalAlert):
        """Add a new signal alert"""
        async with self._lock:
            self.signals.insert(0, signal)  # Most recent first

            # Trim to max size
            if len(self.signals) > self.max_signals:
                self.signals = self.signals[:self.max_signals]

            self.last_alert_time = datetime.now()
            await self._broadcast_event('signal_alert', signal.to_dict())

    async def get_current_state(self) -> Dict:
        """Get complete current state"""
        async with self._lock:
            return {
                'quote': self.quote.to_dict() if self.quote else None,
                'greeks': self.greeks.to_dict() if self.greeks else None,
                'strength': self.strength.to_dict() if self.strength else None,
                'confidence': self.confidence,
                'model_probability': self.model_probability,
                'monitoring_active': self.monitoring_active,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'recent_signals': [s.to_dict() for s in self.signals[:10]]
            }

    async def get_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals"""
        async with self._lock:
            return [s.to_dict() for s in self.signals[:limit]]

    async def set_monitoring_status(self, active: bool):
        """Set monitoring active status"""
        async with self._lock:
            self.monitoring_active = active
            await self._broadcast_event('monitoring_status', {'active': active})

    def can_emit_alert(self, cooldown_seconds: int) -> bool:
        """Check if enough time has passed since last alert"""
        if not self.last_alert_time:
            return True

        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed >= cooldown_seconds

    async def register_sse_client(self) -> asyncio.Queue:
        """Register a new SSE client"""
        client_queue = asyncio.Queue()
        async with self._lock:
            self.sse_clients.append(client_queue)
        return client_queue

    async def unregister_sse_client(self, client_queue: asyncio.Queue):
        """Unregister an SSE client"""
        async with self._lock:
            if client_queue in self.sse_clients:
                self.sse_clients.remove(client_queue)

    async def _broadcast_event(self, event_type: str, data: Dict):
        """Broadcast event to all SSE clients"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        # Send to all clients
        dead_clients = []
        for client in self.sse_clients:
            try:
                await client.put(event)
            except:
                dead_clients.append(client)

        # Clean up dead clients
        for dead in dead_clients:
            if dead in self.sse_clients:
                self.sse_clients.remove(dead)


# Global state instance
app_state = AppState()
