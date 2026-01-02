"""Shared types to avoid circular dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pal_router.complexity import ComplexitySignals


class Lane(str, Enum):
    """The three routing lanes."""

    FAST = "FAST"
    REASONING = "REASONING"
    AGENTIC = "AGENTIC"


@dataclass
class RoutingDecision:
    """Complete routing decision with metadata."""

    lane: Lane
    complexity_score: float
    signals: ComplexitySignals
    reason: str
    confidence: float = 1.0
    probabilities: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
