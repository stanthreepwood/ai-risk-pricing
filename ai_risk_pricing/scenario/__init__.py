"""Scenario generation and schema definitions for AI catastrophe events."""

from .schema import Scenario, EventType, PropagationVector
from .generator import ScenarioGenerator

__all__ = ["Scenario", "EventType", "PropagationVector", "ScenarioGenerator"]
