"""
Scenario schema definitions for AI catastrophe events.

Defines the data structures representing catastrophe scenarios. Each scenario
encapsulates the characteristics of a potential AI failure mode, including
its frequency, severity distribution, and propagation behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """
    Classification of AI catastrophe event types.
    
    Each event type represents a distinct failure mode with different
    risk characteristics, propagation patterns, and severity profiles.
    """
    
    SYSTEMIC_FAILURE = "systemic_failure"
    """Correlated failure across AI systems sharing common components."""
    
    MODEL_COLLAPSE = "model_collapse"
    """Degradation or failure of foundation model capabilities."""
    
    ADVERSARIAL_ATTACK = "adversarial_attack"
    """Malicious exploitation or manipulation of AI systems."""
    
    ALIGNMENT_FAILURE = "alignment_failure"
    """AI system producing harmful outputs due to misalignment."""
    
    REGULATORY_SHOCK = "regulatory_shock"
    """Sudden regulatory action causing operational disruption."""
    
    CYBER_PROPAGATION = "cyber_propagation"
    """AI-enabled or AI-targeting cyber attack with cascade effects."""
    
    DATA_POISONING = "data_poisoning"
    """Contamination of training data affecting model behavior."""
    
    DARK_SCENARIO = "dark_scenario"
    """Extreme low-probability, high-severity tail event."""


class PropagationVector(str, Enum):
    """
    Mechanism by which losses propagate through the AI supply chain.
    
    Different propagation vectors have different amplification characteristics
    and affect different parts of the dependency graph.
    """
    
    SUPPLY_CHAIN = "supply_chain"
    """Losses propagate upstream through AI service dependencies."""
    
    API_DEPENDENCY = "api_dependency"
    """Losses propagate through API interconnections."""
    
    MODEL_DEPENDENCY = "model_dependency"
    """Losses propagate through shared foundation model usage."""
    
    DATA_DEPENDENCY = "data_dependency"
    """Losses propagate through shared training data contamination."""
    
    MARKET_CONTAGION = "market_contagion"
    """Losses propagate through market sentiment and confidence."""
    
    REGULATORY_CASCADE = "regulatory_cascade"
    """Losses propagate through regulatory response to incident."""


@dataclass(frozen=True)
class SeverityDistribution:
    """
    Specification for a severity distribution.
    
    Encapsulates the distribution family and parameters used to sample
    individual event losses. Designed for serialization and runtime
    configuration of severity models.
    """
    
    name: str
    """Distribution identifier: 'pareto' or 'lognormal'."""
    
    params: dict[str, float] = field(default_factory=dict)
    """Distribution parameters (e.g., {'alpha': 1.5, 'scale': 10.0})."""
    
    def __post_init__(self) -> None:
        """Validate distribution specification."""
        valid_dists = {"pareto", "lognormal"}
        if self.name.lower() not in valid_dists:
            raise ValueError(
                f"Invalid distribution: {self.name}. Must be one of {valid_dists}"
            )


@dataclass
class Scenario:
    """
    Complete specification of an AI catastrophe scenario.
    
    A scenario represents a specific type of AI failure event with all
    parameters needed for stochastic simulation. Scenarios are the primary
    input to the Monte Carlo engine.
    
    Actuarial interpretation:
        The scenario defines both the frequency (how often events occur)
        and severity (how large losses are when events occur) components
        of the compound frequency-severity model used in catastrophe pricing.
    
    Attributes:
        name: Human-readable scenario identifier.
        event_type: Classification of the failure mode.
        trigger: Description of what initiates the scenario.
        propagation_vector: How losses spread through the system.
        affected_nodes: List of dependency graph nodes impacted.
        base_frequency: Annual expected event count (Poisson lambda).
        severity_distribution: Specification for loss sampling.
        tail_multiplier: Factor applied to extreme tail losses.
        capability_threshold: Capability score triggering regime switch.
        threshold_multiplier: Severity multiplier when threshold exceeded.
        metadata: Optional additional scenario information.
    """
    
    name: str
    event_type: EventType
    trigger: str
    propagation_vector: PropagationVector
    affected_nodes: list[str]
    base_frequency: float
    severity_distribution: SeverityDistribution
    tail_multiplier: float = 1.0
    capability_threshold: float = 0.7
    threshold_multiplier: float = 3.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate scenario parameters."""
        if self.base_frequency < 0:
            raise ValueError(
                f"base_frequency must be non-negative, got {self.base_frequency}"
            )
        if self.tail_multiplier < 1.0:
            raise ValueError(
                f"tail_multiplier must be >= 1.0, got {self.tail_multiplier}"
            )
        if not 0 <= self.capability_threshold <= 1:
            raise ValueError(
                f"capability_threshold must be in [0, 1], got {self.capability_threshold}"
            )
        if self.threshold_multiplier < 1.0:
            raise ValueError(
                f"threshold_multiplier must be >= 1.0, got {self.threshold_multiplier}"
            )
    
    @property
    def is_dark_scenario(self) -> bool:
        """Check if this is a dark (extreme tail) scenario."""
        return self.event_type == EventType.DARK_SCENARIO
    
    @property
    def expected_annual_loss(self) -> float:
        """
        Rough estimate of expected annual loss for this scenario.
        
        For Pareto with alpha > 1: E[X] = alpha * scale / (alpha - 1)
        For Lognormal: E[X] = exp(mu + sigma²/2)
        
        This is a first-order approximation; actual EL comes from simulation.
        """
        dist = self.severity_distribution
        if dist.name == "pareto":
            alpha = dist.params.get("alpha", 1.5)
            scale = dist.params.get("scale", 10.0)
            if alpha <= 1:
                return float("inf")  # Infinite mean for alpha <= 1
            mean_severity = alpha * scale / (alpha - 1)
        else:  # lognormal
            mu = dist.params.get("mu", 4.0)
            sigma = dist.params.get("sigma", 1.5)
            mean_severity = float(np.exp(mu + sigma**2 / 2))
        
        return self.base_frequency * mean_severity


# Avoid circular import by importing numpy only where needed
import numpy as np
