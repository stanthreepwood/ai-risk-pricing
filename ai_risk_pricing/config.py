"""
Configuration parameters for the AI Catastrophe Model.

Contains global simulation parameters, default loading factors, and model
calibration constants. These values represent expert judgment in the absence
of historical loss data for AI catastrophe events.
"""

from dataclasses import dataclass
from typing import Final


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

DEFAULT_SIMULATION_YEARS: Final[int] = 100_000
"""Number of Monte Carlo simulation years for loss distribution convergence."""

RANDOM_SEED: Final[int | None] = None
"""Optional seed for reproducibility. None for production runs."""


# =============================================================================
# FREQUENCY CALIBRATION
# =============================================================================

@dataclass(frozen=True)
class FrequencyParams:
    """
    Baseline frequency parameters for AI catastrophe events.
    
    These represent annual occurrence rates (lambda) for a Poisson process.
    Calibrated via expert elicitation in absence of credible loss history.
    """
    
    # Major AI system failure affecting multiple enterprises
    systemic_failure: float = 0.15
    
    # Model collapse or capability degradation
    model_collapse: float = 0.25
    
    # Adversarial attack or manipulation
    adversarial_attack: float = 0.35
    
    # Alignment failure with harmful outputs
    alignment_failure: float = 0.08
    
    # Regulatory intervention causing sudden shutdown
    regulatory_shock: float = 0.20


# =============================================================================
# SEVERITY CALIBRATION
# =============================================================================

@dataclass(frozen=True)
class SeverityParams:
    """
    Default severity distribution parameters.
    
    Heavy-tailed distributions are essential for catastrophe modeling because
    AI risks exhibit extreme tail behavior - most years have modest losses,
    but rare events can produce outsized losses that dominate the distribution.
    """
    
    # Pareto shape parameter (alpha) - lower = heavier tail
    # Alpha < 2 implies infinite variance (extreme tail risk)
    pareto_alpha: float = 1.5
    
    # Pareto scale parameter (minimum loss threshold in $M)
    pareto_scale: float = 10.0
    
    # Lognormal mu (log-scale mean)
    lognormal_mu: float = 4.0
    
    # Lognormal sigma (log-scale std dev) - higher = heavier tail
    lognormal_sigma: float = 1.8


# =============================================================================
# PRICING PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class PricingParams:
    """
    Loading factors for technical premium calculation.
    
    In catastrophe pricing, ambiguity loading compensates for parameter
    uncertainty. Unlike traditional lines with credible loss data, AI risks
    have no historical basis, requiring substantial ambiguity margins.
    """
    
    # Ambiguity load multiplied against TVaR
    # Higher values reflect greater parameter uncertainty
    ambiguity_load: float = 0.50
    
    # Expense ratio as proportion of expected loss
    expense_ratio: float = 0.25
    
    # VaR confidence level for risk metrics
    var_confidence: float = 0.99
    
    # TVaR confidence level (typically same as VaR)
    tvar_confidence: float = 0.99


# =============================================================================
# DEPENDENCY GRAPH PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class DependencyParams:
    """
    Parameters governing loss propagation through the AI supply chain.
    
    AI systems exhibit concentration risk - a small number of foundation
    models underpin thousands of downstream applications. This creates
    systemic exposure where a single point of failure cascades losses.
    """
    
    # Base propagation factor (proportion of upstream loss transmitted)
    base_propagation: float = 0.65
    
    # Concentration index exponent for nonlinear amplification
    concentration_exponent: float = 2.0
    
    # Maximum amplification factor to prevent runaway losses
    max_amplification: float = 5.0


# =============================================================================
# CAPABILITY THRESHOLD PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class CapabilityParams:
    """
    Parameters for capability-triggered regime switching.
    
    AI risks may exhibit discontinuous behavior at capability thresholds.
    When systems cross certain capability boundaries, loss severity may
    jump discontinuously rather than increase smoothly.
    """
    
    # Default capability threshold (0-1 scale)
    default_threshold: float = 0.7
    
    # Severity multiplier when threshold exceeded
    threshold_multiplier: float = 3.0


# =============================================================================
# DARK SCENARIO MODE
# =============================================================================

@dataclass(frozen=True)
class DarkScenarioParams:
    """
    Parameters for extreme tail scenario injection.
    
    Dark scenarios represent low-probability, high-severity events that
    may not be captured by standard stochastic simulation. These are
    deterministic stress tests layered onto the loss distribution.
    """
    
    # Annual probability of dark scenario occurring
    occurrence_probability: float = 0.001
    
    # Loss multiplier relative to 99.9th percentile
    severity_multiplier: float = 10.0
    
    # Whether dark scenario mode is enabled by default
    enabled: bool = False


# =============================================================================
# AGGREGATE CONFIG
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Complete configuration for an AI catastrophe model run."""
    
    simulation_years: int = DEFAULT_SIMULATION_YEARS
    random_seed: int | None = RANDOM_SEED
    frequency: FrequencyParams = FrequencyParams()
    severity: SeverityParams = SeverityParams()
    pricing: PricingParams = PricingParams()
    dependency: DependencyParams = DependencyParams()
    capability: CapabilityParams = CapabilityParams()
    dark_scenario: DarkScenarioParams = DarkScenarioParams()
    
    def with_dark_mode(self, enabled: bool = True) -> "ModelConfig":
        """Return a new config with dark scenario mode toggled."""
        return ModelConfig(
            simulation_years=self.simulation_years,
            random_seed=self.random_seed,
            frequency=self.frequency,
            severity=self.severity,
            pricing=self.pricing,
            dependency=self.dependency,
            capability=self.capability,
            dark_scenario=DarkScenarioParams(
                occurrence_probability=self.dark_scenario.occurrence_probability,
                severity_multiplier=self.dark_scenario.severity_multiplier,
                enabled=enabled,
            ),
        )


# Default configuration instance
DEFAULT_CONFIG = ModelConfig()
