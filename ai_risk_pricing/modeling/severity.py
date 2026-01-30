"""
Severity modeling for AI catastrophe events.

Implements heavy-tailed loss distributions with capability-based regime
switching. Severity represents the magnitude of loss when an event occurs.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from ..utils.distributions import sample_pareto, sample_lognormal, sample_from_distribution
from ..scenario.schema import SeverityDistribution


class SeverityModel:
    """
    Heavy-tailed severity model with threshold regime switching.
    
    The severity model determines loss magnitude when an event occurs.
    AI catastrophe losses exhibit heavy tails - most events cause moderate
    losses, but rare events can be catastrophic.
    
    Key features:
        1. Heavy-tailed distributions (Pareto, Lognormal)
        2. Capability-based regime switching (losses jump at thresholds)
        3. Tail multipliers for extreme events
    
    Actuarial interpretation:
        Traditional insurance assumes losses follow well-behaved distributions
        calibrated from historical data. AI risks have no loss history, so we
        use heavy-tailed distributions to reflect deep uncertainty about
        potential loss magnitudes.
    
    Capability threshold regime switching models "tipping points" where
    more capable AI systems may cause discontinuously larger losses.
    """
    
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """
        Initialize the severity model.
        
        Args:
            rng: NumPy random generator for reproducibility.
        """
        self.rng = rng or np.random.default_rng()
    
    def sample_severity(
        self,
        dist_name: str,
        params: dict,
        size: int = 1,
    ) -> NDArray[np.float64]:
        """
        Sample loss severities from a specified distribution.
        
        This is the base severity sampling function without any
        adjustments or regime switching.
        
        Actuarial interpretation:
            Given that an event occurs, what is the loss amount?
            This samples from the conditional loss distribution.
        
        Args:
            dist_name: Distribution name ("pareto" or "lognormal").
            params: Distribution parameters dictionary.
            size: Number of samples to draw.
        
        Returns:
            Array of sampled loss values.
        """
        return sample_from_distribution(dist_name, params, size, self.rng)
    
    def sample_severity_from_spec(
        self,
        spec: SeverityDistribution,
        size: int = 1,
    ) -> NDArray[np.float64]:
        """
        Sample severities from a SeverityDistribution specification.
        
        Convenience method that accepts a SeverityDistribution object
        instead of separate name and params arguments.
        
        Args:
            spec: SeverityDistribution specification.
            size: Number of samples to draw.
        
        Returns:
            Array of sampled loss values.
        """
        return self.sample_severity(spec.name, spec.params, size)
    
    def sample_with_threshold(
        self,
        dist_name: str,
        params: dict,
        capability_score: float,
        threshold: float,
        threshold_multiplier: float,
        size: int = 1,
    ) -> NDArray[np.float64]:
        """
        Sample severities with capability-based regime switching.
        
        When capability_score exceeds threshold, losses are multiplied
        by threshold_multiplier. This models the hypothesis that more
        capable AI systems may cause discontinuously larger harm.
        
        Actuarial interpretation:
            This is analogous to "clash" scenarios in reinsurance where
            multiple conditions must be met for extreme losses:
            - The event must occur (frequency)
            - The loss must be large (severity)
            - The system must be sufficiently capable (threshold)
        
        The threshold creates a regime switch in the severity distribution,
        modeling tipping-point behavior in AI risk.
        
        Args:
            dist_name: Distribution name ("pareto" or "lognormal").
            params: Distribution parameters dictionary.
            capability_score: Current AI capability level (0-1 scale).
            threshold: Capability threshold triggering regime switch.
            threshold_multiplier: Multiplier applied when threshold exceeded.
            size: Number of samples to draw.
        
        Returns:
            Array of sampled loss values, potentially amplified.
        """
        # Sample base severities
        base_losses = self.sample_severity(dist_name, params, size)
        
        # Apply threshold multiplier if capability exceeds threshold
        if capability_score > threshold:
            base_losses = base_losses * threshold_multiplier
        
        return base_losses
    
    def sample_with_tail_multiplier(
        self,
        dist_name: str,
        params: dict,
        tail_multiplier: float,
        tail_percentile: float = 0.95,
        size: int = 1,
    ) -> NDArray[np.float64]:
        """
        Sample severities with enhanced tail amplification.
        
        Extreme losses (above tail_percentile) are multiplied by
        tail_multiplier. This allows scenario-specific tail behavior
        beyond what the base distribution provides.
        
        Actuarial interpretation:
            Some scenarios have "super-heavy" tails where the standard
            distribution underestimates extreme events. The tail multiplier
            allows expert adjustment of specific scenario tails.
        
        Args:
            dist_name: Distribution name ("pareto" or "lognormal").
            params: Distribution parameters dictionary.
            tail_multiplier: Multiplier for extreme tail losses.
            tail_percentile: Threshold for tail (default 95th percentile).
            size: Number of samples to draw.
        
        Returns:
            Array of sampled loss values with tail amplification.
        """
        # Sample base severities
        base_losses = self.sample_severity(dist_name, params, size)
        
        if tail_multiplier > 1.0 and size > 0:
            # Find tail threshold from sample
            tail_threshold = np.percentile(base_losses, tail_percentile * 100)
            
            # Amplify tail losses
            tail_mask = base_losses > tail_threshold
            base_losses[tail_mask] = base_losses[tail_mask] * tail_multiplier
        
        return base_losses
    
    def sample_full_scenario(
        self,
        dist_name: str,
        params: dict,
        capability_score: float,
        threshold: float,
        threshold_multiplier: float,
        tail_multiplier: float,
        size: int = 1,
    ) -> NDArray[np.float64]:
        """
        Sample severities with all adjustments applied.
        
        Combines capability threshold regime switching and tail
        amplification for complete scenario severity modeling.
        
        Order of operations:
            1. Sample from base distribution
            2. Apply tail multiplier to extreme losses
            3. Apply threshold multiplier if capability exceeds threshold
        
        Args:
            dist_name: Distribution name.
            params: Distribution parameters.
            capability_score: Current AI capability level.
            threshold: Capability threshold for regime switch.
            threshold_multiplier: Multiplier when threshold exceeded.
            tail_multiplier: Multiplier for tail losses.
            size: Number of samples.
        
        Returns:
            Array of fully-adjusted loss values.
        """
        # Sample with tail amplification
        losses = self.sample_with_tail_multiplier(
            dist_name=dist_name,
            params=params,
            tail_multiplier=tail_multiplier,
            size=size,
        )
        
        # Apply threshold regime switch
        if capability_score > threshold:
            losses = losses * threshold_multiplier
        
        return losses
    
    def expected_severity(
        self,
        dist_name: str,
        params: dict,
    ) -> float:
        """
        Calculate analytical expected severity (mean loss given event).
        
        For Pareto with alpha > 1: E[X] = alpha * scale / (alpha - 1)
        For Lognormal: E[X] = exp(mu + sigma²/2)
        
        Returns infinity for Pareto with alpha <= 1 (infinite mean).
        
        Args:
            dist_name: Distribution name.
            params: Distribution parameters.
        
        Returns:
            Expected loss value.
        """
        dist_name = dist_name.lower()
        
        if dist_name == "pareto":
            alpha = params["alpha"]
            scale = params["scale"]
            if alpha <= 1:
                return float("inf")
            return alpha * scale / (alpha - 1)
        
        elif dist_name == "lognormal":
            mu = params["mu"]
            sigma = params["sigma"]
            return float(np.exp(mu + sigma**2 / 2))
        
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
    
    def coefficient_of_variation(
        self,
        dist_name: str,
        params: dict,
    ) -> float:
        """
        Calculate coefficient of variation (CV = std/mean).
        
        Higher CV indicates more variability relative to mean.
        Heavy-tailed distributions have high CV.
        
        For Pareto with alpha > 2: CV = 1 / sqrt(alpha * (alpha - 2))
        For Lognormal: CV = sqrt(exp(sigma²) - 1)
        
        Args:
            dist_name: Distribution name.
            params: Distribution parameters.
        
        Returns:
            Coefficient of variation (undefined if mean or variance infinite).
        """
        dist_name = dist_name.lower()
        
        if dist_name == "pareto":
            alpha = params["alpha"]
            if alpha <= 2:
                return float("inf")
            return 1.0 / np.sqrt(alpha * (alpha - 2))
        
        elif dist_name == "lognormal":
            sigma = params["sigma"]
            return float(np.sqrt(np.exp(sigma**2) - 1))
        
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
