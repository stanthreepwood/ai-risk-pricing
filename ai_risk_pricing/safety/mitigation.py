from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .risk_surface import RiskSurface
from .measure import SafetyProfile

if TYPE_CHECKING:
    from ai_risk_pricing.scenario.schema import Scenario


@dataclass(frozen=True)
class MitigationParams:
    """
    Configuration parameters for mitigation calculations.
    
    These parameters bound the effect of safety measures to ensure
    realistic risk modeling. Even the best safety controls cannot
    eliminate all risk, and there are diminishing returns from
    layering multiple controls.
    
    Attributes:
        min_residual_risk: Floor on residual risk factor [0, 1].
            Even with perfect controls, some risk always remains.
            Default 0.1 means at least 10% of scenario severity persists.
        max_total_reduction: Cap on total risk reduction [0, 1].
            Maximum proportion of severity that can be mitigated.
            Default 0.8 means maximum 80% reduction.
        diminishing_returns: Whether to use multiplicative combination.
            If True, multiple controls combine with diminishing returns.
            If False, uses additive combination (capped at max_total_reduction).
    """
    
    min_residual_risk: float = 0.1
    max_total_reduction: float = 0.8
    diminishing_returns: bool = True
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.min_residual_risk <= 1:
            raise ValueError(
                f"min_residual_risk must be in [0, 1], got {self.min_residual_risk}"
            )
        if not 0 <= self.max_total_reduction <= 1:
            raise ValueError(
                f"max_total_reduction must be in [0, 1], got {self.max_total_reduction}"
            )


class MitigationEngine:
    """
    Computes mitigation factors for scenario-company pairs.
    
    The mitigation factor represents the residual risk after accounting
    for a company's safety controls. A factor of 1.0 means no mitigation
    (full severity applies), while a factor of 0.2 means 80% mitigation
    (only 20% of severity applies).
    
    The engine uses scenario-defined `known_mitigations` to determine
    which risk surfaces are relevant for each scenario, then combines
    the company's coverage of those surfaces to compute overall mitigation.
    
    Mitigation formula:
        For each surface in scenario.known_mitigations:
            surface_reduction = max_reduction_for_surface * company_surface_score
        Total reduction = 1 - Π(1 - surface_reduction)  # diminishing returns
        Residual risk = max(min_residual, 1 - total_reduction)
    
    Example:
        >>> engine = MitigationEngine()
        >>> factor = engine.compute_mitigation_factor(scenario, profile)
        >>> adjusted_severity = base_severity * factor
    """
    
    def __init__(self, params: MitigationParams | None = None) -> None:
        """
        Initialize the mitigation engine.
        
        Args:
            params: Mitigation parameters. Uses defaults if not provided.
        """
        self.params = params or MitigationParams()
    
    def compute_mitigation_factor(
        self,
        scenario: Scenario,
        profile: SafetyProfile,
    ) -> float:
        """
        Compute the mitigation factor for a scenario-profile pair.
        
        The mitigation factor is applied multiplicatively to scenario
        severities during Monte Carlo simulation. Lower factors indicate
        better mitigation (more risk reduction).
        
        Args:
            scenario: The catastrophe scenario being evaluated.
            profile: The company's safety profile.
            
        Returns:
            Mitigation factor in [min_residual_risk, 1.0].
            1.0 = no mitigation, min_residual_risk = maximum mitigation.
        """
        known_mitigations = getattr(scenario, "known_mitigations", None) or {}
        
        if not known_mitigations:
            return 1.0
        
        if not profile.measures:
            return 1.0
        
        surface_reductions: list[float] = []
        
        for surface_str, max_reduction in known_mitigations.items():
            try:
                surface = RiskSurface(surface_str)
            except ValueError:
                continue
            
            company_score = profile.surface_score(surface)
            
            if company_score > 0:
                actual_reduction = max_reduction * company_score
                surface_reductions.append(actual_reduction)
        
        if not surface_reductions:
            return 1.0
        
        if self.params.diminishing_returns:
            residual = 1.0
            for reduction in surface_reductions:
                residual *= (1.0 - reduction)
            total_reduction = 1.0 - residual
        else:
            # additive (capped)
            total_reduction = sum(surface_reductions)
        
        total_reduction = min(total_reduction, self.params.max_total_reduction)
        
        residual_risk = 1.0 - total_reduction
        
        return max(self.params.min_residual_risk, residual_risk)
    
    def compute_surface_reductions(
        self,
        scenario: Scenario,
        profile: SafetyProfile,
    ) -> dict[str, float]:
        """
        Compute per-surface risk reductions for detailed analysis.
        
        Useful for understanding which controls are contributing to
        overall mitigation and identifying coverage gaps.
        
        Args:
            scenario: The catastrophe scenario being evaluated.
            profile: The company's safety profile.
            
        Returns:
            Dict mapping surface names to their individual reduction factors.
        """
        known_mitigations = getattr(scenario, "known_mitigations", None) or {}
        reductions = {}
        
        for surface_str, max_reduction in known_mitigations.items():
            try:
                surface = RiskSurface(surface_str)
            except ValueError:
                continue
            
            company_score = profile.surface_score(surface)
            actual_reduction = max_reduction * company_score
            reductions[surface_str] = actual_reduction
        
        return reductions
    
    def analyze_mitigation(
        self,
        scenario: Scenario,
        profile: SafetyProfile,
    ) -> dict:
        """
        Generate detailed mitigation analysis.
        
        Provides breakdown of how safety controls affect risk for a
        specific scenario, useful for reporting and optimization.
        
        Args:
            scenario: The catastrophe scenario being evaluated.
            profile: The company's safety profile.
            
        Returns:
            Dict with analysis including:
            - mitigation_factor: The overall residual risk factor
            - total_reduction: Total risk reduction achieved
            - surface_reductions: Per-surface reduction breakdown
            - uncovered_surfaces: Surfaces with no mitigation
            - coverage_gaps: Surfaces the scenario targets but company lacks
        """
        known_mitigations = getattr(scenario, "known_mitigations", None) or {}
        
        surface_reductions = self.compute_surface_reductions(scenario, profile)
    
        # identify coverage gaps
        coverage_gaps = []
        for surface_str in known_mitigations.keys():
            try:
                surface = RiskSurface(surface_str)
                if profile.surface_score(surface) == 0:
                    coverage_gaps.append(surface_str)
            except ValueError:
                pass
        
        mitigation_factor = self.compute_mitigation_factor(scenario, profile)
        total_reduction = 1.0 - mitigation_factor
        
        return {
            "scenario_name": scenario.name,
            "mitigation_factor": mitigation_factor,
            "total_reduction": total_reduction,
            "surface_reductions": surface_reductions,
            "n_relevant_surfaces": len(known_mitigations),
            "n_covered_surfaces": len(surface_reductions) - len(coverage_gaps),
            "coverage_gaps": coverage_gaps,
            "params": {
                "min_residual_risk": self.params.min_residual_risk,
                "max_total_reduction": self.params.max_total_reduction,
                "diminishing_returns": self.params.diminishing_returns,
            },
        }
