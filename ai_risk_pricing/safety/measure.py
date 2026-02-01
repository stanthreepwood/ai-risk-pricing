from dataclasses import dataclass, field

from .risk_surface import RiskSurface


@dataclass
class SafetyMeasure:
    """
    A specific safety control deployed by a company.
    
    Safety measures are the atomic units of risk mitigation in the model.
    Each measure addresses a specific risk surface with a known or estimated
    effectiveness rating.
    
    Example:
        >>> measure = SafetyMeasure(
        ...     surface=RiskSurface.PROMPT_INJECTION,
        ...     provider="Lakera Guard",
        ...     effectiveness=0.85,
        ...     coverage=0.95,
        ... )
    """
    
    surface: RiskSurface
    provider: str
    effectiveness: float
    coverage: float = 1.0
    
    def __post_init__(self) -> None:
        if not 0 <= self.effectiveness <= 1:
            raise ValueError(
                f"effectiveness must be in [0, 1], got {self.effectiveness}"
            )
        if not 0 <= self.coverage <= 1:
            raise ValueError(
                f"coverage must be in [0, 1], got {self.coverage}"
            )
    
    @property
    def effective_strength(self) -> float:
        """
        Compute effective strength combining effectiveness and coverage.
        
        The effective strength represents the actual risk reduction accounting
        for partial coverage. If a control is 80% effective but only covers
        50% of traffic, the effective strength is 40%.
        
        Returns:
            Effective strength in [0, 1].
        """
        return self.effectiveness * self.coverage


@dataclass
class SafetyProfile:
    """
    Complete safety posture of a company.
    
    A SafetyProfile aggregates multiple SafetyMeasures to represent a company's
    overall AI safety posture. It provides methods for computing aggregate
    scores per risk surface and an overall safety score for backwards
    compatibility with the existing Company.safety_score field.
    
    Multiple measures addressing the same risk surface are combined with
    diminishing returns - two 80% effective controls don't give 160% protection,
    but rather ~96% (using 1 - (1-0.8) * (1-0.8) formula).
    
    Attributes:
        measures: List of safety measures deployed by the company.
    
    Example:
        >>> profile = SafetyProfile(measures=[
        ...     SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Lakera Guard", 0.85),
        ...     SafetyMeasure(RiskSurface.MONITORING, "Langfuse", 0.80),
        ... ])
        >>> print(f"Overall safety: {profile.overall_score:.2f}")
    """
    
    measures: list[SafetyMeasure] = field(default_factory=list)
    
    def surface_score(self, surface: RiskSurface) -> float:
        """
        Compute aggregate effectiveness for a specific risk surface.
        
        Combines all measures addressing this surface using the complementary
        probability formula: 1 - Π(1 - measure.effective_strength).
        
        This models diminishing returns - multiple redundant controls provide
        less incremental benefit than their individual ratings suggest.
        
        Args:
            surface: The risk surface to compute the score for.
            
        Returns:
            Aggregate effectiveness in [0, 1]. Returns 0.0 if no measures
            address this surface.
        
        Example:
            If two measures with effective_strength 0.8 and 0.7 address
            the same surface, the combined score is:
            1 - (1-0.8) * (1-0.7) = 1 - 0.2 * 0.3 = 0.94
        """
        surface_measures = [m for m in self.measures if m.surface == surface]
        
        if not surface_measures:
            return 0.0
        
        # Complementary probability: residual risk is product of residuals
        residual = 1.0
        for measure in surface_measures:
            residual *= (1.0 - measure.effective_strength)
        
        return 1.0 - residual
    
    def surface_scores(self) -> dict[RiskSurface, float]:
        surfaces_covered = set(m.surface for m in self.measures)
        return {surface: self.surface_score(surface) for surface in surfaces_covered}
    
    @property
    def overall_score(self) -> float:
        """
        Compute backwards-compatible overall safety score.
        
        The overall score is the average of all surface scores, weighted
        equally. This provides a single scalar metric compatible with the
        existing Company.safety_score field.
        
        For a more nuanced analysis, use surface_score() to examine
        coverage of specific risk surfaces.
        
        Returns:
            Overall safety score in [0, 1]. Returns 0.0 if no measures.
        """
        if not self.measures:
            return 0.0
        
        scores = self.surface_scores()
        if not scores:
            return 0.0
        
        return sum(scores.values()) / len(RiskSurface)
    
    @property
    def covered_surfaces(self) -> set[RiskSurface]:
        """Return the set of risk surfaces covered by at least one measure."""
        return set(m.surface for m in self.measures)
    
    @property
    def uncovered_surfaces(self) -> set[RiskSurface]:
        """Return the set of risk surfaces with no coverage."""
        all_surfaces = set(RiskSurface)
        return all_surfaces - self.covered_surfaces
    
    def coverage_summary(self) -> dict[str, float | int]:
        """
        Generate a summary of safety coverage.
        
        Returns:
            Dictionary with coverage statistics:
            - n_measures: Total number of measures
            - n_surfaces_covered: Number of surfaces with at least one measure
            - overall_score: The overall safety score
            - coverage_ratio: Proportion of surfaces covered
        """
        n_surfaces = len(RiskSurface)
        n_covered = len(self.covered_surfaces)
        
        return {
            "n_measures": len(self.measures),
            "n_surfaces_covered": n_covered,
            "n_surfaces_total": n_surfaces,
            "coverage_ratio": n_covered / n_surfaces,
            "overall_score": self.overall_score,
        }
    
    def add_measure(self, measure: SafetyMeasure) -> None:
        self.measures.append(measure)
    
    @classmethod
    def empty(cls) -> "SafetyProfile":
        return cls(measures=[])
