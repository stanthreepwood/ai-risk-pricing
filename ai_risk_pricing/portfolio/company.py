from dataclasses import dataclass, field
from typing import Iterator
import numpy as np
from enum import Enum

class Sector(str, Enum):
    """
    A sector of the economy.
    """
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"

@dataclass
class Company:
    """
    An insured company with AI-related risk characteristics.
    
    Each company's exposure to AI catastrophe events depends on:
    - Revenue (determines maximum potential loss)
    - AI dependency (how reliant operations are on AI)
    - Autonomy level (degree of autonomous AI decision-making)
    - Safety score (quality of AI risk management)
    """
    
    name: str
    revenue: float
    ai_dependency_score: float
    autonomy_level: float
    safety_score: float
    sector: Sector = Sector.TECHNOLOGY
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.revenue < 0:
            raise ValueError(f"Revenue must be non-negative, got {self.revenue}")
        
        for attr_name in ["ai_dependency_score", "autonomy_level", "safety_score"]:
            value = getattr(self, attr_name)
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{attr_name} must be in [0, 1], got {value}"
                )
    
    @property
    def exposure(self) -> float:
        """
        Calculate AI catastrophe exposure for this company.
        
        Exposure represents the maximum potential loss in an AI catastrophe
        event. It is derived from revenue and risk characteristics.
        
        Formula:
            exposure = revenue * ai_dependency * (1 + autonomy) * (1 - 0.5 * safety)
        
        - Base exposure scales with revenue (larger companies = larger losses)
        - AI dependency multiplies exposure (no AI = no AI risk)
        - Autonomy adds to exposure (autonomous systems have amplified failure modes)
        - Safety reduces exposure (good controls limit damage)
        """
        base = self.revenue * self.ai_dependency_score
        
        #autonomy amplification (autonomous systems can fail more catastrophically)
        autonomy_factor = 1 + self.autonomy_level
        
        # TODO: unmock this
        #safety mitigation (good safety practices limit exposure)
        #maximum 50% reduction from safety
        safety_factor = 1 - 0.5 * self.safety_score
        
        return base * autonomy_factor * safety_factor
    
    @property
    def risk_score(self) -> float:
        """
        Calculate overall AI risk score for this company.
        
        A normalized score combining all risk factors, useful for
        risk ranking and portfolio analysis.
        
        Returns:
            Risk score between 0 (lowest) and 1 (highest).
        """
        # Weighted combination of risk factors
        # Higher AI dependency and autonomy increase risk
        # Higher safety decreases risk
        raw_score = (
            0.4 * self.ai_dependency_score
            + 0.4 * self.autonomy_level
            - 0.2 * self.safety_score
        )
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, raw_score + 0.2))


@dataclass
class Portfolio:
    """
    Collection of companies representing the insured portfolio.
    
    Total portfolio exposure determines capital requirements and
    premium volume. Portfolio composition affects risk metrics
    through correlation and concentration effects.
    """
    
    name: str
    companies: list[Company] = field(default_factory=list)
    
    def add_company(self, company: Company) -> None:
        """Add a company to the portfolio."""
        self.companies.append(company)
    
    def __iter__(self) -> Iterator[Company]:
        """Iterate over companies in the portfolio."""
        return iter(self.companies)
    
    def __len__(self) -> int:
        """Return number of companies in portfolio."""
        return len(self.companies)
    
    @property
    def total_exposure(self) -> float:
        """
        Calculate total portfolio exposure.
        
        Sum of individual company exposures. This is the theoretical
        maximum loss if all companies were fully affected.
        """
        return sum(c.exposure for c in self.companies)
    
    @property
    def total_revenue(self) -> float:
        """Calculate total portfolio revenue."""
        return sum(c.revenue for c in self.companies)
    
    @property
    def average_risk_score(self) -> float:
        """Calculate exposure-weighted average risk score."""
        if not self.companies:
            return 0.0
        
        total_exposure = self.total_exposure
        if total_exposure == 0:
            return 0.0
        
        weighted_sum = sum(c.exposure * c.risk_score for c in self.companies)
        return weighted_sum / total_exposure
    
    @property
    def average_ai_dependency(self) -> float:
        """Calculate exposure-weighted average AI dependency."""
        if not self.companies:
            return 0.0
        
        total_exposure = self.total_exposure
        if total_exposure == 0:
            return 0.0
        
        weighted_sum = sum(c.exposure * c.ai_dependency_score for c in self.companies)
        return weighted_sum / total_exposure
    
    def exposure_by_sector(self) -> dict[str, float]:
        """Calculate total exposure grouped by sector."""
        sector_exposure: dict[str, float] = {}
        for company in self.companies:
            sector = company.sector.value
            sector_exposure[sector] = sector_exposure.get(sector, 0) + company.exposure
        return sector_exposure
    
    def summary(self) -> dict:
        """
        Generate portfolio summary statistics.
        """
        if not self.companies:
            return {"error": "Empty portfolio"}
        
        exposures = [c.exposure for c in self.companies]
        risk_scores = [c.risk_score for c in self.companies]
        
        return {
            "n_companies": len(self.companies),
            "total_revenue_M": self.total_revenue,
            "total_exposure_M": self.total_exposure,
            "average_exposure_M": np.mean(exposures),
            "max_exposure_M": np.max(exposures),
            "min_exposure_M": np.min(exposures),
            "average_risk_score": self.average_risk_score,
            "max_risk_score": np.max(risk_scores),
            "average_ai_dependency": self.average_ai_dependency,
            "sectors": list(self.exposure_by_sector().keys()),
        }
    
    @classmethod
    def build_sample_portfolio(cls, n_companies: int = 10, seed: int | None = None) -> "Portfolio":
        """
        Build a sample portfolio with realistic company characteristics.
        
        Generates a diverse portfolio with varying sizes, AI dependencies,
        and risk profiles for demonstration and testing.
        """
        rng = np.random.default_rng(seed)
        
        # TODO: unmock sector profiles (sector: (base_revenue, ai_dep_mean, autonomy_mean, safety_mean))
        sector_profiles = {
            "technology": (500, 0.8, 0.6, 0.7),
            "financial_services": (800, 0.7, 0.4, 0.75),
            "healthcare": (400, 0.5, 0.3, 0.8),
            "manufacturing": (600, 0.4, 0.5, 0.6),
            "retail": (300, 0.6, 0.3, 0.55),
        }
        
        sectors = list(sector_profiles.keys())
        portfolio = cls(name="Sample AI Risk Portfolio")
        
        for i in range(n_companies):
            # Randomly assign sector
            sector = rng.choice(sectors)
            base_rev, ai_mean, auto_mean, safety_mean = sector_profiles[sector]
            
            # Generate characteristics with sector-based variation
            revenue = float(rng.lognormal(np.log(base_rev), 0.5))
            ai_dependency = float(np.clip(rng.normal(ai_mean, 0.15), 0.1, 0.95))
            autonomy = float(np.clip(rng.normal(auto_mean, 0.15), 0.1, 0.9))
            safety = float(np.clip(rng.normal(safety_mean, 0.1), 0.2, 0.95))
            
            company = Company(
                name=f"{sector.replace('_', ' ').title()} Co {i+1}",
                revenue=revenue,
                ai_dependency_score=ai_dependency,
                autonomy_level=autonomy,
                safety_score=safety,
                sector=Sector(sector),
            )
            portfolio.add_company(company)
        return portfolio
