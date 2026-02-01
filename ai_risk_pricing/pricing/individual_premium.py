from dataclasses import dataclass
import numpy as np

from ..portfolio.company import Company, Portfolio, Sector
from ..modeling.dependency import DependencyGraph
from ..modeling.monte_carlo import MonteCarloEngine
from ..scenario.generator import ScenarioGenerator
from ..portfolio.aggregation import PortfolioAggregator
from ..config import ModelConfig, DEFAULT_CONFIG, PricingParams


@dataclass
class IndividualPremiumResult:
    company_name: str
    exposure: float
    risk_score: float
    
    # standalone metrics (company simulated alone)
    standalone_expected_loss: float
    standalone_var_99: float
    standalone_tvar_99: float
    standalone_premium: float
    
    #rate metrics
    rate_on_line: float  # Premium as % of exposure
    loss_cost: float  # EL as % of exposure
    
    # premium components
    expected_loss_component: float
    ambiguity_load_component: float
    expense_load_component: float
    
    def to_dict(self) -> dict:
        return {
            "Company": self.company_name,
            "Exposure ($M)": self.exposure,
            "Risk Score": self.risk_score,
            "Expected Loss ($M)": self.standalone_expected_loss,
            "VaR 99% ($M)": self.standalone_var_99,
            "TVaR 99% ($M)": self.standalone_tvar_99,
            "Total Premium ($M)": self.standalone_premium,
            "Rate on Line (%)": self.rate_on_line,
            "Loss Cost (%)": self.loss_cost,
        }


class IndividualPremiumCalculator:
    """
    Calculator for individual company premiums.
    
    Uses a simplified simulation approach where the company is modeled
    as a single-company portfolio to estimate its standalone risk profile.
    
    The premium formula follows the same structure as portfolio pricing:
        Premium = EL + α * TVaR + ε * EL
    
    Where the risk metrics are derived from the company's individual
    exposure and risk characteristics.
    """
    
    def __init__(
        self,
        config: ModelConfig | None = None,
        n_simulation_years: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.n_simulation_years = n_simulation_years
        self.seed = seed
    
    def calculate_premium(self, company: Company) -> IndividualPremiumResult:
        portfolio = Portfolio(name=f"Single: {company.name}")
        portfolio.add_company(company)
        
        generator = ScenarioGenerator(rng=np.random.default_rng(self.seed))
        scenarios = generator.get_all_scenarios(include_dark=False)
        
        # simplified dep. graph for single company
        aggregator = PortfolioAggregator(portfolio)
        graph = aggregator.build_dependency_graph_from_portfolio(
            n_foundation_models=1,
            n_saas_providers=2,
        )
        
        engine = MonteCarloEngine(
            scenarios=scenarios,
            dependency_graph=graph,
            config=self.config,
            seed=self.seed,
        )
        
        result = engine.simulate_years_vectorized(n_years=self.n_simulation_years)
        losses = result.year_loss_table["loss"].values
        
        el = float(np.mean(losses))
        var_99 = float(np.percentile(losses, 99))
        tail_losses = losses[losses > var_99]
        tvar_99 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_99
        
        pricing_params = self.config.pricing
        
        el_component = el
        ambiguity_component = pricing_params.ambiguity_load * tvar_99
        expense_component = pricing_params.expense_ratio * el
        
        total_premium = el_component + ambiguity_component + expense_component
        
        # rate metrics
        exposure = company.exposure
        rate_on_line = (total_premium / exposure * 100) if exposure > 0 else 0.0
        loss_cost = (el / exposure * 100) if exposure > 0 else 0.0
        
        return IndividualPremiumResult(
            company_name=company.name,
            exposure=exposure,
            risk_score=company.risk_score,
            standalone_expected_loss=el,
            standalone_var_99=var_99,
            standalone_tvar_99=tvar_99,
            standalone_premium=total_premium,
            rate_on_line=rate_on_line,
            loss_cost=loss_cost,
            expected_loss_component=el_component,
            ambiguity_load_component=ambiguity_component,
            expense_load_component=expense_component,
        )
    
    def calculate_quick_premium(self, company: Company) -> IndividualPremiumResult:
        """
        Calculate a quick premium estimate without full simulation.
        
        Uses analytical approximations based on company characteristics
        for rapid underwriting estimates. Less accurate than full simulation
        but suitable for initial quotes.
        
        The approximation uses:
        - Expected frequency from scenario averages
        - Severity scaled by company exposure
        - Standard tail factors from configuration
        """
        exposure = company.exposure
        risk_score = company.risk_score
        
        # Approximate expected annual frequency (sum of scenario lambdas)
        # Weighted by company's vulnerability
        base_frequency = 0.2  # Approximate annual event rate
        adj_frequency = base_frequency * (0.5 + risk_score)
        
        # Approximate severity as fraction of exposure
        # Companies with higher AI dependency face larger losses
        severity_factor = 0.15 + 0.2 * company.ai_dependency_score
        mean_severity = exposure * severity_factor
        
        # Expected loss
        el = adj_frequency * mean_severity
        
        # Approximate VaR/TVaR using typical cat model tail ratios
        # TVaR 99% is typically 3-5x EL for heavy-tailed cat risks
        tail_multiplier = 3.5 + 1.5 * company.autonomy_level  # Higher autonomy = heavier tail
        var_99 = el * 2.5
        tvar_99 = el * tail_multiplier
        
        # Safety score reduces tail severity
        tvar_99 *= (1 - 0.3 * company.safety_score)
        var_99 *= (1 - 0.3 * company.safety_score)
        
        # Calculate premium
        pricing_params = self.config.pricing
        el_component = el
        ambiguity_component = pricing_params.ambiguity_load * tvar_99
        expense_component = pricing_params.expense_ratio * el
        total_premium = el_component + ambiguity_component + expense_component
        
        rate_on_line = (total_premium / exposure * 100) if exposure > 0 else 0.0
        loss_cost = (el / exposure * 100) if exposure > 0 else 0.0
        
        return IndividualPremiumResult(
            company_name=company.name,
            exposure=exposure,
            risk_score=risk_score,
            standalone_expected_loss=el,
            standalone_var_99=var_99,
            standalone_tvar_99=tvar_99,
            standalone_premium=total_premium,
            rate_on_line=rate_on_line,
            loss_cost=loss_cost,
            expected_loss_component=el_component,
            ambiguity_load_component=ambiguity_component,
            expense_load_component=expense_component,
        )


def estimate_safety_investment_benefit(
    company: Company,
    safety_improvement: float,
    calculator: IndividualPremiumCalculator | None = None,
) -> dict:
    """
    Estimate premium reduction from safety investment.
    
    Calculates how much premium could be saved by improving
    the company's safety score, useful for demonstrating
    ROI of AI safety investments.
    
    Args:
        company: Current company profile.
        safety_improvement: How much to increase safety_score (0-1 scale).
        calculator: Premium calculator (creates default if not provided).
        
    Returns:
        Dictionary with before/after premiums and savings.
    """
    calculator = calculator or IndividualPremiumCalculator()
    
    # Calculate current premium
    current_result = calculator.calculate_quick_premium(company)
    
    # Create improved company
    new_safety = min(1.0, company.safety_score + safety_improvement)
    improved_company = Company(
        name=company.name,
        revenue=company.revenue,
        ai_dependency_score=company.ai_dependency_score,
        autonomy_level=company.autonomy_level,
        safety_score=new_safety,
        sector=company.sector,
    )
    
    improved_result = calculator.calculate_quick_premium(improved_company)
    
    premium_reduction = current_result.standalone_premium - improved_result.standalone_premium
    reduction_pct = (premium_reduction / current_result.standalone_premium * 100) if current_result.standalone_premium > 0 else 0.0
    
    return {
        "current_safety_score": company.safety_score,
        "improved_safety_score": new_safety,
        "current_premium": current_result.standalone_premium,
        "improved_premium": improved_result.standalone_premium,
        "premium_reduction": premium_reduction,
        "reduction_percentage": reduction_pct,
        "current_exposure": current_result.exposure,
        "improved_exposure": improved_result.exposure,
    }
