"""
Premium calculation for AI catastrophe risk.

Implements technical premium formulas used in catastrophe reinsurance,
adapted for AI risks where historical loss data is unavailable.
"""

from dataclasses import dataclass

from .risk_metrics import RiskMetrics, RiskMetricResults
from ..config import PricingParams


@dataclass
class PremiumBreakdown:
    """
    Detailed breakdown of premium components.
    
    Actuarial interpretation:
        The technical premium has several components:
        - Expected loss: Base cost of claims
        - Ambiguity load: Charge for parameter uncertainty
        - Risk load: Additional margin for volatility
        - Expenses: Operating costs and commissions
    
    Attributes:
        expected_loss: Pure premium (mean loss).
        ambiguity_load: Additional premium for parameter uncertainty.
        risk_load: Margin for loss volatility.
        expense_load: Operating expenses and acquisition costs.
        total_premium: Sum of all components.
        rate_on_line: Premium as percentage of total exposure.
    """
    
    expected_loss: float
    ambiguity_load: float
    risk_load: float
    expense_load: float
    total_premium: float
    rate_on_line: float
    tvar_multiple: float


class PremiumCalculator:
    """
    Calculator for AI catastrophe insurance premiums.
    
    Implements a risk-load pricing formula appropriate for catastrophe
    risks with parameter uncertainty:
    
        Premium = EL + α * TVaR + ε * EL
    
    Where:
        - EL = Expected Loss (pure premium)
        - α = Ambiguity loading factor (default 0.5)
        - TVaR = Tail Value at Risk
        - ε = Expense ratio
    
    Actuarial interpretation:
        The ambiguity load (α * TVaR) is critical for AI risks because:
        
        1. No historical loss data exists for AI catastrophes
        2. Parameter uncertainty is extreme - we don't know the true
           frequency or severity distributions
        3. The TVaR-based load ensures capital adequacy for tail events
        
        Traditional cat pricing uses experience-based rates, but AI
        risks require a more conservative approach that explicitly
        charges for parameter uncertainty.
        
        Higher α values reflect greater uncertainty about model parameters.
        For a completely unknown risk, α might approach 1.0 or higher.
    """
    
    def __init__(
        self,
        risk_metrics: RiskMetrics,
        total_exposure: float,
        params: PricingParams | None = None,
    ) -> None:
        """
        Initialize the premium calculator.
        
        Args:
            risk_metrics: RiskMetrics instance with computed loss statistics.
            total_exposure: Total portfolio exposure in $M.
            params: Pricing parameters (uses defaults if not provided).
        """
        self.risk_metrics = risk_metrics
        self.total_exposure = total_exposure
        self.params = params or PricingParams()
    
    def calculate_premium(self) -> PremiumBreakdown:
        """
        Calculate the technical premium with full breakdown.
        
        Formula:
            Premium = Expected_Loss
                    + ambiguity_load * TVaR
                    + expense_ratio * Expected_Loss
        
        Actuarial interpretation:
            This formula balances three concerns:
            
            1. Coverage of expected losses (EL)
            2. Capital adequacy for tail events (TVaR load)
            3. Operating costs (expense load)
            
            The TVaR-based ambiguity load is the key innovation for AI risks.
            It ensures that premium reflects tail risk severity, not just
            average loss. This is critical when tails may be much heavier
            than historical analogues suggest.
        
        Returns:
            PremiumBreakdown with all premium components.
        """
        # Get metrics from the risk calculator
        metrics = self.risk_metrics.calculate_all(self.total_exposure)
        
        # Calculate premium components
        el = metrics.expected_loss
        tvar = metrics.tvar_99
        
        # Ambiguity load: compensates for parameter uncertainty
        # Higher α reflects greater uncertainty about true distribution
        ambiguity = self.params.ambiguity_load * tvar
        
        # Risk load: additional margin for volatility (captured in TVaR)
        # For simplicity, we include this in the ambiguity load
        # A separate risk load could use standard deviation
        risk_load = 0.0
        
        # Expense load: operating costs, acquisition, administration
        expense = self.params.expense_ratio * el
        
        # Total technical premium
        total = el + ambiguity + risk_load + expense
        
        # Rate on line: premium as percentage of exposure
        rol = (total / self.total_exposure * 100) if self.total_exposure > 0 else 0.0
        
        # TVaR multiple: ratio of premium to expected loss
        tvar_multiple = (total / el) if el > 0 else 0.0
        
        return PremiumBreakdown(
            expected_loss=el,
            ambiguity_load=ambiguity,
            risk_load=risk_load,
            expense_load=expense,
            total_premium=total,
            rate_on_line=rol,
            tvar_multiple=tvar_multiple,
        )
    
    def calculate_premium_at_confidence(
        self,
        confidence: float = 0.99,
    ) -> PremiumBreakdown:
        """
        Calculate premium using TVaR at specified confidence level.
        
        Allows flexibility in choosing the tail metric for ambiguity load.
        Higher confidence = more conservative premium.
        
        Args:
            confidence: Confidence level for TVaR (e.g., 0.99, 0.995).
        
        Returns:
            PremiumBreakdown computed at specified confidence.
        """
        el = self.risk_metrics.expected_loss()
        tvar = self.risk_metrics.tvar(confidence)
        
        ambiguity = self.params.ambiguity_load * tvar
        expense = self.params.expense_ratio * el
        total = el + ambiguity + expense
        
        rol = (total / self.total_exposure * 100) if self.total_exposure > 0 else 0.0
        tvar_multiple = (total / el) if el > 0 else 0.0
        
        return PremiumBreakdown(
            expected_loss=el,
            ambiguity_load=ambiguity,
            risk_load=0.0,
            expense_load=expense,
            total_premium=total,
            rate_on_line=rol,
            tvar_multiple=tvar_multiple,
        )
    
    def premium_sensitivity(
        self,
        ambiguity_loads: list[float] | None = None,
    ) -> list[tuple[float, float]]:
        """
        Calculate premium sensitivity to ambiguity loading.
        
        Useful for understanding how uncertainty pricing affects
        the final premium.
        
        Args:
            ambiguity_loads: List of α values to test (defaults to range).
        
        Returns:
            List of (ambiguity_load, premium) tuples.
        """
        if ambiguity_loads is None:
            ambiguity_loads = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        results = []
        el = self.risk_metrics.expected_loss()
        tvar = self.risk_metrics.tvar(0.99)
        
        for alpha in ambiguity_loads:
            ambiguity = alpha * tvar
            expense = self.params.expense_ratio * el
            total = el + ambiguity + expense
            results.append((alpha, total))
        
        return results
    
    def report(self) -> str:
        """
        Generate a formatted premium report.
        
        Returns:
            Multi-line string with premium breakdown and rationale.
        """
        breakdown = self.calculate_premium()
        metrics = self.risk_metrics.calculate_all(self.total_exposure)
        
        lines = [
            "=" * 60,
            "AI CATASTROPHE RISK - PREMIUM CALCULATION",
            "=" * 60,
            "",
            "RISK METRICS:",
            f"  Expected Loss (EL):        ${breakdown.expected_loss:>12,.2f} M",
            f"  VaR 99%:                   ${metrics.var_99:>12,.2f} M",
            f"  TVaR 99%:                  ${metrics.tvar_99:>12,.2f} M",
            f"  Maximum Simulated Loss:    ${metrics.max_loss:>12,.2f} M",
            f"  Loss Occurrence Rate:      {metrics.occurrence_rate:>12.1%}",
            "",
            "PREMIUM COMPONENTS:",
            f"  Expected Loss:             ${breakdown.expected_loss:>12,.2f} M",
            f"  Ambiguity Load ({self.params.ambiguity_load:.0%} TVaR):   ${breakdown.ambiguity_load:>12,.2f} M",
            f"  Expense Load ({self.params.expense_ratio:.0%} EL):      ${breakdown.expense_load:>12,.2f} M",
            "",
            f"  TOTAL PREMIUM:             ${breakdown.total_premium:>12,.2f} M",
            "",
            "KEY RATIOS:",
            f"  Rate on Line (RoL):        {breakdown.rate_on_line:>12.2f}%",
            f"  Premium / EL Multiple:     {breakdown.tvar_multiple:>12.2f}x",
            f"  TVaR / EL Ratio:           {metrics.tvar_99 / metrics.expected_loss if metrics.expected_loss > 0 else 0:>12.2f}x",
            "",
            "RATIONALE FOR AMBIGUITY LOADING:",
            "  AI catastrophe risks have no historical loss data.",
            "  Parameter uncertainty is extreme - true frequency and",
            "  severity distributions are unknown. The ambiguity load",
            f"  ({self.params.ambiguity_load:.0%} of TVaR) provides capital adequacy margin",
            "  for this fundamental uncertainty about model parameters.",
            "=" * 60,
        ]
        
        return "\n".join(lines)
