"""
Risk metrics for catastrophe loss distributions.

Implements standard actuarial risk measures used in catastrophe pricing
and capital modeling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskMetricResults:
    """
    Container for computed risk metrics.
    
    Attributes:
        expected_loss: Mean annual loss (Pure Premium).
        var_99: 99th percentile loss (Value at Risk).
        tvar_99: Mean of losses exceeding VaR 99 (Tail Value at Risk).
        var_995: 99.5th percentile loss.
        tvar_995: Mean of losses exceeding VaR 99.5.
        max_loss: Maximum observed loss in simulation.
        std_dev: Standard deviation of annual losses.
        loss_ratio: Expected loss as fraction of total exposure.
        occurrence_rate: Proportion of years with non-zero loss.
    """
    
    expected_loss: float
    var_99: float
    tvar_99: float
    var_995: float
    tvar_995: float
    max_loss: float
    std_dev: float
    loss_ratio: float
    occurrence_rate: float


class RiskMetrics:
    """
    Calculator for catastrophe risk metrics.
    
    Computes standard risk measures from a Year Loss Table, including
    expected loss, VaR, and TVaR at various confidence levels.
    
    Actuarial interpretation:
        These metrics form the basis of technical pricing and capital
        allocation in catastrophe reinsurance:
        
        - Expected Loss (EL): The average annual loss, forms the base
          of the pure premium.
        
        - Value at Risk (VaR): The loss threshold such that the
          probability of exceeding it equals (1 - confidence).
          VaR 99% is the loss exceeded only 1% of years.
        
        - Tail Value at Risk (TVaR): The expected loss given that
          loss exceeds VaR. Also called Conditional Tail Expectation
          (CTE) or Expected Shortfall (ES). TVaR is preferred for
          capital calculations because it captures tail severity,
          not just tail probability.
    """
    
    def __init__(self, year_loss_table: pd.DataFrame) -> None:
        """
        Initialize with a Year Loss Table.
        
        Args:
            year_loss_table: DataFrame with 'year' and 'loss' columns.
        """
        if "loss" not in year_loss_table.columns:
            raise ValueError("Year Loss Table must have 'loss' column")
        
        self.ylt = year_loss_table
        self.losses = year_loss_table["loss"].values
    
    def expected_loss(self) -> float:
        """
        Calculate Expected Loss (Pure Premium).
        
        The arithmetic mean of annual losses. This is the minimum premium
        needed to cover losses on average (before expenses and margins).
        
        Actuarial interpretation:
            EL = E[L] = (1/n) * Σ L_i
            
            In pricing terms, this is the "burning cost" - what losses
            would cost if spread evenly across all years.
        
        Returns:
            Expected annual loss.
        """
        return float(np.mean(self.losses))
    
    def var(self, confidence: float = 0.99) -> float:
        """
        Calculate Value at Risk at specified confidence level.
        
        VaR is the loss amount such that the probability of exceeding
        it equals (1 - confidence).
        
        Actuarial interpretation:
            VaR_99% answers: "What loss do we expect to exceed only
            1 year in 100?"
            
            VaR is widely used but has limitations:
            - It doesn't indicate how bad losses are when exceeded
            - It's not sub-additive (can discourage diversification)
        
        Args:
            confidence: Confidence level (e.g., 0.99 for 99%).
        
        Returns:
            VaR at specified confidence.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")
        
        return float(np.percentile(self.losses, confidence * 100))
    
    def tvar(self, confidence: float = 0.99) -> float:
        """
        Calculate Tail Value at Risk (Expected Shortfall).
        
        TVaR is the expected loss conditional on exceeding VaR.
        It answers: "When losses are bad, how bad are they on average?"
        
        Actuarial interpretation:
            TVaR_99% = E[L | L > VaR_99%]
            
            This is the mean of all losses in the tail beyond VaR.
            TVaR is preferred over VaR for capital calculations because:
            - It captures tail severity, not just tail probability
            - It is coherent and sub-additive
            - It provides incentive for diversification
        
        For catastrophe risks with heavy tails, TVaR >> VaR,
        indicating severe tail risk.
        
        Args:
            confidence: Confidence level (e.g., 0.99 for 99%).
        
        Returns:
            TVaR (Conditional Tail Expectation) at specified confidence.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")
        
        var_threshold = self.var(confidence)
        tail_losses = self.losses[self.losses > var_threshold]
        
        if len(tail_losses) == 0:
            # If no losses exceed VaR, return VaR itself
            return var_threshold
        
        return float(np.mean(tail_losses))
    
    def calculate_all(self, total_exposure: float | None = None) -> RiskMetricResults:
        """
        Calculate all risk metrics.
        
        Computes a comprehensive set of metrics for pricing and
        capital analysis.
        
        Args:
            total_exposure: Total portfolio exposure for loss ratio calculation.
        
        Returns:
            RiskMetricResults containing all computed metrics.
        """
        el = self.expected_loss()
        var_99 = self.var(0.99)
        tvar_99 = self.tvar(0.99)
        var_995 = self.var(0.995)
        tvar_995 = self.tvar(0.995)
        max_loss = float(np.max(self.losses))
        std_dev = float(np.std(self.losses))
        
        # Calculate loss ratio if exposure provided
        loss_ratio = el / total_exposure if total_exposure and total_exposure > 0 else 0.0
        
        # Calculate occurrence rate
        occurrence_rate = float(np.mean(self.losses > 0))
        
        return RiskMetricResults(
            expected_loss=el,
            var_99=var_99,
            tvar_99=tvar_99,
            var_995=var_995,
            tvar_995=tvar_995,
            max_loss=max_loss,
            std_dev=std_dev,
            loss_ratio=loss_ratio,
            occurrence_rate=occurrence_rate,
        )
    
    def return_period_loss(self, return_period: float) -> float:
        """
        Calculate loss corresponding to a specific return period.
        
        Return period T means "this loss is expected once every T years."
        The corresponding confidence level is (1 - 1/T).
        
        Actuarial interpretation:
            A 1-in-100 year loss is exceeded, on average, once per 100 years.
            This does NOT mean exactly once in any 100-year period, but
            rather a 1% annual exceedance probability.
        
        Args:
            return_period: Return period in years (e.g., 100 for 1-in-100).
        
        Returns:
            Loss amount at specified return period.
        """
        if return_period <= 1:
            raise ValueError(f"Return period must be > 1, got {return_period}")
        
        confidence = 1 - 1 / return_period
        return self.var(confidence)
    
    def exceedance_probability(self, loss_threshold: float) -> float:
        """
        Calculate probability of exceeding a specified loss.
        
        The empirical exceedance probability from the simulation.
        
        Args:
            loss_threshold: Loss amount to calculate exceedance for.
        
        Returns:
            Probability of exceeding the threshold.
        """
        return float(np.mean(self.losses > loss_threshold))
    
    def loss_at_percentile(self, percentile: float) -> float:
        """
        Calculate loss at a specific percentile.
        
        Args:
            percentile: Percentile (0-100 scale, e.g., 99 for 99th percentile).
        
        Returns:
            Loss value at specified percentile.
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}")
        
        return float(np.percentile(self.losses, percentile))
    
    def summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of key risk metrics.
        
        Returns:
            DataFrame with metric names, values, and descriptions.
        """
        metrics = self.calculate_all()
        
        data = [
            ("Expected Loss (EL)", metrics.expected_loss, "Mean annual loss"),
            ("Standard Deviation", metrics.std_dev, "Loss volatility"),
            ("VaR 99%", metrics.var_99, "1-in-100 year loss"),
            ("TVaR 99%", metrics.tvar_99, "Expected loss when exceeding VaR 99%"),
            ("VaR 99.5%", metrics.var_995, "1-in-200 year loss"),
            ("TVaR 99.5%", metrics.tvar_995, "Expected loss when exceeding VaR 99.5%"),
            ("Maximum Loss", metrics.max_loss, "Largest simulated loss"),
            ("Occurrence Rate", metrics.occurrence_rate, "Proportion of years with loss"),
        ]
        
        return pd.DataFrame(data, columns=["Metric", "Value ($M)", "Description"])
