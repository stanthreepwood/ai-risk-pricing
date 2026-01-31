"""
Frequency modeling for AI catastrophe events.

Implements Poisson-based event occurrence models. In catastrophe modeling,
frequency represents how often events occur, separate from how severe they are.
"""

import numpy as np
from numpy.typing import NDArray

from ..utils.distributions import sample_poisson


class FrequencyModel:
    """
    Poisson frequency model for catastrophe event occurrence.
    
    The frequency model determines how many events occur in each simulation
    year. The Poisson distribution is standard in catastrophe modeling because
    it assumes events occur independently at a constant rate over time.
    
    - Lambda (λ) represents the expected annual event count
    - P(N=k) = (λ^k * e^-λ) / k! gives probability of k events
    - For rare events (λ << 1), most years have zero events
    - The Poisson assumption implies events are independent
    
    Future extensions could include:
        - Non-homogeneous Poisson (time-varying intensity)
        - Negative binomial (overdispersion / clustering)
        - Self-exciting processes (event clustering)
    """
    
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """
        Initialize the frequency model.
        
        Args:
            rng: NumPy random generator for reproducibility.
        """
        self.rng = rng or np.random.default_rng()
    
    def sample_event_count(self, lambda_: float) -> int:
        """
        Sample the number of events occurring in a single year.
        
        This is the core frequency sampling function. Given an annual
        event rate, returns the realized number of events for one year.
        
        - lambda_ = 0.1 → expect 1 event per 10 years, most years 0
        - lambda_ = 1.0 → expect 1 event per year on average
        - lambda_ = 0.01 → 1-in-100 year event frequency
        """
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        
        return int(sample_poisson(lambda_, size=1, rng=self.rng)[0])
    
    def sample_event_counts(self, lambda_: float, n_years: int) -> NDArray[np.int64]:
        """
        Sample event counts for multiple simulation years (vectorized).
        
        Generates event counts for an entire simulation run.
        This is the preferred method for Monte Carlo simulation due to
        vectorization efficiency.
        """
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        if n_years <= 0:
            raise ValueError(f"n_years must be positive, got {n_years}")
        
        return sample_poisson(lambda_, size=n_years, rng=self.rng)
    
    def sample_multi_scenario_counts(
        self,
        lambdas: list[float],
        n_years: int,
    ) -> NDArray[np.int64]:
        """
        Sample event counts for multiple scenarios across multiple years.
        
        Returns a 2D array where each row is a scenario and each column
        is a simulation year. This enables vectorized simulation across
        all scenarios simultaneously.
        
        In a multi-peril catastrophe model, we simulate multiple
        event types (earthquake, hurricane, etc.) simultaneously.
        For AI risk, each scenario type has its own frequency.
        """
        n_scenarios = len(lambdas)
        counts = np.zeros((n_scenarios, n_years), dtype=np.int64)
        
        for i, lambda_ in enumerate(lambdas):
            counts[i, :] = self.sample_event_counts(lambda_, n_years)
        
        return counts
    
    def expected_events_per_year(self, lambda_: float) -> float:
        """
        Return the expected number of events per year.
        
        For a Poisson distribution, E[N] = λ. This is useful for
        analytical validation and reporting.
        """
        return lambda_
    
    def probability_of_event(self, lambda_: float) -> float:
        """
        Calculate probability that at least one event occurs in a year.
        
        P(N >= 1) = 1 - P(N = 0) = 1 - e^(-λ)
        
        This is useful for reporting event probabilities in intuitive terms.
        """
        return 1.0 - np.exp(-lambda_)
    
    def return_period_to_lambda(self, return_period: float) -> float:
        """
        Convert a return period to a Poisson intensity.
        
        Return period T means "1 event every T years on average."
        λ = 1/T
        
        A "1-in-100 year event" has return period T=100 and λ=0.01.
        This does NOT mean exactly one event per 100 years, but that
        the probability of an event in any given year is 1%.
        """
        if return_period <= 0:
            raise ValueError(f"Return period must be positive, got {return_period}")
        return 1.0 / return_period
    
    def lambda_to_return_period(self, lambda_: float) -> float:
        """
        Convert a Poisson intensity to a return period.
        
        T = 1/λ
        """
        if lambda_ <= 0:
            raise ValueError(f"Lambda must be positive for return period, got {lambda_}")
        return 1.0 / lambda_


class NonHomogeneousPoissonModel(FrequencyModel):
    """
    Non-homogeneous Poisson process with time-varying intensity.
    
    Extension point for modeling scenarios where event frequency
    changes over time (e.g., increasing AI capability leading to
    increasing risk frequency).
    
    In climate modeling, hurricane frequency may increase over time.
    Similarly, AI catastrophe frequency may increase as systems
    become more capable and more widely deployed.
    """
    
    def __init__(
        self,
        base_lambda: float,
        trend_rate: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize non-homogeneous Poisson model.
        """
        super().__init__(rng)
        self.base_lambda = base_lambda
        self.trend_rate = trend_rate
    
    def intensity_at_year(self, year: int) -> float:
        """
        Calculate intensity at a specific future year.
        
        Uses exponential growth: λ(t) = λ_0 * exp(r * t)
        """
        return self.base_lambda * np.exp(self.trend_rate * year)
    
    def sample_with_trend(self, n_years: int) -> NDArray[np.int64]:
        """
        Sample event counts with time-varying intensity.
        
        Each year uses its own intensity based on the trend model.
        """
        counts = np.zeros(n_years, dtype=np.int64)
        for year in range(n_years):
            lambda_t = self.intensity_at_year(year)
            counts[year] = self.sample_event_count(lambda_t)
        return counts
