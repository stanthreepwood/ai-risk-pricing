import numpy as np
from numpy.typing import NDArray

import math
import statistics


def sample_poisson(lambda_: float, size: int = 1, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
    """
    Sample event counts from a Poisson distribution.
    
    In catastrophe modeling, the Poisson distribution represents the number
    of events occurring in a fixed time period (typically one year). The
    parameter lambda represents the expected annual frequency of events.
    
    - lambda = 0.1 means roughly 1 event per 10 years on average
    - lambda = 2.0 means roughly 2 events per year on average
    - The Poisson assumption implies events occur independently
    """
    if lambda_ < 0:
        raise ValueError(f"Poisson lambda must be non-negative, got {lambda_}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.poisson(lam=lambda_, size=size)


def sample_pareto(
    alpha: float,
    scale: float,
    size: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample losses from a Pareto distribution.
    
    The Pareto distribution is fundamental to catastrophe modeling due to
    its heavy tail. It captures the empirical observation that loss severity
    follows a power law: extreme losses are rare but disproportionately large.
    
    - alpha (shape): Controls tail heaviness. Lower alpha = heavier tail.
    - alpha < 1: Infinite mean (extremely heavy tail)
    - alpha < 2: Infinite variance (heavy tail, finite mean)
    - alpha > 2: Finite variance (moderate tail)
    - scale (x_m): Minimum possible loss (threshold parameter)
    
    For AI catastrophe risks, alpha ~ 1.5 reflects extreme uncertainty
    about tail behavior given the absence of historical loss data.
    
    Args:
        alpha: Shape parameter (tail index). Must be > 0.
        scale: Scale parameter (minimum loss). Must be > 0.
        size: Number of samples to draw.
        rng: NumPy random generator for reproducibility.
    """
    if alpha <= 0:
        raise ValueError(f"Pareto alpha must be positive, got {alpha}")
    if scale <= 0:
        raise ValueError(f"Pareto scale must be positive, got {scale}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # numpy's pareto returns (Pareto - 1), so we adjust
    # Standard Pareto: X = scale / U^(1/alpha) where U ~ Uniform(0,1)
    return (rng.pareto(a=alpha, size=size) + 1) * scale


def sample_lognormal(
    mu: float,
    sigma: float,
    size: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample losses from a Lognormal distribution.
    
    The Lognormal distribution is widely used in insurance for modeling
    loss severity. It produces right-skewed distributions where the log
    of losses is normally distributed.
    
    - mu: Mean of the underlying normal distribution (log-scale)
    - sigma: Standard deviation of the underlying normal (log-scale)
    - Higher sigma produces heavier tails and more extreme events
    - The mean of the lognormal is exp(mu + sigma²/2)
    - The median is exp(mu)
    
    For AI risks, lognormal may be preferred over Pareto when moderate
    tail behavior is appropriate (e.g., operational errors vs. systemic
    failures).
    """
    if sigma <= 0:
        return np.full(size, 0)
        raise ValueError(f"Lognormal sigma must be positive, got {sigma}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def sample_from_distribution(
    dist_name: str,
    params: dict,
    size: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Generic distribution sampler with dispatch by name.
    
    Provides a unified interface for sampling from any supported severity
    distribution. Useful when distribution choice is determined at runtime
    (e.g., loaded from scenario configuration).
    
    Args:
        dist_name: Distribution identifier ("pareto" or "lognormal").
        params: Dictionary of distribution parameters.
        size: Number of samples to draw.
        rng: NumPy random generator for reproducibility.
    
    Examples:
        >>> sample_from_distribution("pareto", {"alpha": 1.5, "scale": 10}, size=1000)
        >>> sample_from_distribution("lognormal", {"mu": 4.0, "sigma": 1.5}, size=1000)
    """
    dist_name = dist_name.lower().strip()
    
    if dist_name == "pareto":
        return sample_pareto(
            alpha=params["alpha"],
            scale=params["scale"],
            size=size,
            rng=rng,
        )
    elif dist_name == "lognormal":
        return sample_lognormal(
            mu=params["mu"],
            sigma=params["sigma"],
            size=size,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unsupported distribution: {dist_name}. "
            f"Supported: 'pareto', 'lognormal'"
        )

def fit_lognormal_mom(severities: list[float]) -> tuple[str, dict[str, float]]:
    """Fit lognormal parameters via moment matching to positive severities."""
    x = [float(s) for s in severities if math.isfinite(float(s)) and float(s) > 0.0]
    if len(x) == 0:
        raise ValueError("Cannot fit severity: no positive finite severities.")

    m = float(statistics.fmean(x))
    v = float(statistics.variance(x)) if len(x) > 1 else 0.0
    # Lognormal moment relations: m = exp(mu + s^2/2), v = (exp(s^2)-1)*exp(2mu+s^2)
    sigma2 = float(math.log1p(v / (m * m))) if m > 0 else 0.0
    sigma = float(math.sqrt(max(sigma2, 0.0)))
    mu = float(math.log(m) - 0.5 * sigma2)
    return "lognormal", {"mu": mu, "sigma": sigma}