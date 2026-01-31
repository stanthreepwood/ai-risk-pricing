from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ai_risk_pricing.config import ModelConfig
from ai_risk_pricing.modeling.monte_carlo import MonteCarloEngine
from ai_risk_pricing.pricing.risk_metrics import RiskMetrics
from ai_risk_pricing.scenario.schema import Scenario, SeverityDistribution
from ai_risk_pricing.modeling.dependency import DependencyGraph


_INSTITUTIONAL_BLUE = "#123A63"
_ACCENT_RED = "#A61C2B"
_NEUTRAL_GREY = "#5B6770"


def _apply_institutional_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1F2A33",
            "axes.labelcolor": "#1F2A33",
            "xtick.color": "#1F2A33",
            "ytick.color": "#1F2A33",
            "axes.grid": True,
            "grid.color": "#D0D7DE",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.dpi": 160,
        }
    )


def _as_positive_floats(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Coerce a loss vector to finite non-negative floats.

    Losses are modeled as non-negative severities/aggregates. Sanitizing
    inputs prevents plotting artifacts and preserves interpretation.
    """
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.maximum(x, 0.0)


def _ep_from_losses(losses: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exceedance probability plotting positions from annual losses.

    Uses the standard unbiased plotting position p = r/(n+1) for
    annual exceedance probability, where rank 1 corresponds to the
    largest loss year.
    """
    x = _as_positive_floats(losses)
    x_sorted = np.sort(x)[::-1]
    n = len(x_sorted)
    ranks = np.arange(1, n + 1)
    p = ranks / (n + 1)
    return x_sorted, p


def _format_currency_millions(x: float, _pos: int) -> str:
    if x >= 1_000:
        return f"${x/1_000:,.1f}B"
    return f"${x:,.0f}M"


def plot_aep(losses: Sequence[float] | np.ndarray, *, title: str = "Aggregate Exceedance Probability (AEP)") -> Figure:
    """
    Plot Aggregate Exceedance Probability (AEP) curve.

    AEP is the probability that *annual aggregate* portfolio loss exceeds
    a threshold. This is the standard curve used for pricing layers,
    setting attachment points, and communicating tail risk.

    Method:
        - Sort annual losses (descending)
        - Compute exceedance probability using plotting positions
        - Use log scale on X (loss), institutional styling
    """
    _apply_institutional_style()
    x, p = _ep_from_losses(losses)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, p, color=_INSTITUTIONAL_BLUE, linewidth=2.4)
    ax.fill_between(x, p, color=_INSTITUTIONAL_BLUE, alpha=0.10)

    ax.set_xscale("log")
    ax.set_ylabel("Annual Exceedance Probability")
    ax.set_xlabel("Annual Aggregate Loss")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_currency_millions))
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":", alpha=0.5)

    # Focus on meaningful domain: ignore zeros for log scale
    positive = x[x > 0]
    if len(positive) > 0:
        ax.set_xlim(left=max(1.0, float(np.min(positive)) * 0.8), right=float(np.max(positive)) * 1.05)
    ax.set_ylim(bottom=max(1.0 / (len(x) + 1), 1e-6), top=1.0)

    fig.tight_layout()
    return fig


def plot_oep(
    event_losses: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    title: str = "Occurrence Exceedance Probability (OEP)",
) -> Figure:
    """
    Plot Occurrence Exceedance Probability (OEP) curve.

    OEP represents the probability that the *largest single loss in a year*
    exceeds a threshold. This is critical for occurrence-based covers and
    for understanding peak event risk.

    When event-level losses are unavailable, we approximate occurrence loss
    as the maximum scenario contribution per simulated year.
    """
    if isinstance(event_losses, pd.DataFrame):
        if not {"year", "loss"}.issubset(event_losses.columns):
            raise ValueError("event_losses DataFrame must contain columns: 'year', 'loss'")
        per_year = event_losses.groupby("year", as_index=False)["loss"].max()["loss"].to_numpy()
        losses = per_year
    else:
        losses = np.asarray(event_losses, dtype=float)

    fig = plot_aep(losses, title=title)
    ax = fig.axes[0]
    ax.set_xlabel("Annual Maximum Event Loss")
    return fig


def plot_return_period(
    losses: Sequence[float] | np.ndarray,
    *,
    title: str = "Return Period Curve",
) -> Figure:
    """
    Plot a return period curve from annual losses.

    Converts exceedance probability to return period T = 1/P.
    This is the standard committee presentation for capital and
    tolerance framing (e.g., 1-in-200, 1-in-500 year loss).
    """
    _apply_institutional_style()
    x, p = _ep_from_losses(losses)
    rp = 1.0 / p

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(rp, x, color=_INSTITUTIONAL_BLUE, linewidth=2.4)
    ax.fill_between(rp, x, color=_INSTITUTIONAL_BLUE, alpha=0.10)

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (Years)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_currency_millions))
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle=":", alpha=0.5)

    # Committee-friendly ticks
    ticks = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
    ax.set_xticks([t for t in ticks if t >= np.min(rp) and t <= np.max(rp)])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))

    fig.tight_layout()
    return fig


def plot_ruin_probability(
    losses: Sequence[float] | np.ndarray,
    capital_range: Sequence[float] | np.ndarray,
    *,
    title: str = "Capital Adequacy (Ruin Probability)",
) -> Figure:
    """
    Plot capital adequacy via ruin probability.

    Definition:
        ruin_probability(capital) = P(annual_loss > capital)

    This is a simplified one-year solvency view. It is a standard way to
    communicate the implied protection level of a capital buffer against
    an annual loss distribution from a catastrophe model.
    """
    _apply_institutional_style()
    x = _as_positive_floats(losses)
    caps = _as_positive_floats(np.asarray(capital_range, dtype=float))

    ruin = np.array([float(np.mean(x > c)) for c in caps], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(caps, ruin, color=_ACCENT_RED, linewidth=2.4)
    ax.fill_between(caps, ruin, color=_ACCENT_RED, alpha=0.10)

    ax.set_xlabel("Capital")
    ax.set_ylabel("P(Loss > Capital)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_currency_millions))
    ax.set_ylim(bottom=0.0, top=min(1.0, max(ruin) * 1.10 if len(ruin) else 1.0))
    ax.grid(True, which="major")

    fig.tight_layout()
    return fig


def compute_tvar_contributions(loss_df: pd.DataFrame) -> tuple[pd.DataFrame, Figure]:
    """
    Compute and plot TVaR contribution by scenario (99% tail).

    Required input columns:
        - year
        - scenario
        - loss

    Steps:
        - Compute annual aggregate loss by year, then VaR99
        - Filter tail years where total loss > VaR99
        - Compute mean loss by scenario within tail years
        - Normalize contributions to sum to 1.0
        - Plot horizontal bar chart (committee-ready)

    This is a risk attribution view of tail risk. While TVaR is a
    *portfolio* tail metric, scenario contributions help committees
    understand which systemic failure modes dominate tail capital.
    """
    required = {"year", "scenario", "loss"}
    if not required.issubset(loss_df.columns):
        raise ValueError(f"loss_df must contain columns: {sorted(required)}")

    df = loss_df.copy()
    df["loss"] = df["loss"].astype(float)

    annual = df.groupby("year", as_index=False)["loss"].sum().rename(columns={"loss": "annual_loss"})
    var99 = float(np.quantile(annual["annual_loss"].to_numpy(), 0.99))

    tail_years = annual.loc[annual["annual_loss"] > var99, "year"]
    tail = df[df["year"].isin(tail_years)]

    by_scenario = tail.groupby("scenario", as_index=False)["loss"].mean().rename(
        columns={"loss": "mean_tail_loss"}
    )
    total = float(by_scenario["mean_tail_loss"].sum())
    by_scenario["contribution"] = by_scenario["mean_tail_loss"] / total if total > 0 else 0.0
    by_scenario = by_scenario.sort_values("contribution", ascending=True).reset_index(drop=True)

    _apply_institutional_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(by_scenario["scenario"], by_scenario["contribution"], color=_INSTITUTIONAL_BLUE, alpha=0.9)
    ax.set_title("TVaR(99%) Contribution by Scenario")
    ax.set_xlabel("Normalized Contribution to Tail Mean")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    # annotation with VaR99
    ax.text(
        0.99,
        0.02,
        f"VaR(99%): {_format_currency_millions(var99, 0)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=_NEUTRAL_GREY,
        fontsize=10,
    )

    fig.tight_layout()
    return by_scenario, fig


def plot_sensitivity_tornado(
    *,
    scenarios: Sequence[Scenario],
    dependency_graph: DependencyGraph,
    config: ModelConfig,
    n_years: int,
    seed: int,
    title: str = "TVaR(99%) Sensitivity Tornado",
) -> tuple[pd.DataFrame, Figure]:
    """
    Produce a TVaR sensitivity tornado chart over key catastrophe parameters.

    Parameters varied:
        - Pareto alpha (tail index) for Pareto scenarios
        - lambda (scenario frequencies)
        - dependency amplification (concentration exponent)
        - threshold multiplier (capability regime severity multiplier)

    A tornado chart summarizes model risk: which assumptions drive tail
    capital the most. This is essential for governance when pricing
    emerging risks without deep historical calibration.

    This function runs multiple reduced Monte Carlo runs. It is intended
    for reporting packs and may be slower than a single base run.
    """
    def tvar99_for(scenarios_: Sequence[Scenario], config_: ModelConfig, seed_: int) -> float:
        engine = MonteCarloEngine(
            scenarios=scenarios_,
            dependency_graph=dependency_graph,
            config=config_,
            seed=seed_,
        )
        result = engine.simulate_years_vectorized(n_years=n_years)
        metrics = RiskMetrics(result.year_loss_table)
        return float(metrics.calculate_all(total_exposure=dependency_graph.total_exposure()).tvar_99)

    base_tvar = tvar99_for(scenarios, config, seed)

    # --- Define low/high variants (conservative, symmetric) ---
    alpha_delta = 0.20
    lambda_factor_low, lambda_factor_high = 0.75, 1.25
    dep_amp_low, dep_amp_high = 0.80, 1.20
    thresh_low, thresh_high = 0.80, 1.20

    def with_pareto_alpha(delta: float) -> list[Scenario]:
        out: list[Scenario] = []
        for s in scenarios:
            if s.severity_distribution.name.lower() == "pareto":
                alpha = float(s.severity_distribution.params.get("alpha", 1.5))
                new_alpha = max(1.01, alpha + delta)
                out.append(
                    replace(
                        s,
                        severity_distribution=SeverityDistribution(
                            name="pareto",
                            params={**s.severity_distribution.params, "alpha": new_alpha},
                        ),
                    )
                )
            else:
                out.append(s)
        return out

    def with_lambda(factor: float) -> list[Scenario]:
        return [replace(s, base_frequency=float(s.base_frequency) * factor) for s in scenarios]

    def with_threshold_multiplier(factor: float) -> list[Scenario]:
        return [replace(s, threshold_multiplier=float(s.threshold_multiplier) * factor) for s in scenarios]

    def with_dependency_amplification(factor: float) -> ModelConfig:
        dep = config.dependency
        return replace(config, dependency=replace(dep, concentration_exponent=float(dep.concentration_exponent) * factor))

    variants: list[tuple[str, float, float]] = []

    t_low = tvar99_for(with_pareto_alpha(-alpha_delta), config, seed + 1)
    t_high = tvar99_for(with_pareto_alpha(+alpha_delta), config, seed + 2)
    variants.append(("Pareto alpha", t_low, t_high))

    t_low = tvar99_for(with_lambda(lambda_factor_low), config, seed + 3)
    t_high = tvar99_for(with_lambda(lambda_factor_high), config, seed + 4)
    variants.append(("Lambda (frequency)", t_low, t_high))

    t_low = tvar99_for(scenarios, with_dependency_amplification(dep_amp_low), seed + 5)
    t_high = tvar99_for(scenarios, with_dependency_amplification(dep_amp_high), seed + 6)
    variants.append(("Dependency amplification", t_low, t_high))

    t_low = tvar99_for(with_threshold_multiplier(thresh_low), config, seed + 7)
    t_high = tvar99_for(with_threshold_multiplier(thresh_high), config, seed + 8)
    variants.append(("Threshold multiplier", t_low, t_high))

    rows = []
    for name, low, high in variants:
        rows.append(
            {
                "parameter": name,
                "base_tvar99": base_tvar,
                "tvar99_low": low,
                "tvar99_high": high,
                "delta_low": low - base_tvar,
                "delta_high": high - base_tvar,
                "swing": max(abs(low - base_tvar), abs(high - base_tvar)),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)

    _apply_institutional_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    y = np.arange(len(result_df))
    low = result_df["delta_low"].to_numpy()
    high = result_df["delta_high"].to_numpy()

    ax.barh(y, low, color=_NEUTRAL_GREY, alpha=0.75, label="Low")
    ax.barh(y, high, color=_INSTITUTIONAL_BLUE, alpha=0.85, label="High")
    ax.axvline(0.0, color="#1F2A33", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(result_df["parameter"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Δ TVaR(99%) vs Base")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_currency_millions))
    ax.legend(loc="lower right")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    ax.text(
        0.99,
        0.02,
        f"Base TVaR(99%): {_format_currency_millions(base_tvar, 0)}  |  Sensitivity years/run: {n_years:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=_NEUTRAL_GREY,
        fontsize=10,
    )

    fig.tight_layout()
    return result_df, fig


def save_figure(fig: Figure, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")

