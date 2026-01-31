"""
AI Catastrophe Model - Main Entry Point.

Executes a complete catastrophe model run:
1. Build a sample portfolio of insured companies
2. Generate catastrophe scenarios (predefined frontier AI scenarios)
3. Construct the AI supply chain dependency graph
4. Run Monte Carlo simulation (100,000 years)
5. Calculate risk metrics (EL, VaR, TVaR)
6. Compute technical premium with ambiguity loading
7. Generate exceedance probability curve

This script demonstrates a miniature reinsurance-style catastrophe model
for AI risks, producing outputs similar to those used in actual cat modeling.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, DEFAULT_CONFIG
from .scenario import ScenarioGenerator
from .modeling import MonteCarloEngine, DependencyGraph
from .portfolio import Company, Portfolio, PortfolioAggregator
from .pricing import RiskMetrics, PremiumCalculator
from .visualization import ExceedanceCurve


def build_sample_portfolio(n_companies: int = 15, seed: int = 42) -> Portfolio:
    """
    Build a diverse sample portfolio of companies with AI exposure.
    
    Creates a realistic portfolio with varying company sizes, sectors,
    and AI risk characteristics for demonstration.
    
    Args:
        n_companies: Number of companies to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        Populated Portfolio instance.
    """
    print(f"Building portfolio with {n_companies} companies...")
    
    portfolio = Portfolio.build_sample_portfolio(n_companies=n_companies, seed=seed)
    
    summary = portfolio.summary()
    print(f"  Total revenue: ${summary['total_revenue_M']:,.0f}M")
    print(f"  Total exposure: ${summary['total_exposure_M']:,.0f}M")
    print(f"  Average AI dependency: {summary['average_ai_dependency']:.1%}")
    print(f"  Sectors: {', '.join(summary['sectors'])}")
    
    return portfolio


def generate_scenarios(include_dark: bool = False, seed: int = 42) -> list:
    """
    Generate catastrophe scenarios for simulation.
    
    Uses predefined frontier AI scenarios calibrated through expert
    judgment. Optionally includes a dark (extreme tail) scenario
    for stress testing.
    
    Args:
        include_dark: Whether to include the extreme tail scenario.
        seed: Random seed for reproducibility.
    
    Returns:
        List of Scenario objects ready for simulation.
    """
    print("\nGenerating catastrophe scenarios...")
    
    generator = ScenarioGenerator(rng=np.random.default_rng(seed))
    scenarios = generator.get_all_scenarios(include_known_knowns=True, include_dark=include_dark)
    
    print(f"  Generated {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"    - {s.name} (λ={s.base_frequency:.3f}, {s.severity_distribution.name})")
    
    return scenarios


def build_dependency_graph(portfolio: Portfolio) -> DependencyGraph:
    """
    Build the AI supply chain dependency graph.
    
    Creates a three-tier graph structure:
    - Foundation models (upstream)
    - SaaS providers (middle tier)
    - Enterprises from portfolio (downstream)
    
    Args:
        portfolio: Portfolio to incorporate into graph.
    
    Returns:
        Populated DependencyGraph instance.
    """
    print("\nBuilding dependency graph...")
    
    aggregator = PortfolioAggregator(portfolio)
    graph = aggregator.build_dependency_graph_from_portfolio(
        n_foundation_models=2,
        n_saas_providers=4,
    )
    
    concentration = graph.calculate_concentration_index()
    print(f"  Nodes: {len(graph._nodes)}")
    print(f"  Total exposure: ${graph.total_exposure():,.0f}M")
    print(f"  Concentration index (HHI): {concentration:.3f}")
    
    return graph


def run_simulation(
    scenarios: list,
    graph: DependencyGraph,
    config: ModelConfig,
    n_years: int = 100_000,
) -> pd.DataFrame:
    """
    Execute Monte Carlo simulation.
    
    Runs the core simulation engine to generate a Year Loss Table
    by simulating many years of potential losses.
    
    Args:
        scenarios: List of catastrophe scenarios.
        graph: Dependency graph for loss propagation.
        config: Model configuration.
        n_years: Number of simulation years.
    
    Returns:
        Year Loss Table DataFrame.
    """
    print(f"\nRunning Monte Carlo simulation ({n_years:,} years)...")
    
    start_time = time.time()
    
    engine = MonteCarloEngine(
        scenarios=scenarios,
        dependency_graph=graph,
        config=config,
        seed=config.random_seed,
    )
    
    result = engine.simulate_years_vectorized(n_years=n_years)
    
    elapsed = time.time() - start_time
    
    print(f"  Simulation complete in {elapsed:.1f} seconds")
    print(f"  Years with loss: {result.metadata['years_with_loss']:,} ({result.metadata['years_with_loss']/n_years:.1%})")
    print(f"  Mean annual loss: ${result.metadata['mean_annual_loss']:,.2f}M")
    print(f"  Maximum loss: ${result.metadata['max_annual_loss']:,.2f}M")
    
    return result.year_loss_table


def calculate_risk_metrics(ylt: pd.DataFrame, total_exposure: float) -> RiskMetrics:
    """
    Calculate risk metrics from Year Loss Table.
    
    Computes standard actuarial risk measures including EL, VaR, and TVaR.
    
    Args:
        ylt: Year Loss Table from simulation.
        total_exposure: Total portfolio exposure.
    
    Returns:
        RiskMetrics instance with computed values.
    """
    print("\nCalculating risk metrics...")
    
    metrics = RiskMetrics(ylt)
    results = metrics.calculate_all(total_exposure)
    
    print(f"  Expected Loss (EL):    ${results.expected_loss:>12,.2f}M")
    print(f"  Standard Deviation:    ${results.std_dev:>12,.2f}M")
    print(f"  VaR 99%:               ${results.var_99:>12,.2f}M")
    print(f"  TVaR 99%:              ${results.tvar_99:>12,.2f}M")
    print(f"  VaR 99.5%:             ${results.var_995:>12,.2f}M")
    print(f"  TVaR 99.5%:            ${results.tvar_995:>12,.2f}M")
    print(f"  Maximum Loss:          ${results.max_loss:>12,.2f}M")
    
    return metrics


def calculate_premium(metrics: RiskMetrics, total_exposure: float, config: ModelConfig) -> None:
    """
    Calculate and display technical premium.
    
    Computes premium using the ambiguity-loaded pricing formula
    appropriate for AI catastrophe risks without historical data.
    
    Args:
        metrics: RiskMetrics instance with loss statistics.
        total_exposure: Total portfolio exposure.
        config: Model configuration with pricing parameters.
    """
    print("\nCalculating technical premium...")
    
    calculator = PremiumCalculator(
        risk_metrics=metrics,
        total_exposure=total_exposure,
        params=config.pricing,
    )
    
    # Print full premium report
    print(calculator.report())


def plot_exceedance_curve(
    ylt: pd.DataFrame,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """
    Generate and display exceedance probability curve.
    
    Creates a professional-quality EP curve visualization with
    key return period markers.
    
    Args:
        ylt: Year Loss Table from simulation.
        output_dir: Directory to save plot (optional).
        show: Whether to display the plot interactively.
    """
    print("\nGenerating exceedance curve...")
    
    curve = ExceedanceCurve(ylt, title="AI Catastrophe Risk - Exceedance Probability Curve")
    
    # Print return period table
    rp_table = curve.return_period_table()
    print("\nReturn Period Table:")
    print(rp_table.to_string(index=False))
    
    # Determine save path
    save_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / "exceedance_curve.png")
    
    # Generate plot
    curve.plot(
        figsize=(11, 8),
        show_return_periods=True,
        return_periods=[10, 50, 100, 200],
        save_path=save_path,
        show=show,
    )
    
    if save_path:
        print(f"\n  Saved plot to: {save_path}")


def main(
    n_companies: int = 15,
    n_years: int = 100_000,
    dark_mode: bool = False,
    output_dir: str | None = None,
    show_plot: bool = True,
    seed: int = 42,
) -> None:
    """
    Execute complete AI catastrophe model run.
    
    This is the main entry point that orchestrates the full model workflow.
    
    Args:
        n_companies: Number of companies in portfolio.
        n_years: Number of simulation years.
        dark_mode: Whether to include extreme tail scenario.
        output_dir: Directory for output files (optional).
        show_plot: Whether to display plots interactively.
        seed: Random seed for reproducibility.
    """
    print("=" * 70)
    print("AI CATASTROPHE MODEL")
    print("Reinsurance-Style Risk Pricing Engine")
    print("=" * 70)
    
    # Initialize configuration
    config = DEFAULT_CONFIG
    if seed is not None:
        config = ModelConfig(
            simulation_years=n_years,
            random_seed=seed,
            frequency=config.frequency,
            severity=config.severity,
            pricing=config.pricing,
            dependency=config.dependency,
            capability=config.capability,
            dark_scenario=config.dark_scenario,
        )
    
    if dark_mode:
        config = config.with_dark_mode(enabled=True)
        print("\n*** DARK SCENARIO MODE ENABLED ***")
        print("    Extreme tail events will be injected into simulation")
    
    # Step 1: Build portfolio
    portfolio = build_sample_portfolio(n_companies=n_companies, seed=seed)
    
    # Step 2: Generate scenarios
    scenarios = generate_scenarios(include_dark=dark_mode, seed=seed)
    
    # Step 3: Build dependency graph
    graph = build_dependency_graph(portfolio)
    
    # Step 4: Run simulation
    ylt = run_simulation(
        scenarios=scenarios,
        graph=graph,
        config=config,
        n_years=n_years,
    )
    
    # Step 5: Calculate risk metrics
    total_exposure = portfolio.total_exposure
    metrics = calculate_risk_metrics(ylt, total_exposure)
    
    # Step 6: Calculate premium
    calculate_premium(metrics, total_exposure, config)
    
    # Step 7: Plot exceedance curve
    output_path = Path(output_dir) if output_dir else None
    plot_exceedance_curve(ylt, output_dir=output_path, show=show_plot)
    
    print("\n" + "=" * 70)
    print("MODEL RUN COMPLETE")
    print("=" * 70)


def cli() -> None:
    """Command-line interface entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Catastrophe Risk Model - Premium Calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--companies",
        type=int,
        default=15,
        help="Number of companies in portfolio",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=100_000,
        help="Number of simulation years",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Enable dark scenario (extreme tail event injection)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive plot display",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    main(
        n_companies=args.companies,
        n_years=args.years,
        dark_mode=args.dark_mode,
        output_dir=args.output_dir,
        show_plot=not args.no_plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
