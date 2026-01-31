
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ModelConfig, DEFAULT_CONFIG
from .scenario import ScenarioGenerator
from .modeling import MonteCarloEngine, DependencyGraph
from .modeling.monte_carlo import SimulationResult
from .portfolio import Company, Portfolio, PortfolioAggregator
from .pricing import RiskMetrics, PremiumCalculator
from .visualization import (
    ExceedanceCurve,
    compute_concentration_index,
    export_gexf,
    export_graphml,
    export_png,
    plot_aep,
    plot_oep,
    plot_return_period,
    plot_ruin_probability,
    compute_tvar_contributions,
    plot_sensitivity_tornado,
    save_figure,
)


def build_sample_portfolio(n_companies: int = 15, seed: int = 42) -> Portfolio:

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
        print(f"    - {s.name} (lambda={s.base_frequency:.3f}, {s.severity_distribution.name})")
    
    return scenarios


def build_dependency_graph(portfolio: Portfolio) -> DependencyGraph:
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
) -> SimulationResult:
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
    
    return result


def calculate_risk_metrics(ylt: pd.DataFrame, total_exposure: float) -> RiskMetrics:
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
    
    portfolio = build_sample_portfolio(n_companies=n_companies, seed=seed)
    
    scenarios = generate_scenarios(include_dark=dark_mode, seed=seed)
    
    graph = build_dependency_graph(portfolio)

    systemic_risk_score = compute_concentration_index(graph.graph)
    print(f"\nSystemic Risk Score (H): {systemic_risk_score:.6f}")

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        graph_dir = output_path / "graph"
        export_graphml(graph.graph, graph_dir / "dependency_graph.graphml")
        export_gexf(graph.graph, graph_dir / "dependency_graph.gexf")
        export_png(graph.graph, graph_dir / "dependency_graph.png")
        print(f"  Exported dependency graph to: {graph_dir}")
    
    sim_result = run_simulation(
        scenarios=scenarios,
        graph=graph,
        config=config,
        n_years=n_years,
    )
    ylt = sim_result.year_loss_table
    
    total_exposure = portfolio.total_exposure
    metrics = calculate_risk_metrics(ylt, total_exposure)
    
    calculate_premium(metrics, total_exposure, config)
    
    plot_exceedance_curve(ylt, output_dir=output_path, show=show_plot)

    print("\nGenerating actuarial reporting plots...")
    losses = ylt["loss"].to_numpy(dtype=float)

    # OEP approximation: max scenario contribution per year (occurrence proxy)
    oep_losses = losses
    if sim_result.scenario_losses:
        scenario_matrix = np.vstack([arr for arr in sim_result.scenario_losses.values()])
        if scenario_matrix.size > 0:
            oep_losses = scenario_matrix.max(axis=0)

    fig_aep = plot_aep(losses)
    fig_oep = plot_oep(oep_losses)
    fig_rp = plot_return_period(losses)

    # Capital adequacy grid anchored to tail losses
    cap_max = float(np.quantile(losses, 0.999) * 1.25) if np.any(losses > 0) else 1.0
    capital_range = np.linspace(0.0, max(1.0, cap_max), 60)
    fig_ruin = plot_ruin_probability(losses, capital_range)

    # TVaR contribution by scenario
    if sim_result.scenario_losses:
        loss_df = pd.DataFrame({"year": ylt["year"]})
        for scenario_name, arr in sim_result.scenario_losses.items():
            loss_df[scenario_name] = arr
        loss_long = loss_df.melt(id_vars=["year"], var_name="scenario", value_name="loss")
        _contrib_df, fig_tvar = compute_tvar_contributions(loss_long)
    else:
        fig_tvar = None
        print("  Skipping TVaR contribution plot (scenario breakdown unavailable).")

    #sensitivity tornado
    sensitivity_years = min(n_years, 25_000)
    tornado_df, fig_tornado = plot_sensitivity_tornado(
        scenarios=scenarios,
        dependency_graph=graph,
        config=config,
        n_years=sensitivity_years,
        seed=seed,
    )

    if output_path:
        plots_dir = output_path / "actuarial_plots"
        save_figure(fig_aep, plots_dir / "aep.png")
        save_figure(fig_oep, plots_dir / "oep.png")
        save_figure(fig_rp, plots_dir / "return_period.png")
        save_figure(fig_ruin, plots_dir / "capital_adequacy_ruin_probability.png")
        if fig_tvar is not None:
            save_figure(fig_tvar, plots_dir / "tvar_contributions.png")
        save_figure(fig_tornado, plots_dir / "tvar_sensitivity_tornado.png")
        tornado_df.to_csv(plots_dir / "tvar_sensitivity_tornado.csv", index=False)
        print(f"  Saved actuarial plots to: {plots_dir}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig_aep)
        plt.close(fig_oep)
        plt.close(fig_rp)
        plt.close(fig_ruin)
        if fig_tvar is not None:
            plt.close(fig_tvar)
        plt.close(fig_tornado)
    
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
        "--show-plots",
        action="store_true",
        default=False,
        help="Show interactive plot display",
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
        show_plot=args.show_plots,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
