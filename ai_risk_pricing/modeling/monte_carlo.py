from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

from .frequency import FrequencyModel
from .severity import SeverityModel
from .dependency import DependencyGraph
from ..scenario.schema import Scenario
from ..config import ModelConfig, DEFAULT_CONFIG

if TYPE_CHECKING:
    from ai_risk_pricing.safety.measure import SafetyProfile
    from ai_risk_pricing.safety.mitigation import MitigationEngine


@dataclass
class SimulationResult:
    """
    Container for Monte Carlo simulation results.
    """
    
    year_loss_table: pd.DataFrame
    scenario_losses: dict[str, np.ndarray] | None = None
    metadata: dict | None = None


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for AI catastrophe loss modeling.
    
    The engine simulates multiple years of potential losses by:
    1. Sampling event occurrences from frequency distributions
    2. Sampling event severities from heavy-tailed distributions
    3. Propagating losses through the dependency graph
    4. Aggregating annual portfolio losses
    
    Monte Carlo simulation is the standard method for catastrophe
    pricing when analytical solutions are intractable. By simulating
    many years (typically 10,000-100,000), we build an empirical
    distribution of annual losses that can be used for pricing
    and capital calculations.
    
    The resulting Year Loss Table (YLT) is the fundamental output of
    catastrophe models, showing the distribution of possible annual losses.
    """
    
    def __init__(
        self,
        scenarios: Sequence[Scenario],
        dependency_graph: DependencyGraph,
        config: ModelConfig = DEFAULT_CONFIG,
        seed: int | None = None,
        mitigation_engine: MitigationEngine | None = None,
        safety_profile: SafetyProfile | None = None,
    ) -> None:
        """
        Initialize the Monte Carlo simulation engine.
        
        Args:
            scenarios: List of catastrophe scenarios to simulate.
            dependency_graph: Graph defining loss propagation structure.
            config: Model configuration parameters.
            seed: Random seed for reproducibility.
            mitigation_engine: Optional engine for computing safety mitigation.
                When provided along with safety_profile, applies mitigation
                factors to severity samples.
            safety_profile: Optional safety profile for mitigation calculations.
                Can be a portfolio-level profile or individual company profile.
        """
        self.scenarios = list(scenarios)
        self.dependency_graph = dependency_graph
        self.config = config
        
        self.rng = np.random.default_rng(seed or config.random_seed)
        
        self.frequency_model = FrequencyModel(rng=self.rng)
        self.severity_model = SeverityModel(rng=self.rng)
        
        self._scenario_lambdas = np.array([s.base_frequency for s in self.scenarios])
        
        # Mitigation components (optional)
        self.mitigation_engine = mitigation_engine
        self.safety_profile = safety_profile
        
        # Pre-compute mitigation factors for each scenario if engine is provided
        self._mitigation_factors: dict[str, float] = {}
        if self.mitigation_engine and self.safety_profile:
            for scenario in self.scenarios:
                factor = self.mitigation_engine.compute_mitigation_factor(
                    scenario, self.safety_profile
                )
                self._mitigation_factors[scenario.name] = factor
    
    def _apply_mitigation(
        self,
        scenario: Scenario,
        severities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply safety mitigation to severity samples.
        
        If a mitigation engine and safety profile are configured, reduces
        severities based on the scenario's known_mitigations and the
        profile's coverage of relevant risk surfaces.
        
        Args:
            scenario: The scenario being simulated.
            severities: Array of sampled severity values.
            
        Returns:
            Mitigated severity values (may be modified in place).
        """
        if not self._mitigation_factors:
            return severities
        
        factor = self._mitigation_factors.get(scenario.name, 1.0)
        if factor < 1.0:
            return severities * factor
        return severities
    
    def simulate_year(self) -> tuple[float, dict[str, float]]:
        """
        Simulate a single year of losses.
        
        Executes one iteration of the simulation loop:
        1. Sample event counts for each scenario
        2. Sample severities for events that occur
        3. Propagate losses through dependency graph
        4. Return total annual loss
        
        Returns:
            Tuple of (total_annual_loss, losses_by_scenario)
        """
        total_loss = 0.0
        scenario_losses: dict[str, float] = {}
        
        for scenario in self.scenarios:
            # Sample number of events
            n_events = self.frequency_model.sample_event_count(scenario.base_frequency)
            
            if n_events == 0:
                scenario_losses[scenario.name] = 0.0
                continue
            
            # sample severities for all events
            severities = self.severity_model.sample_full_scenario(
                dist_name=scenario.severity_distribution.name,
                params=scenario.severity_distribution.params,
                capability_score=self._get_capability_score(scenario),
                threshold=scenario.capability_threshold,
                threshold_multiplier=scenario.threshold_multiplier,
                tail_multiplier=scenario.tail_multiplier,
                size=n_events,
            )
            
            severities = self._apply_mitigation(scenario, severities)
            
            # propagate each event through dependency graph
            scenario_total = 0.0
            for severity in severities:
                root_node = self._get_root_node(scenario)
                
                propagated_loss = self.dependency_graph.total_propagated_loss(
                    root_node=root_node,
                    root_loss=severity,
                    base_propagation=self.config.dependency.base_propagation,
                    concentration_exponent=self.config.dependency.concentration_exponent,
                    max_amplification=self.config.dependency.max_amplification,
                )
                scenario_total += propagated_loss
            
            scenario_losses[scenario.name] = scenario_total
            total_loss += scenario_total
        
        return total_loss, scenario_losses
    
    def simulate_years(
        self,
        n_years: int | None = None,
        show_progress: bool = False,
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation for multiple years.
        
        This is the main entry point for simulation. Generates a complete
        Year Loss Table by simulating many independent years.
        
        Each simulated year represents one possible realization of
        annual losses. The collection of simulated years forms an
        empirical distribution that approximates the true (unknown)
        loss distribution.
        
        More simulation years → more stable estimates, especially
        for tail metrics like VaR 99.5%.
        """
        n_years = n_years or self.config.simulation_years
        
        # Pre-allocate arrays for efficiency
        annual_losses = np.zeros(n_years)
        scenario_losses_arr = {s.name: np.zeros(n_years) for s in self.scenarios}
        
        # Progress tracking
        progress_interval = max(1, n_years // 10)
        
        for year in range(n_years):
            if show_progress and year % progress_interval == 0:
                print(f"  Simulating year {year:,} / {n_years:,}")
            
            total_loss, s_losses = self.simulate_year()
            annual_losses[year] = total_loss
            
            for scenario_name, loss in s_losses.items():
                scenario_losses_arr[scenario_name][year] = loss
        
        # Build Year Loss Table
        ylt = pd.DataFrame({
            "year": np.arange(1, n_years + 1),
            "loss": annual_losses,
        })
        
        # Build metadata
        metadata = {
            "n_years": n_years,
            "n_scenarios": len(self.scenarios),
            "scenario_names": [s.name for s in self.scenarios],
            "total_exposure": self.dependency_graph.total_exposure(),
            "concentration_index": self.dependency_graph.calculate_concentration_index(),
            "mean_annual_loss": float(np.mean(annual_losses)),
            "max_annual_loss": float(np.max(annual_losses)),
            "years_with_loss": int(np.sum(annual_losses > 0)),
        }
        
        return SimulationResult(
            year_loss_table=ylt,
            scenario_losses=scenario_losses_arr,
            metadata=metadata,
        )
    
    def simulate_years_vectorized(
        self,
        n_years: int | None = None,
    ) -> SimulationResult:
        """
        Vectorized Monte Carlo simulation for improved performance.
        
        Uses NumPy vectorization to simulate all years simultaneously
        where possible. This is faster than the loop-based approach
        but uses more memory.
        
        Note: Dependency propagation still requires iteration, but
        frequency and severity sampling is fully vectorized.
        
        Args:
            n_years: Number of years to simulate.
        
        Returns:
            SimulationResult containing Year Loss Table and metadata.
        """
        n_years = n_years or self.config.simulation_years
        n_scenarios = len(self.scenarios)
        
        # Vectorized frequency sampling: (n_scenarios, n_years)
        event_counts = self.frequency_model.sample_multi_scenario_counts(
            lambdas=[s.base_frequency for s in self.scenarios],
            n_years=n_years,
        )
        
        # Pre-allocate loss arrays
        annual_losses = np.zeros(n_years)
        scenario_losses_arr = {s.name: np.zeros(n_years) for s in self.scenarios}
        
        # Process each scenario
        for i, scenario in enumerate(self.scenarios):
            scenario_year_losses = np.zeros(n_years)
            
            # Find years with events
            years_with_events = np.where(event_counts[i] > 0)[0]
            
            for year_idx in years_with_events:
                n_events = event_counts[i, year_idx]
                
                # Sample severities for all events in this year
                severities = self.severity_model.sample_full_scenario(
                    dist_name=scenario.severity_distribution.name,
                    params=scenario.severity_distribution.params,
                    capability_score=self._get_capability_score(scenario),
                    threshold=scenario.capability_threshold,
                    threshold_multiplier=scenario.threshold_multiplier,
                    tail_multiplier=scenario.tail_multiplier,
                    size=n_events,
                )
                
                # Apply safety mitigation
                severities = self._apply_mitigation(scenario, severities)
                
                # Propagate each event
                root_node = self._get_root_node(scenario)
                year_total = sum(
                    self.dependency_graph.total_propagated_loss(
                        root_node=root_node,
                        root_loss=sev,
                        base_propagation=self.config.dependency.base_propagation,
                        concentration_exponent=self.config.dependency.concentration_exponent,
                        max_amplification=self.config.dependency.max_amplification,
                    )
                    for sev in severities
                )
                
                scenario_year_losses[year_idx] = year_total
            
            scenario_losses_arr[scenario.name] = scenario_year_losses
            annual_losses += scenario_year_losses
        
        # year loss table
        ylt = pd.DataFrame({
            "year": np.arange(1, n_years + 1),
            "loss": annual_losses,
        })
        
        metadata = {
            "n_years": n_years,
            "n_scenarios": len(self.scenarios),
            "scenario_names": [s.name for s in self.scenarios],
            "total_exposure": self.dependency_graph.total_exposure(),
            "concentration_index": self.dependency_graph.calculate_concentration_index(),
            "mean_annual_loss": float(np.mean(annual_losses)),
            "max_annual_loss": float(np.max(annual_losses)),
            "years_with_loss": int(np.sum(annual_losses > 0)),
            "vectorized": True,
        }
        
        return SimulationResult(
            year_loss_table=ylt,
            scenario_losses=scenario_losses_arr,
            metadata=metadata,
        )
    
    def inject_dark_scenario(
        self,
        dark_scenario: Scenario,
        ylt: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inject dark scenario losses into existing Year Loss Table.
        
        Dark scenarios are extreme tail events that may not be captured
        by standard simulation. This method adds their contribution to
        an existing YLT.
        
        Dark scenarios are stress tests overlaid on the base
        distribution. They represent "what-if" extreme events
        that warrant separate capital consideration.
        """
        n_years = len(ylt)
        
        dark_events = self.frequency_model.sample_event_counts(
            dark_scenario.base_frequency,
            n_years,
        )
        
        dark_years = np.where(dark_events > 0)[0]
        
        for year_idx in dark_years:
            n_events = dark_events[year_idx]
            
            severities = self.severity_model.sample_full_scenario(
                dist_name=dark_scenario.severity_distribution.name,
                params=dark_scenario.severity_distribution.params,
                capability_score=self._get_capability_score(dark_scenario),
                threshold=dark_scenario.capability_threshold,
                threshold_multiplier=dark_scenario.threshold_multiplier,
                tail_multiplier=dark_scenario.tail_multiplier,
                size=n_events,
            )
            
            # Apply safety mitigation (compute dynamically for dark scenarios)
            if self.mitigation_engine and self.safety_profile:
                factor = self.mitigation_engine.compute_mitigation_factor(
                    dark_scenario, self.safety_profile
                )
                severities = severities * factor
            
            root_node = self._get_root_node(dark_scenario)
            dark_loss = sum(
                self.dependency_graph.total_propagated_loss(
                    root_node=root_node,
                    root_loss=sev,
                    base_propagation=self.config.dependency.base_propagation,
                    concentration_exponent=self.config.dependency.concentration_exponent,
                    max_amplification=self.config.dependency.max_amplification,
                )
                for sev in severities
            )
            
            ylt.loc[year_idx, "loss"] += dark_loss
        
        return ylt
    
    def _get_capability_score(self, scenario: Scenario) -> float:
        """
        Get capability score for a scenario.
        
        For now, returns a default based on scenario type.
        Future versions could use time-varying capability scores.
        """
        if scenario.is_dark_scenario:
            return 0.9
        elif scenario.event_type.value == "alignment_failure":
            return 0.75
        else:
            return 0.65
    
    def _get_root_node(self, scenario: Scenario) -> str:
        """
        Determine the root node for loss propagation based on scenario.
        
        Different scenario types originate at different points in the
        dependency graph.
        """
        affected = scenario.affected_nodes
        
        if "foundation_model" in affected:
            fm_nodes = self.dependency_graph.get_nodes_by_type("foundation_model")
            if fm_nodes:
                return max(fm_nodes, key=lambda n: n.criticality_score).name
        
        if "saas_provider" in affected:
            saas_nodes = self.dependency_graph.get_nodes_by_type("saas_provider")
            if saas_nodes:
                return max(saas_nodes, key=lambda n: n.criticality_score).name
        
        enterprise_nodes = self.dependency_graph.get_nodes_by_type("enterprise")
        if enterprise_nodes:
            return enterprise_nodes[0].name
        
        raise ValueError("No valid root node found in dependency graph")
