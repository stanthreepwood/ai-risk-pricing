from ai_risk_pricing.scenario.known_known.kk_dummy_llm import known_knowns_scenarios
from ai_risk_pricing.scenario.known_known.kk_stat_parameter_estimation import create_aggregated_scenarios
from ai_risk_pricing.scenario.known_unk.ku_dummy_llm import ku_dummy_llm_scenarios
import numpy as np

from .schema import (
    Scenario,
    EventType,
    PropagationVector,
    SeverityDistribution,
)


class ScenarioGenerator:
    """
    Generator for AI catastrophe scenarios.
    
    Supports two modes of operation:
    1. Random generation: Creates scenarios with randomized parameters
       within plausible bounds for sensitivity analysis.
    2. Predefined scenarios: Returns expert-calibrated scenarios
       representing specific frontier AI failure modes.
    
    The predefined scenarios represent our best estimate of plausible
    AI catastrophe events given current technology trajectories.
    """
    
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """
        Initialize the scenario generator.
        
        Args:
            rng: NumPy random generator for reproducibility.
        """
        self.rng = rng or np.random.default_rng()
    
    def generate_random(self, n_scenarios: int = 5) -> list[Scenario]:
        """
        Generate random scenarios with plausible parameters.
        
        Useful for sensitivity analysis and exploring parameter space.
        Parameters are sampled from reasonable ranges based on expert
        judgment about AI risk characteristics.
        
        Args:
            n_scenarios: Number of scenarios to generate.
        
        Returns:
            List of randomly parameterized scenarios.
        """
        scenarios = []
        event_types = list(EventType)
        # Exclude dark scenario from random generation
        event_types = [et for et in event_types if et != EventType.DARK_SCENARIO]
        prop_vectors = list(PropagationVector)
        
        for i in range(n_scenarios):
            event_type = self.rng.choice(event_types)
            
            # Sample frequency: typically rare events (0.05 - 0.5 per year)
            base_frequency = float(self.rng.uniform(0.05, 0.5))
            
            # Sample severity distribution
            if self.rng.random() > 0.5:
                dist = SeverityDistribution(
                    name="pareto",
                    params={
                        "alpha": float(self.rng.uniform(1.2, 2.5)),
                        "scale": float(self.rng.uniform(5, 50)),
                    },
                )
            else:
                dist = SeverityDistribution(
                    name="lognormal",
                    params={
                        "mu": float(self.rng.uniform(3.0, 5.0)),
                        "sigma": float(self.rng.uniform(1.0, 2.5)),
                    },
                )
            
            # Sample affected nodes
            node_types = ["foundation_model", "saas_provider", "enterprise"]
            n_affected = self.rng.integers(1, len(node_types) + 1)
            affected = list(self.rng.choice(node_types, size=n_affected, replace=False))
            
            scenario = Scenario(
                name=f"random_scenario_{i+1}",
                event_type=event_type,
                trigger=f"Randomly generated trigger for {event_type.value}",
                propagation_vector=self.rng.choice(prop_vectors),
                affected_nodes=affected,
                base_frequency=base_frequency,
                severity_distribution=dist,
                tail_multiplier=float(self.rng.uniform(1.0, 2.0)),
                capability_threshold=float(self.rng.uniform(0.5, 0.9)),
                threshold_multiplier=float(self.rng.uniform(2.0, 5.0)),
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def get_predefined_scenarios(self) -> list[Scenario]:
        """
        Return expert-calibrated frontier AI catastrophe scenarios.
        
        These scenarios represent our current best estimates for plausible
        AI failure modes. Parameters are calibrated through:
        - Expert elicitation workshops
        - Analogies to historical technology failures
        - Analysis of AI system architectures and dependencies
        
        Returns:
            List of predefined catastrophe scenarios.
        """
        return [
            self._scenario_foundation_model_failure(),
            self._scenario_adversarial_cascade(),
            self._scenario_alignment_incident(),
            self._scenario_regulatory_shutdown(),
            self._scenario_supply_chain_compromise(),
        ]
    
    def get_dark_scenario(self) -> Scenario:
        """
        Return an extreme tail scenario for stress testing.
        
        The dark scenario represents a low-probability, civilization-scale
        AI failure event. It is deliberately calibrated to produce losses
        far exceeding the standard distribution to test portfolio resilience
        against tail risks.
        
        This scenario should be analyzed separately from the main
        distribution. It represents a "what-if" stress test rather
        than a probability-weighted expected outcome.
        """
        return Scenario(
            name="Dark Scenario: Cascading AI Collapse",
            event_type=EventType.DARK_SCENARIO,
            trigger=(
                "Correlated failure across multiple frontier AI systems "
                "due to shared vulnerability in foundation architecture, "
                "triggering global economic disruption."
            ),
            propagation_vector=PropagationVector.SUPPLY_CHAIN,
            affected_nodes=["foundation_model", "saas_provider", "enterprise"],
            base_frequency=0.001,  # 1 in 1000 year event
            severity_distribution=SeverityDistribution(
                name="pareto",
                params={"alpha": 1.1, "scale": 500.0},  # Extremely heavy tail
            ),
            tail_multiplier=5.0,
            capability_threshold=0.5,
            threshold_multiplier=10.0,
            known_mitigations={
                "gateway": 0.3,
                "monitoring": 0.2,
                "model_governance": 0.2,
                "evals": 0.2,
            },
            metadata={
                "description": "Extreme tail stress test scenario",
                "return_period_target": 1000,
                "calibration_basis": "Expert judgment - no historical analogue",
            },
        )
    
    def _scenario_foundation_model_failure(self) -> Scenario:
        """
        Scenario: Major foundation model experiences critical failure.
        
        A leading foundation model (e.g., GPT-class, Claude-class) experiences
        a catastrophic failure affecting all downstream applications. Could be
        caused by infrastructure failure, model collapse, or critical bug.
        """
        return Scenario(
            name="Foundation Model Critical Failure",
            event_type=EventType.MODEL_COLLAPSE,
            trigger=(
                "Critical infrastructure failure or model collapse in a major "
                "foundation model, causing service outage across all dependent "
                "applications and enterprises."
            ),
            propagation_vector=PropagationVector.MODEL_DEPENDENCY,
            affected_nodes=["foundation_model", "saas_provider", "enterprise"],
            base_frequency=0.15,  # ~1 event every 6-7 years
            severity_distribution=SeverityDistribution(
                name="pareto",
                params={"alpha": 1.5, "scale": 50.0},
            ),
            tail_multiplier=1.5,
            capability_threshold=0.8,
            threshold_multiplier=3.0,
            known_mitigations={
                "gateway": 0.6,
                "monitoring": 0.4,
                "model_governance": 0.3,
            },
            metadata={
                "historical_analogue": "Major cloud provider outages",
                "concentration_concern": "High - few providers dominate",
            },
        )
    
    def _scenario_adversarial_cascade(self) -> Scenario:
        """
        Scenario: Coordinated adversarial attack on AI systems.
        
        Sophisticated adversarial attack exploits shared vulnerabilities
        across multiple AI systems, potentially including prompt injection,
        model extraction, or coordinated manipulation.
        """
        return Scenario(
            name="Coordinated Adversarial Attack",
            event_type=EventType.ADVERSARIAL_ATTACK,
            trigger=(
                "Nation-state or sophisticated threat actor launches coordinated "
                "adversarial attack exploiting shared vulnerabilities in AI systems."
            ),
            propagation_vector=PropagationVector.API_DEPENDENCY,
            affected_nodes=["saas_provider", "enterprise"],
            base_frequency=0.25,  # ~1 event every 4 years
            severity_distribution=SeverityDistribution(
                name="lognormal",
                params={"mu": 4.0, "sigma": 1.8},
            ),
            tail_multiplier=2.0,
            capability_threshold=0.7,
            threshold_multiplier=2.5,
            known_mitigations={
                "prompt_injection": 0.7,
                "gateway": 0.5,
                "monitoring": 0.3,
                "access_control": 0.4,
            },
            metadata={
                "historical_analogue": "NotPetya, SolarWinds",
                "trend": "Increasing with AI capability growth",
            },
        )
    
    def _scenario_alignment_incident(self) -> Scenario:
        """
        Scenario: AI system produces harmful outputs due to misalignment.
        
        A major AI system exhibits harmful behavior due to alignment failure,
        causing direct harm to users or downstream systems. Could include
        deceptive behavior, manipulation, or unsafe recommendations.
        """
        return Scenario(
            name="Alignment Failure Incident",
            event_type=EventType.ALIGNMENT_FAILURE,
            trigger=(
                "Frontier AI system exhibits harmful misaligned behavior, "
                "causing direct user harm, reputational damage, and regulatory "
                "response across the AI industry."
            ),
            propagation_vector=PropagationVector.MARKET_CONTAGION,
            affected_nodes=["foundation_model", "saas_provider"],
            base_frequency=0.08,  # ~1 event every 12 years
            severity_distribution=SeverityDistribution(
                name="pareto",
                params={"alpha": 1.3, "scale": 100.0},  # Very heavy tail
            ),
            tail_multiplier=3.0,
            capability_threshold=0.6,
            threshold_multiplier=5.0,  # High multiplier - more capable = more risk
            known_mitigations={
                # Output filtering catches harmful outputs
                "output_filtering": 0.6,
                # Evals can detect misalignment pre-deployment
                "evals": 0.5,
                # Monitoring enables detection and shutdown
                "monitoring": 0.4,
                # Model governance for rollback
                "model_governance": 0.3,
            },
            metadata={
                "no_historical_analogue": True,
                "tail_concern": "Extreme - potential for catastrophic harm",
            },
        )
    
    def _scenario_regulatory_shutdown(self) -> Scenario:
        """
        Scenario: Emergency regulatory action halts AI operations.
        
        Major jurisdiction implements emergency AI regulations requiring
        immediate compliance or shutdown. Could be triggered by safety
        incident, geopolitical event, or precautionary action.
        """
        return Scenario(
            name="Emergency Regulatory Shutdown",
            event_type=EventType.REGULATORY_SHOCK,
            trigger=(
                "Major jurisdiction (US, EU, or China) implements emergency "
                "AI regulations requiring immediate operational changes or "
                "service suspension."
            ),
            propagation_vector=PropagationVector.REGULATORY_CASCADE,
            affected_nodes=["foundation_model", "saas_provider", "enterprise"],
            base_frequency=0.20,  # ~1 event every 5 years
            severity_distribution=SeverityDistribution(
                name="lognormal",
                params={"mu": 3.5, "sigma": 1.5},
            ),
            tail_multiplier=1.2,
            capability_threshold=0.9,
            threshold_multiplier=2.0,
            known_mitigations={
                "model_governance": 0.5,
                "evals": 0.4,
                "monitoring": 0.3,
                "access_control": 0.3,
            },
            metadata={
                "historical_analogue": "GDPR implementation, China algorithm rules",
                "trend": "Increasing regulatory attention globally",
            },
        )
    
    def _scenario_supply_chain_compromise(self) -> Scenario:
        """
        Scenario: AI training pipeline or model supply chain compromised.
        
        Malicious actor compromises AI training data, model weights, or
        deployment pipeline. Could remain undetected for extended period
        before triggering widespread failures.
        """
        return Scenario(
            name="AI Supply Chain Compromise",
            event_type=EventType.DATA_POISONING,
            trigger=(
                "Discovery that training data, model weights, or deployment "
                "pipeline for widely-used AI system has been compromised, "
                "requiring extensive remediation across dependent systems."
            ),
            propagation_vector=PropagationVector.SUPPLY_CHAIN,
            affected_nodes=["foundation_model", "saas_provider", "enterprise"],
            base_frequency=0.10,  # ~1 event every 10 years
            severity_distribution=SeverityDistribution(
                name="pareto",
                params={"alpha": 1.4, "scale": 75.0},
            ),
            tail_multiplier=2.5,
            capability_threshold=0.7,
            threshold_multiplier=3.5,
            known_mitigations={
                "data_quality": 0.8,
                "evals": 0.4,
                "model_governance": 0.5,
                "monitoring": 0.3,
            },
            metadata={
                "historical_analogue": "SolarWinds, XZ Utils",
                "detection_lag": "Potentially months to years",
            },
        )
    
    def get_all_scenarios(self, include_known_knowns: bool = True, include_known_unknowns: bool = True, include_dark: bool = True) -> list[Scenario]:
        scenarios = self.get_predefined_scenarios()
        if include_dark:
            scenarios.append(self.get_dark_scenario())
        if include_known_unknowns:
            scenarios.extend(ku_dummy_llm_scenarios)
        if include_known_knowns:
            scenarios.extend(create_aggregated_scenarios(known_knowns_scenarios))
        return scenarios
