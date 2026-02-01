from ai_risk_pricing.scenario.schema import (
    NodeLayer,
    Scenario,
    EventType,
    PropagationVector,
    SeverityDistribution,
)

ku_dummy_llm_scenarios = [
    Scenario(
    name="Automated CV Screening System Discriminates Against Female Applicants",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "A foundation model trained on historically male-dominated hiring data "
        "systematically ranks female candidates lower for technical roles, "
        "leading to discriminatory hiring decisions at scale."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.FOUNDATION_MODEL.value,
        NodeLayer.SaaS_PROVIDER.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.28,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 2.8, "sigma": 1.1},
    ),
    legal_frameworks={
        "AIAct": "high",
        "EmploymentLaw": "high",
        "AntiDiscriminationLaw": "high",
    },
    exclusive_to_sectors=["Employment"],
    tail_multiplier=1.6,
    capability_threshold=0.65,
    metadata={
        "failure_mode": "gender bias in ranking",
        "protected_attribute": "sex",
        "decision_type": "hiring",
    },
    ),
    Scenario(
    name="Retail Credit Scoring Model Discriminates Against Ethnic Minorities",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "A credit scoring system proxies ethnicity through postcode and spending "
        "patterns, resulting in systematically higher rejection rates for minority applicants."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.SaaS_PROVIDER.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.22,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 3.0, "sigma": 1.0},
    ),
    legal_frameworks={
        "AIAct": "high",
        "ConsumerCreditLaw": "high",
        "GDPR": "medium",
    },
    exclusive_to_sectors=["Finance"],
    tail_multiplier=1.8,
    capability_threshold=0.7,
    metadata={
        "failure_mode": "proxy discrimination",
        "protected_attribute": "ethnicity",
        "decision_type": "credit approval",
    },
),
Scenario(
    name="Health Insurance Pricing Discriminates Against Disabled Individuals",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "An insurance pricing model uses wearable and prescription data that "
        "acts as a proxy for disability, leading to higher premiums or exclusions."
    ),
    propagation_vector=PropagationVector.MARKET_CONTAGION,
    affected_nodes=[
        NodeLayer.FOUNDATION_MODEL.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.18,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 3.2, "sigma": 1.3},
    ),
    legal_frameworks={
        "AIAct": "high",
        "InsuranceRegulation": "high",
        "DisabilityLaw": "high",
    },
    exclusive_to_sectors=["Insurance"],
    tail_multiplier=2.0,
    capability_threshold=0.75,
    metadata={
        "failure_mode": "health proxy bias",
        "protected_attribute": "disability",
        "decision_type": "pricing",
    },
),
Scenario(
    name="Automated Welfare Eligibility System Discriminates Against Single Parents",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "A public-sector eligibility model penalizes non-traditional household structures, "
        "leading to disproportionate benefit denials for single parents."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.SaaS_PROVIDER.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.25,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 3.1, "sigma": 1.2},
    ),
    legal_frameworks={
        "AIAct": "high",
        "AdministrativeLaw": "high",
        "HumanRightsLaw": "high",
    },
    exclusive_to_sectors=["Public sector"],
    tail_multiplier=1.9,
    capability_threshold=0.6,
    metadata={
        "failure_mode": "family status bias",
        "protected_attribute": "family status",
        "decision_type": "benefit eligibility",
    },
)

]
