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

llm_dsa_dummy_scenarios = [
    Scenario(
        name="Recommendation System Exposes Minors to Harmful Content",
        event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
        trigger=(
            "A content recommendation algorithm lacks adequate age-sensitive safeguards, "
            "leading to repeated exposure of minors to self-harm, sexual, or addictive content."
        ),
        propagation_vector=PropagationVector.SUPPLY_CHAIN,
        affected_nodes=[
            NodeLayer.SaaS_PROVIDER.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.35,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.4, "sigma": 1.0},
        ),
        legal_frameworks={
            "DSA": "medium",
            "ChildProtectionLaw": "high",
        },
        exclusive_to_sectors=["Online platforms", "Content services"],
        tail_multiplier=1.4,
        capability_threshold=0.6,
        metadata={
            "dsa_articles": ["Art. 28"],
            "protected_group": "minors",
            "failure_mode": "lack of age-appropriate design",
        },
    ),
    Scenario(
        name="Illegal Content Persists Due to Ineffective Notice-and-Action Mechanism",
        event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
        trigger=(
            "User reports of illegal content are not processed in a timely or transparent manner, "
            "allowing unlawful material to remain accessible."
        ),
        propagation_vector=PropagationVector.SUPPLY_CHAIN,
        affected_nodes=[
            NodeLayer.SaaS_PROVIDER.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.30,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.2, "sigma": 0.9},
        ),
        legal_frameworks={
            "DSA": "medium",
        },
        exclusive_to_sectors=["Online platforms", "Marketplaces"],
        tail_multiplier=1.3,
        capability_threshold=0.5,
        metadata={
            "dsa_articles": ["Art. 16"],
            "failure_mode": "ineffective content moderation process",
        },
    ),
Scenario(
    name="Users Not Informed About Content Moderation Decisions",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "Users whose content is removed or deprioritized are not provided "
        "with clear reasons or appeal mechanisms."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.40,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 1.9, "sigma": 0.8},
    ),
    legal_frameworks={
        "DSA": "low-medium",
    },
    exclusive_to_sectors=["Online platforms"],
    tail_multiplier=1.2,
    capability_threshold=0.5,
    metadata={
        "dsa_articles": ["Art. 17", "Art. 20"],
        "failure_mode": "opaque moderation decisions",
    },
),
Scenario(
    name="Dark Patterns Manipulate Minors Into Extended Platform Use",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "Interface design nudges minors into prolonged engagement or unwanted purchases "
        "through manipulative choice architectures."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.SaaS_PROVIDER.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.33,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 2.5, "sigma": 1.1},
    ),
    legal_frameworks={
        "DSA": "medium",
        "ConsumerProtectionLaw": "medium",
    },
    exclusive_to_sectors=["Online platforms", "Gaming"],
    tail_multiplier=1.5,
    capability_threshold=0.55,
    metadata={
        "dsa_articles": ["Art. 25", "Art. 28"],
        "failure_mode": "dark patterns affecting minors",
    },
),
Scenario(
    name="Large Platform Algorithm Amplifies Harmful Content at Societal Scale",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "A large-scale recommender system systematically promotes polarizing or harmful content, "
        "creating widespread societal harm beyond individual user impacts."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.SaaS_PROVIDER.value,
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.18,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 3.6, "sigma": 1.4},
    ),
    legal_frameworks={
        "DSA": "very_high",
    },
    exclusive_to_sectors=["Very Large Online Platforms"],
    tail_multiplier=3.0,
    capability_threshold=0.8,
    threshold_multiplier=2.5,
    metadata={
        "dsa_articles": ["Art. 34", "Art. 35"],
        "systemic_risk_type": "algorithmic amplification",
    },
),
Scenario(
    name="VLOP Fails to Identify and Mitigate Systemic Risks",
    event_type=EventType.FUNDAMENTAL_RIGHTS_VIOLATION,
    trigger=(
        "A very large online platform performs superficial or incomplete systemic risk assessments, "
        "missing foreseeable harms linked to recommender systems."
    ),
    propagation_vector=PropagationVector.SUPPLY_CHAIN,
    affected_nodes=[
        NodeLayer.ENTERPRISE.value,
    ],
    base_frequency=0.20,
    severity_distribution=SeverityDistribution(
        name="lognormal",
        params={"mu": 3.2, "sigma": 1.2},
    ),
    legal_frameworks={
        "DSA": "very_high",
    },
    exclusive_to_sectors=["Very Large Online Platforms"],
    tail_multiplier=2.5,
    capability_threshold=0.75,
    metadata={
        "dsa_articles": ["Art. 34"],
        "failure_mode": "inadequate systemic risk assessment",
    },
),
]