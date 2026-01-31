from ai_risk_pricing.scenario.schema import (
    Scenario,
    EventType,
    PropagationVector,
    SeverityDistribution,
)

known_knowns_scenarios = [
    Scenario(
        name="Claude AI Hallucinated Legal Citation in Anthropic Case",
        event_type=EventType.ALIGNMENT_FAILURE,
        trigger=(
            "Defense counsel relied on Claude AI–generated legal citations "
            "that contained fabricated titles and authors, which were not "
            "caught during manual review and submitted to court."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=["ai_provider", "legal_system", "enterprise"],
        base_frequency=0.15,  # rare but increasing AI-assisted legal errors
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.5, "sigma": 1.2},
        ),
        tail_multiplier=1.5,
        capability_threshold=0.6,
        threshold_multiplier=2.0,
        metadata={
            "id": 2,
            "published_date": "2025-05-19",
            "source": "TechCrunch",
            "url": "https://techcrunch.com/2025/05/15/anthropics-lawyer-was-forced-to-apologize-after-claude-hallucinated-a-legal-citation/",
            "external_context": "Anthropic v. music publishers",
            "failure_mode": "hallucinated legal citation",
        },
    ),

    Scenario(
        name="Las Vegas Autonomous Shuttle Collision",
        event_type=EventType.SYSTEMIC_FAILURE,
        trigger=(
            "Autonomous public shuttle failed to correctly respond to "
            "a human-driven delivery truck during live urban operation."
        ),
        propagation_vector=PropagationVector.SUPPLY_CHAIN,
        affected_nodes=["autonomous_vehicle", "public_transport", "city_infrastructure"],
        base_frequency=0.05,  # low-frequency but well-documented AV incidents
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 1.8, "scale": 5.0},
        ),
        tail_multiplier=2.5,
        capability_threshold=0.7,
        threshold_multiplier=3.0,
        metadata={
            "id": 3,
            "incident_id": "Incident 23",
            "date": "2017-11-08",
            "deployer": ["Navya", "Keolis North America"],
            "harm": ["bus passengers"],
            "source": "incidentdatabase.ai",
        },
    ),

    Scenario(
        name="Uber Autonomous Vehicle Fatal Pedestrian Collision",
        event_type=EventType.DARK_SCENARIO,
        trigger=(
            "Autonomous vehicle operating in self-driving mode failed to "
            "detect and appropriately respond to a pedestrian crossing."
        ),
        propagation_vector=PropagationVector.REGULATORY_CASCADE,
        affected_nodes=["autonomous_vehicle", "ridesharing_platform", "regulators"],
        base_frequency=0.01,  # extreme, low-frequency fatal outcome
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 1.3, "scale": 50.0},
        ),
        tail_multiplier=5.0,
        capability_threshold=0.65,
        threshold_multiplier=4.0,
        metadata={
            "id": 4,
            "incident_id": "Incident 4",
            "date": "2018-03-18",
            "victim": "Elaine Herzberg",
            "deployer": "Uber",
            "harm": ["fatality", "public trust erosion"],
            "source": "incidentdatabase.ai",
        },
    ),

    Scenario(
        name="YouTube Kids Algorithm Exposes Children to Inappropriate Content",
        event_type=EventType.MODEL_COLLAPSE,
        trigger=(
            "Content recommendation and filtering algorithms failed to "
            "reliably exclude disturbing and inappropriate videos from "
            "children-focused platforms."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=["content_platform", "children", "advertisers"],
        base_frequency=0.3,  # recurring issue in recommender systems
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 3.0, "sigma": 1.4},
        ),
        tail_multiplier=2.0,
        capability_threshold=0.55,
        threshold_multiplier=2.5,
        metadata={
            "id": 5,
            "incident_id": "Incident 1",
            "date": "2015-05-19",
            "platform": "YouTube Kids",
            "harm": ["children exposure", "brand damage"],
            "source": "incidentdatabase.ai",
        },
    ),

]
