from ai_risk_pricing.scenario.schema import (
    NodeLayer,
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
        affected_nodes=[NodeLayer.FOUNDATION_MODEL.value, NodeLayer.SaaS_PROVIDER.value, NodeLayer.ENTERPRISE.value],
        base_frequency=0.15,  # rare but increasing AI-assisted legal errors
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.5, "sigma": 1.2},
        ),
        severity_materialization=None,
        legal_frameworks={"ProductLiability": "low", "AIAct": "low"},
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
        affected_nodes=[NodeLayer.ISOLATED_NODE.value],
        base_frequency=0.05,  # low-frequency but well-documented AV incidents
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 1.8, "scale": 5.0},
        ),
        severity_materialization=2,
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
        affected_nodes=[NodeLayer.ISOLATED_NODE.value],
        base_frequency=0.01,  # extreme, low-frequency fatal outcome
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 1.3, "scale": 50.0},
        ),
        severity_materialization=10,
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
        affected_nodes=[NodeLayer.ISOLATED_NODE.value],
        base_frequency=0.3,  # recurring issue in recommender systems
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 3.0, "sigma": 1.4},
        ),
        tail_multiplier=2.0,
        severity_materialization=None,
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

    Scenario(
        name="Deepfake Videos Impersonating Yanis Varoufakis Spread on Social Media",
        event_type=EventType.ADVERSARIAL_ATTACK,
        trigger=(
            "Malicious actors generated and distributed deepfake videos "
            "impersonating Yanis Varoufakis, fabricating political statements "
            "and repeatedly re-uploading content after takedowns."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.SaaS_PROVIDER.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.35,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 3.0, "sigma": 1.4},
        ),
        tail_multiplier=2.0,
        severity_materialization=None,
        metadata={
            "id": 1148,
            "incident_id": "Incident 1331",
            "date": "2026-01-05",
            "victim": "Yanis Varoufakis",
            "harm": ["impersonation", "political misinformation", "epistemic erosion"],
            "source": "incidentdatabase.ai",
        },
    ),

    Scenario(
        name="National Weather Service Published AI-Generated Map With Fabricated Towns",
        event_type=EventType.ALIGNMENT_FAILURE,
        trigger=(
            "A generative AI tool used by the National Weather Service "
            "produced a public forecast map containing non-existent "
            "and misspelled town names."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.2,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.2, "sigma": 1.1},
        ),
        severity_materialization=None,
        metadata={
            "id": 1149,
            "incident_id": "Incident 1332",
            "date": "2026-01-03",
            "deployer": "National Weather Service",
            "harm": ["public misinformation", "loss of trust"],
            "domain": "public safety communications",
        },
    ),

    Scenario(
        name="AI-Generated Media Spread False Claims About Nicolás Maduro Capture",
        event_type=EventType.ADVERSARIAL_ATTACK,
        trigger=(
            "AI-generated images and videos depicting fabricated scenarios "
            "about Nicolás Maduro circulated widely following a real-world "
            "political event, reaching millions of viewers."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.SaaS_PROVIDER.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.4,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 3.4, "sigma": 1.5},
        ),
        tail_multiplier=2.5,
        severity_materialization=None,
        metadata={
            "id": 1150,
            "incident_id": "Incident 1333",
            "victim": "Nicolás Maduro",
            "reach_estimate": "14M+ views",
            "harm": ["disinformation", "political destabilization"],
            "platforms": ["X", "social media"],
        },
    ),

    Scenario(
        name="Grok Generated False Identity of ICE Agent Leading to Harassment",
        event_type=EventType.ALIGNMENT_FAILURE,
        trigger=(
            "Users prompted Grok to identify a masked ICE agent, resulting "
            "in fabricated images and a false name that spread online and "
            "triggered harassment of uninvolved individuals."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.SaaS_PROVIDER.value,
        ],
        base_frequency=0.25,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 3.1, "sigma": 1.3},
        ),
        severity_materialization=None,
        tail_multiplier=2.0,
        metadata={
            "id": 1151,
            "incident_id": "Incident 1334",
            "model": "Grok",
            "developer": "xAI",
            "harm": ["misidentification", "harassment", "reputational damage"],
        },
    ),

    Scenario(
        name="AI-Cloned Voice Scam Defrauded Indian Play School Owner",
        event_type=EventType.ADVERSARIAL_ATTACK,
        trigger=(
            "Scammers used AI-based voice cloning to impersonate a police "
            "employee, convincing a victim to transfer funds under false "
            "medical emergency claims."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.45,
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 2.0, "scale": 1.0},
        ),
        tail_multiplier=2.5,
        severity_materialization=None,
        metadata={
            "id": 1152,
            "incident_id": "Incident 1339",
            "loss_amount_usd": 1080,
            "region": "Indore, India",
            "harm": ["financial fraud", "impersonation"],
        },
    ),

    Scenario(
        name="ICE AI Resume Screening Misclassified Recruits",
        event_type=EventType.SYSTEMIC_FAILURE,
        trigger=(
            "An AI-assisted résumé screening system misclassified applicants "
            "as having law-enforcement experience, routing them into "
            "inadequate training pathways."
        ),
        propagation_vector=PropagationVector.REGULATORY_CASCADE,
        affected_nodes=[
            NodeLayer.SaaS_PROVIDER.value,
            NodeLayer.ENTERPRISE.value,
        ],
        base_frequency=0.15,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.8, "sigma": 1.2},
        ),
        severity_materialization=None,
        metadata={
            "id": 1153,
            "incident_id": "Incident 1343",
            "agency": "U.S. ICE",
            "harm": ["operational risk", "public safety exposure"],
        },
    ),

    Scenario(
        name="AI-Generated Images Falsely Depicted Kate Garraway With Fictional Partner",
        event_type=EventType.ADVERSARIAL_ATTACK,
        trigger=(
            "AI-generated images falsely portraying a public figure in a "
            "fabricated personal relationship circulated online, causing "
            "emotional distress and misinformation."
        ),
        propagation_vector=PropagationVector.MARKET_CONTAGION,
        affected_nodes=[
            NodeLayer.FOUNDATION_MODEL.value,
            NodeLayer.SaaS_PROVIDER.value,
        ],
        base_frequency=0.3,
        severity_distribution=SeverityDistribution(
            name="lognormal",
            params={"mu": 2.9, "sigma": 1.3},
        ),
        severity_materialization=None,
        metadata={
            "id": 1154,
            "incident_id": "Incident 1344",
            "victim": "Kate Garraway",
            "harm": ["reputational damage", "family distress"],
        },
    ),

    Scenario(
        name="Automated Shuttle Bus Rear-Ended During USDOT Demonstration",
        event_type=EventType.SYSTEMIC_FAILURE,
        trigger=(
            "An autonomous shuttle bus operating in a public demonstration "
            "was rear-ended by a human-driven vehicle making an illegal "
            "lane change."
        ),
        propagation_vector=PropagationVector.SUPPLY_CHAIN,
        affected_nodes=[
            NodeLayer.ENTERPRISE.value,
            NodeLayer.ISOLATED_NODE.value,
        ],
        base_frequency=0.05,
        severity_distribution=SeverityDistribution(
            name="pareto",
            params={"alpha": 2.5, "scale": 3.0},
        ),
        severity_materialization=None,
        metadata={
            "id": 1155,
            "incident_id": "Incident 1347",
            "operator": "Beep, Inc.",
            "location": "Washington, D.C.",
            "injuries_reported": False,
        },
    ),

]
