from statistics import mean

SEVERITY_MAP = {
    "Low": 0.2,
    "Medium": 0.5,
    "Medium-High": 0.8,
    "High": 1.2,
    "Very High": 1.8,
    "Extreme": 2.5,
}

FREQUENCY_MAP = {
    "Low": 0.05,
    "Medium": 0.1,
    "Medium-High": 0.15,
    "High": 0.25,
    "Very High": 0.4,
}


def aggregate_frequency(risks: list[dict]) -> float:
    values = [
        FREQUENCY_MAP[r["expected_frequency"]]
        for r in risks
        if r.get("expected_frequency") in FREQUENCY_MAP
    ]
    return round(mean(values), 3) if values else 0.1


def aggregate_severity_distribution(risks: list[dict]) -> dict:
    values = [
        SEVERITY_MAP[r["expected_severity"]]
        for r in risks
        if r.get("expected_severity") in SEVERITY_MAP
    ]

    mu = mean(values) if values else 1.0
    sigma = 0.5 + mu / 3

    return {
        "name": "lognormal",
        "params": {
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
        },
    }
