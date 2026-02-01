from pathlib import Path
from statistics import mean

import yaml


def load_risks(path: str) -> list[dict]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def index_by_category(risks: list[dict]) -> dict[str, list[dict]]:
    index = {}
    for r in risks:
        category = r.get("category")
        if category:
            index.setdefault(category, []).append(r)
    return index


def index_by_layer(risks: list[dict]) -> dict[str, list[dict]]:
    index = {}
    for r in risks:
        layer = r.get("layer")
        if layer:
            index.setdefault(layer, []).append(r)
    return index

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


def aggregate_severity_params(risks: list[dict]) -> dict:
    values = [
        SEVERITY_MAP[r["expected_severity"]]
        for r in risks
        if r.get("expected_severity") in SEVERITY_MAP
    ]

    mu = mean(values) if values else 1.0
    sigma = 0.5 + (mu / 3)

    return {
        "name": "lognormal",
        "params": {
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
        },
    }


if __name__ == "__main__":
    ai_act_risks = load_risks(f"{Path(__file__).parent}/data/ai_act.yaml")
    #iso_nist_risks = load_risks(f"{Path(__file__).parent}/data/iso_nist.yaml")
    ai_act_index = index_by_category(ai_act_risks["ai_act_risk_taxonomy"])
   # iso_nist_index = index_risks_by_category(iso_nist_risks)
    print(ai_act_index)
    #print(iso_nist_index)