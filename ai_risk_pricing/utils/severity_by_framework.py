from typing import Literal, Mapping

SeverityLevel = Literal["low", "medium", "high"]


FRAMEWORK_SEVERITY: dict[str, dict[SeverityLevel, float]] = {
    "ProductLiability": {
        "low": 1000000,
        "medium": 2000000,
        "high": 3000000,
    },
    "DSA": {
        "low": 1000000,
        "medium": 2000000,
        "high": 3000000,
    },
    "AIAct": {
        "low": 1000000,
        "medium": 2000000,
        "high": 3000000,
    },
}

FRAMEWORK_INTERACTIONS: dict[frozenset[str], float] = {
    frozenset({"AIAct", "ProductLiability"}): 1.4,
    frozenset({"AIAct", "DSA"}): 1.25,
    frozenset({"DSA", "ProductLiability"}): 1.15,
}


def severity_by_framework(framework_levels: Mapping[str, SeverityLevel]) -> float:
    """Return a combined severity level implied by multiple frameworks.

    Actuarial meaning: this maps each applicable legal / regulatory framework to a
    deterministic gross loss severity proxy (in currency units). When multiple
    frameworks apply, severities are aggregated additively (multiple heads of
    liability/penalties), and then adjusted by multiplicative interaction uplifts
    representing non-linear amplification when regimes overlap.
    """

    base_severities: list[float] = []
    for framework, level in framework_levels.items():
        try:
            base_severities.append(FRAMEWORK_SEVERITY[framework][level])
        except KeyError as exc:
            raise KeyError(
                f"Unknown framework/level: framework={framework!r}, level={level!r}. "
                f"Known frameworks={sorted(FRAMEWORK_SEVERITY.keys())}."
            ) from exc

    total_base = float(sum(base_severities))

    frameworks_present = set(framework_levels.keys())
    interaction_factor = 1.0
    for pair, factor in FRAMEWORK_INTERACTIONS.items():
        if pair.issubset(frameworks_present):
            interaction_factor *= float(factor)

    return total_base * interaction_factor
