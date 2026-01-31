from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
import math
import statistics
from typing import Iterable

from ai_risk_pricing.scenario.data.known_knowns_generator import known_knowns_scenarios
from ai_risk_pricing.scenario.schema import (
    EventType,
    Scenario,
    SeverityDistribution,
)
from ai_risk_pricing.utils.distributions import fit_lognormal_mom
from ai_risk_pricing.utils.severity_by_framework import severity_by_framework


def aggregate_incident_risk(
    incidents: Iterable[Scenario],
) -> dict[tuple[str, str], dict]:
    """
    Aggregate incident data into frequency and severity
    grouped by (node_type, risk_type).
    """

    grouped = defaultdict(list)
    for incident in incidents:
        key = (incident.affected_nodes[0], incident.event_type)
        grouped[key].append(incident)

    results = {}

    for key, group in grouped.items():
        node_type, risk_type = key
        frequency = len(group)

        severities = []
        for inc in group:
            if inc.severity_materialization is not None:
                severities.append(inc.severity_materialization)
            else:
                severities.append(
                    severity_by_framework(inc.legal_frameworks or {})
                )

        avg_severity = sum(severities) / len(severities)

        results[key] = {
            "node_type": node_type,
            "risk_type": risk_type,
            "frequency": frequency,
            "avg_severity": avg_severity,
            "total_expected_loss": frequency * avg_severity,
        }

    return results


def create_aggregated_scenarios(incidents: Iterable[Scenario]) -> list[Scenario]:
    """Create group-level `Scenario` objects from incident-level scenarios.

    Actuarial meaning: each incident is treated as one observation drawn from an
    underlying compound process for a given (node layer, event type). We estimate:

    - Frequency: a Poisson intensity \(\lambda\) using the incident observation window
      when dates are available (count / exposure-years).
    - Severity: a parametric lognormal proxy fitted by moment-matching to observed or
      estimated incident severities (materialized loss if known, else a framework-
      implied proxy, else the incident's analytical mean severity).
    """

    def _incident_date(inc: Scenario) -> date | None:
        """Extract an ISO date from incident metadata when available."""
        raw = inc.metadata.get("published_date") or inc.metadata.get("date")
        if isinstance(raw, str):
            try:
                return date.fromisoformat(raw)
            except ValueError:
                return None
        return None

    def _incident_severity_proxy(inc: Scenario) -> float | None:
        """Return a positive severity proxy for parameter estimation."""
        if inc.severity_materialization is not None:
            return float(inc.severity_materialization)
        if inc.legal_frameworks:
            sev = float(severity_by_framework(inc.legal_frameworks))
            return sev if sev > 0 else None

        dist = inc.severity_distribution
        if dist.name == "pareto":
            alpha = dist.params.get("alpha")
            scale = dist.params.get("scale")
            if alpha is None or scale is None or alpha <= 1:
                return None
            return float(alpha * scale / (alpha - 1))
        if dist.name == "lognormal":
            mu = dist.params.get("mu")
            sigma = dist.params.get("sigma")
            if mu is None or sigma is None:
                return None
            return float(math.exp(mu + sigma**2 / 2.0))
        return None

    grouped: dict[tuple[str, EventType], list[Scenario]] = defaultdict(list)
    for inc in incidents:
        if not inc.affected_nodes:
            continue
        grouped[(inc.affected_nodes[0], inc.event_type)].append(inc)

    aggregated: list[Scenario] = []
    for (node_type, event_type), group in grouped.items():
        # Frequency estimation: annualize by observation window when possible.
        dates = [d for d in (_incident_date(inc) for inc in group) if d is not None]
        if len(dates) >= 2:
            obs_years = max((max(dates) - min(dates)).days / 365.25, 1.0)
        else:
            obs_years = 1.0
        lambda_hat = float(len(group) / obs_years)

        severities = [
            s
            for s in (_incident_severity_proxy(inc) for inc in group)
            if s is not None and s > 0.0
        ]
        sev_dist_name, sev_dist_params = fit_lognormal_mom(severities) if severities else group[0].severity_distribution
        sev_dist = SeverityDistribution(name=sev_dist_name, params=sev_dist_params)
        pv = Counter(inc.propagation_vector for inc in group).most_common(1)[0][0]
        
        tail_multiplier = float(statistics.fmean([inc.tail_multiplier for inc in group]))
        capability_threshold = float(statistics.fmean([inc.capability_threshold for inc in group]))
        threshold_multiplier = float(statistics.fmean([inc.threshold_multiplier for inc in group]))

        ids = [inc.metadata.get("id") for inc in group if "id" in inc.metadata]
        meta = {
            "aggregated_from_n_incidents": len(group),
            "aggregated_from_ids": ids,
            "observation_window_min_date": min(dates).isoformat() if dates else None,
            "observation_window_max_date": max(dates).isoformat() if dates else None,
        }

        template = group[0]
        aggregated.append(
            replace(
                template,
                name=f"Aggregated Incidents for {node_type} / {event_type.value}",
                trigger=f"Aggregated from {len(group)} incident scenarios",
                propagation_vector=pv,
                affected_nodes=[node_type],
                base_frequency=lambda_hat,
                severity_distribution=sev_dist,
                severity_materialization=None,
                legal_frameworks=None,
                tail_multiplier=tail_multiplier,
                capability_threshold=capability_threshold,
                threshold_multiplier=threshold_multiplier,
                metadata=meta,
            )
        )

    return aggregated

if __name__ == "__main__":
    results = aggregate_incident_risk(known_knowns_scenarios)
    print(results)

    aggregated = create_aggregated_scenarios(known_knowns_scenarios)
    breakpoint()
    print(f"Aggregated scenarios: {len(aggregated)}")

    