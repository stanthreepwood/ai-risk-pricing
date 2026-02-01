from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .risk_surface import RiskSurface
from .measure import SafetyMeasure, SafetyProfile


@dataclass
class ProviderInfo:
    """
    Metadata about a known safety provider.
    
    Contains effectiveness scores and classification information
    loaded from the providers registry YAML.
    """
    
    name: str
    surface: RiskSurface
    effectiveness: float
    tier: str
    notes: str | None = None


class ProviderRegistry:
    """
    Registry of known safety providers loaded from YAML.
    
    Provides lookup of provider effectiveness scores and factory methods
    for creating SafetyMeasures. The registry is loaded from a YAML file
    containing curated effectiveness ratings for common AI safety tools.
    
    The registry supports:
    - Looking up effectiveness scores by surface and provider name
    - Creating SafetyMeasure instances with automatic effectiveness lookup
    - Listing available providers for each risk surface
    
    Example:
        >>> registry = ProviderRegistry()
        >>> score = registry.get_effectiveness(RiskSurface.PROMPT_INJECTION, "Lakera Guard")
        >>> print(f"Effectiveness: {score:.2f}")
        Effectiveness: 0.85
    """
    
    _DEFAULT_YAML = Path(__file__).parent / "data" / "providers.yaml"
    
    def __init__(self, yaml_path: Path | None = None) -> None:
        self._providers: dict[RiskSurface, dict[str, ProviderInfo]] = {}
        self._load(yaml_path or self._DEFAULT_YAML)
    
    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        for surface_key, surface_data in data.get("risk_surfaces", {}).items():
            try:
                surface = RiskSurface(surface_key)
            except ValueError:
                continue
            
            self._providers[surface] = {}
            for prov in surface_data.get("providers", []):
                info = ProviderInfo(
                    name=prov["name"],
                    surface=surface,
                    effectiveness=prov["effectiveness"],
                    tier=prov.get("tier", "other"),
                    notes=prov.get("notes"),
                )

                self._providers[surface][prov["name"].lower()] = info
    
    def get_provider(
        self,
        surface: RiskSurface,
        name: str,
    ) -> ProviderInfo | None:
        surface_providers = self._providers.get(surface, {})
        return surface_providers.get(name.lower())
    
    def get_effectiveness(
        self,
        surface: RiskSurface,
        name: str,
        default: float = 0.25,
    ) -> float:
        info = self.get_provider(surface, name)
        return info.effectiveness if info else default
    
    def list_providers(self, surface: RiskSurface) -> list[str]:
        providers = self._providers.get(surface, {})
        return [info.name for info in providers.values()]
    
    def list_all_surfaces(self) -> list[RiskSurface]:
        return list(self._providers.keys())
    
    def create_measure(
        self,
        surface: RiskSurface,
        provider_name: str,
        coverage: float = 1.0,
        effectiveness_override: float | None = None,
    ) -> SafetyMeasure:
        """
        Create a SafetyMeasure from registry data.
        
        Looks up the provider effectiveness from the registry and creates
        a SafetyMeasure instance. Allows overriding the effectiveness score
        for custom implementations that may differ from the registry default.
        
        Args:
            surface: The risk surface this measure addresses.
            provider_name: Name of the provider (looked up in registry).
            coverage: Proportion of traffic/data covered [0, 1].
            effectiveness_override: Optional override for registry effectiveness.
            
        Returns:
            SafetyMeasure configured with registry or overridden effectiveness.
            
        Example:
            >>> registry = ProviderRegistry()
            >>> measure = registry.create_measure(
            ...     surface=RiskSurface.PROMPT_INJECTION,
            ...     provider_name="Lakera Guard",
            ...     coverage=0.95,
            ... )
        """
        if effectiveness_override is not None:
            effectiveness = effectiveness_override
        else:
            effectiveness = self.get_effectiveness(surface, provider_name)
        
        return SafetyMeasure(
            surface=surface,
            provider=provider_name,
            effectiveness=effectiveness,
            coverage=coverage,
        )


# Module-level singleton for convenience
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def build_safety_profile(
    measures_config: list[dict[str, Any]],
    registry: ProviderRegistry | None = None,
) -> SafetyProfile:
    """
    Build a SafetyProfile from a configuration list.
    
    Creates a SafetyProfile by parsing a list of measure specifications
    and looking up provider effectiveness scores from the registry.
    
    Args:
        measures_config: List of measure specs, each with:
            - surface: str (RiskSurface value)
            - provider: str (provider name)
            - coverage: float (optional, default 1.0)
            - effectiveness: float (optional override)
        registry: Provider registry (uses global if not provided).
        
    Returns:
        SafetyProfile with all specified measures.
        
    Example:
        >>> config = [
        ...     {"surface": "prompt_injection", "provider": "Lakera Guard"},
        ...     {"surface": "gateway", "provider": "Portkey", "coverage": 0.9},
        ...     {"surface": "evals", "provider": "Custom", "effectiveness": 0.6},
        ... ]
        >>> profile = build_safety_profile(config)
    """
    registry = registry or get_registry()
    measures = []
    
    for spec in measures_config:
        surface = RiskSurface(spec["surface"])
        measure = registry.create_measure(
            surface=surface,
            provider_name=spec["provider"],
            coverage=spec.get("coverage", 1.0),
            effectiveness_override=spec.get("effectiveness"),
        )
        measures.append(measure)
    
    return SafetyProfile(measures=measures)


def build_profile_from_selections(
    selections: dict[str, str | tuple[str, float]],
    registry: ProviderRegistry | None = None,
) -> SafetyProfile:
    """
    Build SafetyProfile from a simple surface->provider mapping.
    
    Convenience function for building profiles when only provider names
    are known. Supports optional coverage specification via tuples.
    
    Args:
        selections: Dict mapping surface names to:
            - Provider name (str) for full coverage
            - Tuple of (provider_name, coverage) for partial coverage
        registry: Provider registry (uses global if not provided).
        
    Returns:
        SafetyProfile with measures for each selection.
        
    Example:
        >>> profile = build_profile_from_selections({
        ...     "prompt_injection": "Lakera Guard",
        ...     "gateway": ("Portkey", 0.9),
        ...     "monitoring": "Langfuse",
        ... })
    """
    registry = registry or get_registry()
    measures = []
    
    for surface_str, selection in selections.items():
        surface = RiskSurface(surface_str)
        
        if isinstance(selection, tuple):
            provider, coverage = selection
        else:
            provider, coverage = selection, 1.0
        
        measure = registry.create_measure(
            surface=surface,
            provider_name=provider,
            coverage=coverage,
        )
        measures.append(measure)
    
    return SafetyProfile(measures=measures)
