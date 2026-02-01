import pytest
from pathlib import Path

from ai_risk_pricing.safety.risk_surface import RiskSurface
from ai_risk_pricing.safety.measure import SafetyMeasure, SafetyProfile
from ai_risk_pricing.safety.providers import (
    ProviderRegistry,
    ProviderInfo,
    get_registry,
    build_safety_profile,
    build_profile_from_selections,
)
from ai_risk_pricing.safety.mitigation import MitigationEngine, MitigationParams
from ai_risk_pricing.scenario.schema import (
    Scenario,
    EventType,
    PropagationVector,
    SeverityDistribution,
)


# =============================================================================
# RiskSurface Tests
# =============================================================================

class TestRiskSurface:
    """Tests for RiskSurface enum."""
    
    def test_all_surfaces_defined(self):
        """Verify all expected risk surfaces are defined."""
        expected = {
            "data_quality",
            "prompt_injection",
            "gateway",
            "evals",
            "monitoring",
            "access_control",
            "output_filtering",
            "model_governance",
            "other",
        }
        actual = {s.value for s in RiskSurface}
        assert actual == expected
    
    def test_from_string_exact(self):
        """Test parsing exact enum values."""
        assert RiskSurface.from_string("prompt_injection") == RiskSurface.PROMPT_INJECTION
        assert RiskSurface.from_string("data_quality") == RiskSurface.DATA_QUALITY
    
    def test_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert RiskSurface.from_string("PROMPT_INJECTION") == RiskSurface.PROMPT_INJECTION
        assert RiskSurface.from_string("Prompt_Injection") == RiskSurface.PROMPT_INJECTION
    
    def test_from_string_with_aliases(self):
        """Test parsing with common variations."""
        assert RiskSurface.from_string("prompt-injection") == RiskSurface.PROMPT_INJECTION
        assert RiskSurface.from_string("prompt injection") == RiskSurface.PROMPT_INJECTION
    
    def test_from_string_invalid(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="Unknown risk surface"):
            RiskSurface.from_string("invalid_surface")


# =============================================================================
# SafetyMeasure Tests
# =============================================================================

class TestSafetyMeasure:
    """Tests for SafetyMeasure dataclass."""
    
    def test_create_measure(self):
        """Test basic measure creation."""
        measure = SafetyMeasure(
            surface=RiskSurface.PROMPT_INJECTION,
            provider="Lakera Guard",
            effectiveness=0.85,
        )
        assert measure.surface == RiskSurface.PROMPT_INJECTION
        assert measure.provider == "Lakera Guard"
        assert measure.effectiveness == 0.85
        assert measure.coverage == 1.0  # default
    
    def test_create_measure_with_coverage(self):
        """Test measure creation with partial coverage."""
        measure = SafetyMeasure(
            surface=RiskSurface.GATEWAY,
            provider="Portkey",
            effectiveness=0.82,
            coverage=0.9,
        )
        assert measure.coverage == 0.9
    
    def test_effective_strength(self):
        """Test effective strength calculation."""
        measure = SafetyMeasure(
            surface=RiskSurface.MONITORING,
            provider="Langfuse",
            effectiveness=0.8,
            coverage=0.5,
        )
        assert measure.effective_strength == 0.4  # 0.8 * 0.5
    
    def test_effective_strength_full_coverage(self):
        """Test effective strength with full coverage."""
        measure = SafetyMeasure(
            surface=RiskSurface.EVALS,
            provider="Promptfoo",
            effectiveness=0.78,
            coverage=1.0,
        )
        assert measure.effective_strength == 0.78
    
    def test_invalid_effectiveness(self):
        """Test that invalid effectiveness raises error."""
        with pytest.raises(ValueError, match="effectiveness must be in"):
            SafetyMeasure(
                surface=RiskSurface.MONITORING,
                provider="Test",
                effectiveness=1.5,
            )
    
    def test_invalid_coverage(self):
        """Test that invalid coverage raises error."""
        with pytest.raises(ValueError, match="coverage must be in"):
            SafetyMeasure(
                surface=RiskSurface.MONITORING,
                provider="Test",
                effectiveness=0.8,
                coverage=-0.1,
            )


# =============================================================================
# SafetyProfile Tests
# =============================================================================

class TestSafetyProfile:
    """Tests for SafetyProfile aggregation."""
    
    def test_empty_profile(self):
        """Test empty profile has zero score."""
        profile = SafetyProfile()
        assert profile.overall_score == 0.0
        assert profile.covered_surfaces == set()
    
    def test_single_measure_profile(self):
        """Test profile with a single measure."""
        measure = SafetyMeasure(
            surface=RiskSurface.PROMPT_INJECTION,
            provider="Lakera Guard",
            effectiveness=0.85,
        )
        profile = SafetyProfile(measures=[measure])
        
        assert profile.surface_score(RiskSurface.PROMPT_INJECTION) == 0.85
        assert profile.surface_score(RiskSurface.GATEWAY) == 0.0
    
    def test_multi_measure_same_surface(self):
        """Test diminishing returns for multiple measures on same surface."""
        measures = [
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Provider1", 0.8),
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Provider2", 0.7),
        ]
        profile = SafetyProfile(measures=measures)
        
        # Combined: 1 - (1-0.8) * (1-0.7) = 1 - 0.2 * 0.3 = 0.94
        score = profile.surface_score(RiskSurface.PROMPT_INJECTION)
        assert abs(score - 0.94) < 0.001
    
    def test_overall_score(self):
        """Test overall score averaging."""
        measures = [
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.9),
            SafetyMeasure(RiskSurface.GATEWAY, "Test", 0.8),
        ]
        profile = SafetyProfile(measures=measures)
        
        # Average of surface scores / total surfaces
        # (0.9 + 0.8) / 9 surfaces = 1.7 / 9 ≈ 0.189
        expected = (0.9 + 0.8) / len(RiskSurface)
        assert abs(profile.overall_score - expected) < 0.001
    
    def test_covered_surfaces(self):
        """Test covered surfaces detection."""
        measures = [
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8),
            SafetyMeasure(RiskSurface.MONITORING, "Test", 0.7),
        ]
        profile = SafetyProfile(measures=measures)
        
        covered = profile.covered_surfaces
        assert RiskSurface.PROMPT_INJECTION in covered
        assert RiskSurface.MONITORING in covered
        assert RiskSurface.GATEWAY not in covered
    
    def test_uncovered_surfaces(self):
        """Test uncovered surfaces detection."""
        measures = [SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8)]
        profile = SafetyProfile(measures=measures)
        
        uncovered = profile.uncovered_surfaces
        assert RiskSurface.GATEWAY in uncovered
        assert RiskSurface.PROMPT_INJECTION not in uncovered
    
    def test_coverage_summary(self):
        """Test coverage summary generation."""
        measures = [
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8),
            SafetyMeasure(RiskSurface.GATEWAY, "Test", 0.7),
        ]
        profile = SafetyProfile(measures=measures)
        
        summary = profile.coverage_summary()
        assert summary["n_measures"] == 2
        assert summary["n_surfaces_covered"] == 2
        assert summary["n_surfaces_total"] == len(RiskSurface)
    
    def test_add_measure(self):
        """Test adding measures to profile."""
        profile = SafetyProfile()
        profile.add_measure(SafetyMeasure(RiskSurface.MONITORING, "Test", 0.75))
        
        assert len(profile.measures) == 1
        assert profile.surface_score(RiskSurface.MONITORING) == 0.75


# =============================================================================
# ProviderRegistry Tests
# =============================================================================

class TestProviderRegistry:
    """Tests for ProviderRegistry loading and lookup."""
    
    @pytest.fixture
    def registry(self):
        """Get the default provider registry."""
        return get_registry()
    
    def test_registry_loads(self, registry):
        """Test that registry loads without error."""
        assert registry is not None
        surfaces = registry.list_all_surfaces()
        assert len(surfaces) > 0
    
    def test_get_provider_existing(self, registry):
        """Test looking up an existing provider."""
        info = registry.get_provider(RiskSurface.PROMPT_INJECTION, "Lakera Guard")
        assert info is not None
        assert info.name == "Lakera Guard"
        assert info.effectiveness == 0.85
    
    def test_get_provider_case_insensitive(self, registry):
        """Test case-insensitive provider lookup."""
        info = registry.get_provider(RiskSurface.PROMPT_INJECTION, "lakera guard")
        assert info is not None
        assert info.name == "Lakera Guard"
    
    def test_get_provider_not_found(self, registry):
        """Test looking up non-existent provider."""
        info = registry.get_provider(RiskSurface.PROMPT_INJECTION, "NonExistent")
        assert info is None
    
    def test_get_effectiveness_existing(self, registry):
        """Test getting effectiveness for existing provider."""
        eff = registry.get_effectiveness(RiskSurface.GATEWAY, "Portkey")
        assert eff == 0.82
    
    def test_get_effectiveness_default(self, registry):
        """Test default effectiveness for unknown provider."""
        eff = registry.get_effectiveness(RiskSurface.GATEWAY, "Unknown", default=0.3)
        assert eff == 0.3
    
    def test_list_providers(self, registry):
        """Test listing providers for a surface."""
        providers = registry.list_providers(RiskSurface.PROMPT_INJECTION)
        assert "Lakera Guard" in providers
        assert "NeMo Guardrails" in providers
    
    def test_create_measure(self, registry):
        """Test creating a measure from registry."""
        measure = registry.create_measure(
            surface=RiskSurface.MONITORING,
            provider_name="Langfuse",
            coverage=0.95,
        )
        assert measure.surface == RiskSurface.MONITORING
        assert measure.provider == "Langfuse"
        assert measure.effectiveness == 0.80
        assert measure.coverage == 0.95
    
    def test_create_measure_with_override(self, registry):
        """Test creating measure with effectiveness override."""
        measure = registry.create_measure(
            surface=RiskSurface.EVALS,
            provider_name="Custom",
            effectiveness_override=0.65,
        )
        assert measure.effectiveness == 0.65


# =============================================================================
# Profile Building Tests
# =============================================================================

class TestProfileBuilding:
    """Tests for profile building functions."""
    
    def test_build_safety_profile(self):
        """Test building profile from config list."""
        config = [
            {"surface": "prompt_injection", "provider": "Lakera Guard"},
            {"surface": "gateway", "provider": "Portkey", "coverage": 0.9},
        ]
        profile = build_safety_profile(config)
        
        assert len(profile.measures) == 2
        assert profile.surface_score(RiskSurface.PROMPT_INJECTION) == 0.85
    
    def test_build_safety_profile_with_override(self):
        """Test building profile with effectiveness override."""
        config = [
            {"surface": "evals", "provider": "Custom", "effectiveness": 0.65},
        ]
        profile = build_safety_profile(config)
        
        assert profile.surface_score(RiskSurface.EVALS) == 0.65
    
    def test_build_profile_from_selections(self):
        """Test building profile from simple selections."""
        selections = {
            "prompt_injection": "Lakera Guard",
            "monitoring": "Langfuse",
        }
        profile = build_profile_from_selections(selections)
        
        assert len(profile.measures) == 2
        assert profile.surface_score(RiskSurface.PROMPT_INJECTION) == 0.85
    
    def test_build_profile_from_selections_with_coverage(self):
        """Test building profile with coverage tuples."""
        selections = {
            "gateway": ("Portkey", 0.9),
        }
        profile = build_profile_from_selections(selections)
        
        measure = profile.measures[0]
        assert measure.coverage == 0.9


# =============================================================================
# MitigationEngine Tests
# =============================================================================

class TestMitigationEngine:
    """Tests for MitigationEngine calculations."""
    
    @pytest.fixture
    def engine(self):
        """Create a mitigation engine with default params."""
        return MitigationEngine()
    
    @pytest.fixture
    def scenario_with_mitigations(self):
        """Create a scenario with known mitigations."""
        return Scenario(
            name="Test Adversarial Attack",
            event_type=EventType.ADVERSARIAL_ATTACK,
            trigger="Test trigger",
            propagation_vector=PropagationVector.API_DEPENDENCY,
            affected_nodes=["enterprise"],
            base_frequency=0.25,
            severity_distribution=SeverityDistribution(
                name="lognormal",
                params={"mu": 4.0, "sigma": 1.5},
            ),
            known_mitigations={
                "prompt_injection": 0.7,
                "gateway": 0.5,
                "monitoring": 0.3,
            },
        )
    
    @pytest.fixture
    def scenario_without_mitigations(self):
        """Create a scenario without mitigations."""
        return Scenario(
            name="Test Scenario",
            event_type=EventType.SYSTEMIC_FAILURE,
            trigger="Test trigger",
            propagation_vector=PropagationVector.SUPPLY_CHAIN,
            affected_nodes=["enterprise"],
            base_frequency=0.1,
            severity_distribution=SeverityDistribution(
                name="pareto",
                params={"alpha": 1.5, "scale": 10.0},
            ),
        )
    
    def test_no_profile_returns_one(self, engine, scenario_with_mitigations):
        """Test that empty profile gives no mitigation."""
        profile = SafetyProfile()
        factor = engine.compute_mitigation_factor(scenario_with_mitigations, profile)
        assert factor == 1.0
    
    def test_no_mitigations_returns_one(self, engine, scenario_without_mitigations):
        """Test that scenario without mitigations gives no reduction."""
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.9),
        ])
        factor = engine.compute_mitigation_factor(scenario_without_mitigations, profile)
        assert factor == 1.0
    
    def test_single_surface_mitigation(self, engine, scenario_with_mitigations):
        """Test mitigation with single surface coverage."""
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8),
        ])
        factor = engine.compute_mitigation_factor(scenario_with_mitigations, profile)
        
        # Expected: max_reduction=0.7, company_score=0.8
        # Reduction = 0.7 * 0.8 = 0.56
        # Factor = 1 - 0.56 = 0.44
        assert abs(factor - 0.44) < 0.001
    
    def test_multi_surface_mitigation(self, engine, scenario_with_mitigations):
        """Test mitigation with multiple surfaces covered."""
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8),
            SafetyMeasure(RiskSurface.GATEWAY, "Test", 0.7),
        ])
        factor = engine.compute_mitigation_factor(scenario_with_mitigations, profile)
        
        # Reductions: prompt=0.7*0.8=0.56, gateway=0.5*0.7=0.35
        # Combined: 1 - (1-0.56)*(1-0.35) = 1 - 0.44*0.65 = 1 - 0.286 = 0.714
        # Factor = 1 - 0.714 = 0.286
        # But capped at min_residual_risk=0.1? No, total_reduction capped at 0.8
        expected_reduction = 1 - (1 - 0.56) * (1 - 0.35)
        expected_factor = max(0.1, 1 - min(expected_reduction, 0.8))
        assert abs(factor - expected_factor) < 0.01
    
    def test_min_residual_floor(self, engine):
        """Test that mitigation respects min residual risk floor."""
        # Create scenario with very high mitigation potential
        scenario = Scenario(
            name="High Mitigation Scenario",
            event_type=EventType.ADVERSARIAL_ATTACK,
            trigger="Test",
            propagation_vector=PropagationVector.API_DEPENDENCY,
            affected_nodes=["enterprise"],
            base_frequency=0.1,
            severity_distribution=SeverityDistribution(
                name="lognormal", params={"mu": 3, "sigma": 1}
            ),
            known_mitigations={
                "prompt_injection": 0.9,
                "gateway": 0.9,
                "monitoring": 0.9,
                "evals": 0.9,
            },
        )
        
        # Create profile with perfect coverage
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 1.0),
            SafetyMeasure(RiskSurface.GATEWAY, "Test", 1.0),
            SafetyMeasure(RiskSurface.MONITORING, "Test", 1.0),
            SafetyMeasure(RiskSurface.EVALS, "Test", 1.0),
        ])
        
        factor = engine.compute_mitigation_factor(scenario, profile)
        
        # Should be at least min_residual_risk (0.1)
        assert factor >= engine.params.min_residual_risk
    
    def test_max_reduction_cap(self):
        """Test that total reduction is capped."""
        params = MitigationParams(max_total_reduction=0.5)
        engine = MitigationEngine(params=params)
        
        scenario = Scenario(
            name="Test",
            event_type=EventType.ADVERSARIAL_ATTACK,
            trigger="Test",
            propagation_vector=PropagationVector.API_DEPENDENCY,
            affected_nodes=["enterprise"],
            base_frequency=0.1,
            severity_distribution=SeverityDistribution(
                name="lognormal", params={"mu": 3, "sigma": 1}
            ),
            known_mitigations={"prompt_injection": 0.9},
        )
        
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 1.0),
        ])
        
        factor = engine.compute_mitigation_factor(scenario, profile)
        
        # With 90% potential and 100% effectiveness, reduction would be 0.9
        # But capped at 0.5, so factor = 1 - 0.5 = 0.5
        assert factor >= 0.5
    
    def test_analyze_mitigation(self, engine, scenario_with_mitigations):
        """Test mitigation analysis output."""
        profile = SafetyProfile(measures=[
            SafetyMeasure(RiskSurface.PROMPT_INJECTION, "Test", 0.8),
        ])
        
        analysis = engine.analyze_mitigation(scenario_with_mitigations, profile)
        
        assert "mitigation_factor" in analysis
        assert "total_reduction" in analysis
        assert "surface_reductions" in analysis
        assert "coverage_gaps" in analysis
        
        # Should have coverage gaps for gateway and monitoring
        assert "gateway" in analysis["coverage_gaps"]
        assert "monitoring" in analysis["coverage_gaps"]


# =============================================================================
# MitigationParams Tests
# =============================================================================

class TestMitigationParams:
    """Tests for MitigationParams validation."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = MitigationParams()
        assert params.min_residual_risk == 0.1
        assert params.max_total_reduction == 0.8
        assert params.diminishing_returns is True
    
    def test_custom_params(self):
        """Test custom parameter values."""
        params = MitigationParams(
            min_residual_risk=0.2,
            max_total_reduction=0.6,
            diminishing_returns=False,
        )
        assert params.min_residual_risk == 0.2
        assert params.max_total_reduction == 0.6
        assert params.diminishing_returns is False
    
    def test_invalid_min_residual(self):
        """Test that invalid min_residual_risk raises error."""
        with pytest.raises(ValueError, match="min_residual_risk must be in"):
            MitigationParams(min_residual_risk=1.5)
    
    def test_invalid_max_reduction(self):
        """Test that invalid max_total_reduction raises error."""
        with pytest.raises(ValueError, match="max_total_reduction must be in"):
            MitigationParams(max_total_reduction=-0.1)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for safety module components."""
    
    def test_full_workflow(self):
        """Test complete workflow from selections to mitigation."""
        # Build profile from selections
        profile = build_profile_from_selections({
            "prompt_injection": "Lakera Guard",
            "gateway": ("Portkey", 0.95),
            "monitoring": "Langfuse",
            "evals": "Promptfoo",
        })
        
        # Create a scenario
        scenario = Scenario(
            name="Adversarial Attack",
            event_type=EventType.ADVERSARIAL_ATTACK,
            trigger="Test",
            propagation_vector=PropagationVector.API_DEPENDENCY,
            affected_nodes=["enterprise"],
            base_frequency=0.25,
            severity_distribution=SeverityDistribution(
                name="lognormal",
                params={"mu": 4.0, "sigma": 1.8},
            ),
            known_mitigations={
                "prompt_injection": 0.7,
                "gateway": 0.5,
                "monitoring": 0.3,
            },
        )
        
        # Compute mitigation
        engine = MitigationEngine()
        factor = engine.compute_mitigation_factor(scenario, profile)
        
        # Should have significant mitigation
        assert factor < 1.0
        assert factor >= engine.params.min_residual_risk
        
        # Verify analysis
        analysis = engine.analyze_mitigation(scenario, profile)
        assert analysis["n_covered_surfaces"] == 3  # All scenario surfaces covered
        assert len(analysis["coverage_gaps"]) == 0
