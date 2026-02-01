"""
The taxonomy is informed by:
- OWASP Top 10 for LLM Applications
- NIST AI Risk Management Framework
- Industry AI security best practices
"""

from enum import Enum


class RiskSurface(str, Enum):
    """
    Categories of AI risk exposure that can be mitigated by safety controls.
    
    Each risk surface represents a distinct attack vector or failure mode
    that specific safety measures can address. Companies may have different
    levels of coverage across surfaces depending on their safety posture.
    
    The effectiveness of safety controls varies by surface - some surfaces
    (like prompt injection) have well-established mitigation tools, while
    others (like alignment failures) remain challenging to address.
    """
    
    DATA_QUALITY = "data_quality"
    PROMPT_INJECTION = "prompt_injection"
    GATEWAY = "gateway"
    EVALS = "evals"
    MONITORING = "monitoring"
    ACCESS_CONTROL = "access_control"
    OUTPUT_FILTERING = "output_filtering"
    MODEL_GOVERNANCE = "model_governance"
    OTHER = "other"

    
    @classmethod
    def from_string(cls, value: str) -> "RiskSurface":
        normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        
        try:
            return cls(normalized)
        except ValueError:
            pass
        
        for surface in cls:
            if surface.name.lower() == normalized:
                return surface
        
        raise ValueError(
            f"Unknown risk surface: '{value}'. "
            f"Valid values: {[s.value for s in cls]}"
        )
