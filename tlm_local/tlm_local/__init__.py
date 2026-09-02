# .client (which imports .config first) must come before the "from tlm..."
# re-exports: .config loads .env, and tlm caches its settings on first import.
# This file runs before any submodule, so this ordering is what the package's
# correctness rests on. Do not let a linter reorder it. See config.py.
from .client import (
    ADVANCED_CONFIG_FIELDS,
    KNOWN_QUALITY_PRESETS,
    VALIDATED_QUALITY_PRESETS,
    Generation,
    LocalTLM,
    ScoreResult,
)
from .config import LocalTLMConfig
from .errors import (
    EmptyGenerationError,
    EvalsNotSupportedError,
    JudgeCallFailedError,
    JudgeModelNotLocalError,
    ModelNotPulledError,
    OllamaUnavailableError,
    RagNotSupportedError,
)

from tlm.config.presets import ReasoningEffort
from tlm.types import Eval, SimilarityMeasure

__all__ = [
    "LocalTLM",
    "Generation",
    "ScoreResult",
    "LocalTLMConfig",
    "ModelNotPulledError",
    "OllamaUnavailableError",
    "EmptyGenerationError",
    "EvalsNotSupportedError",
    "JudgeCallFailedError",
    "JudgeModelNotLocalError",
    "RagNotSupportedError",
    "KNOWN_QUALITY_PRESETS",
    "VALIDATED_QUALITY_PRESETS",
    "ADVANCED_CONFIG_FIELDS",
    # Re-exported from tlm for discoverability - see CONFIG_REFERENCE.md
    "ReasoningEffort",
    "SimilarityMeasure",
    "Eval",
]
