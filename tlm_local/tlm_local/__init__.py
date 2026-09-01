# .client (which imports .config first) must be imported before anything
# that touches `tlm` directly, including the "from tlm..." re-exports below:
# .config's import-time side effect (loading .env before tlm caches its
# settings) has to run first. See config.py's docstring for why. Importing
# `tlm_local` at all - by any submodule, from anywhere - always runs this
# file first, so this ordering is the one guarantee the whole package's
# correctness rests on. Do not let a linter or auto-import-sorter reorder
# this block.
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
