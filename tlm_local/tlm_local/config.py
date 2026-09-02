"""Local-Ollama configuration, plus the two process-wide fixes tlm needs.

Both are import-time side effects, and that is the point: tlm caches its
settings the moment it is first imported, so a .env loaded afterwards is too
late for the life of the process. Importing anything from this package runs
this file first. Pitfalls 1 and 2 in the package README explain what each
side effect prevents.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import litellm
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Walks up from the current working directory looking for a .env file -
# standard python-dotenv discovery, no hardcoded path so this works
# regardless of where this package is installed relative to a consuming
# app's repo layout.
load_dotenv()

# Without this, every scoring call crashes: tlm's judge calls always send a
# `logprobs` key Ollama rejects, and tlm's handling of that rejection dies on
# an unrelated AttributeError. The dropped value is already False/None, so no
# signal is lost. Pitfall 2 in the package README.
litellm.drop_params = True

logger.debug("tlm_local: .env loaded, litellm.drop_params enabled")


@dataclass(frozen=True)
class LocalTLMConfig:
    """Immutable configuration for a `LocalTLM` instance. Every field falls
    back to an environment variable if not passed explicitly; see
    ../env.example at the repo root for a documented .env template and
    CONFIG_REFERENCE.md for what each tlm-level setting actually does.
    """

    ollama_api_base: str = field(default_factory=lambda: os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"))
    generator_model: str = field(default_factory=lambda: os.environ.get("GENERATOR_MODEL", "ollama/ministral-3:3b"))
    quality_preset: str = field(default_factory=lambda: os.environ.get("TLM_QUALITY_PRESET", "medium"))
    # Both None by default: tlm picks a sensible workflow-specific default for
    # each when left unset (see CONFIG_REFERENCE.md). Only set these if you
    # want to override that default.
    reasoning_effort: str | None = field(default_factory=lambda: os.environ.get("TLM_REASONING_EFFORT") or None)
    similarity_measure: str | None = field(default_factory=lambda: os.environ.get("TLM_SIMILARITY_MEASURE") or None)

    @property
    def judge_model(self) -> str:
        """Read-only: the judge model `tlm` has actually resolved.

        Asks tlm rather than reading DEFAULT_MODEL, because once tlm has cached
        its settings the two can disagree and it is the environment that lies.
        The import is lazy so this module can finish loading .env first.

        To change the judge, set DEFAULT_MODEL before the process starts.
        """
        from tlm.config.defaults import get_settings

        return str(get_settings().DEFAULT_MODEL)

    @property
    def judge_is_local(self) -> bool:
        """Whether the resolved judge model is served by Ollama.

        False means every scoring call leaves this machine. LocalTLM refuses to
        construct in that case unless explicitly told not to care.
        """
        return self.judge_model.startswith("ollama/")

    def export_ollama_api_base(self) -> None:
        """Publish ollama_api_base where litellm's judge calls will find it.

        Judge calls are built inside tlm and carry no api_base, so litellm
        resolves their host from this variable; without the export, generation
        and scoring would target different servers. Pitfall 3 in the package
        README.

        An explicit config value deliberately wins over a pre-existing env var:
        the two can only differ when the caller passed one, and that is the
        more specific intent. Skipped when they already agree, since mutating
        the environment from a library is not free.
        """
        current = os.environ.get("OLLAMA_API_BASE")
        if current == self.ollama_api_base:
            return
        if current:
            logger.info(
                "Overriding OLLAMA_API_BASE (%s) with this config's ollama_api_base (%s) so tlm's "
                "judge calls reach the same host as generation.",
                current,
                self.ollama_api_base,
            )
        os.environ["OLLAMA_API_BASE"] = self.ollama_api_base
        logger.debug("tlm_local: exported OLLAMA_API_BASE=%s for tlm's judge calls", self.ollama_api_base)
