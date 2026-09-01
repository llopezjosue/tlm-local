"""Loads local-Ollama configuration and applies the process-wide fixes tlm
needs to work safely against Ollama.

Import this module (or anything from the tlm_local package, which imports
this first) before `import tlm` appears anywhere in your process: tlm builds
and caches its own Settings object - which reads the judge model from the
DEFAULT_MODEL env var - the moment it is first imported. Set env vars after
that point and they're too late for the rest of the process's lifetime, no
matter what you do afterwards. This is why the .env loading below happens
unconditionally at import time, not lazily inside a function.
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

# tlm's self-reflection/consistency judge calls always send a `logprobs` key
# that Ollama's chat API rejects outright. tlm catches the resulting
# completion failure internally, but the downstream code handling that
# failure doesn't check for it and crashes with an unrelated AttributeError
# instead of degrading gracefully - so without this, every single scoring
# call crashes. The value tlm actually requests there is already
# False/None, so dropping the param loses no real signal - it just lets
# Ollama accept the request instead of rejecting it outright.
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

        This deliberately asks `tlm` rather than reading DEFAULT_MODEL from the
        environment. The two can disagree, and when they do it is the env var
        that is misleading: `tlm` caches its Settings on first import
        (lru_cache), so a DEFAULT_MODEL exported after that point is read back
        by os.environ but has no effect on any judge call. An earlier version
        of this property read the env var with a hardcoded "ollama/qwen2.5:7b"
        fallback, which reported a local judge in exactly the situation that
        matters most: DEFAULT_MODEL not visible to `tlm`, whose own fallback is
        the hosted `gpt-4.1-mini`.

        Imported lazily: this module must finish loading .env before `tlm` is
        first imported anywhere in the process (see the module docstring), so
        it cannot import `tlm` at module level.

        To actually change the judge model, set DEFAULT_MODEL in your
        environment or .env BEFORE the process starts.
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
        """Publish ollama_api_base where litellm's ollama route will find it.

        Only `generate()` can pass api_base explicitly, because it builds its
        own litellm call. Judge and consistency calls are built inside `tlm`,
        which never sets api_base on them (verified: those calls go out with
        api_base=None), and `tlm`'s own Config.api_base field is not honored on
        that path either. litellm therefore resolves the Ollama host from the
        OLLAMA_API_BASE env var, falling back to http://localhost:11434.

        Without this, LocalTLM(LocalTLMConfig(ollama_api_base="http://box:11434"))
        would generate on that host and silently score on localhost.

        This config object wins over a pre-existing env var when the two differ,
        which is the opposite of the usual precedence and is deliberate. The two
        can only differ when the caller passed ollama_api_base explicitly, since
        the field otherwise defaults to reading that same variable. An explicit
        argument is the more specific intent, and honoring the env var instead
        would split generation and scoring across two hosts, which is never what
        anyone wants. Mutating the environment from a library is not free, so it
        is skipped entirely when the values already agree.
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
