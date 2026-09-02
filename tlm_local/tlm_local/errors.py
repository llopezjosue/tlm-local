"""Typed exceptions for Ollama and upstream-tlm failure modes, so callers can
show a clean message instead of a raw traceback.

Two things shape this module. litellm raises different exception types per
provider for the same failure, so the two named cases are told apart by their
message text rather than their class. And the ancestor imported below is
`openai.APIError`, not the same-named class in litellm.exceptions, which is
not in the inheritance chain despite the name. Pitfall 6 in the package
README.
"""

from __future__ import annotations

from openai import APIError


class OllamaUnavailableError(Exception):
    """The local Ollama server could not be reached at all."""

    def __init__(self):
        super().__init__("Could not reach the local Ollama server. Is `ollama serve` running?")


class ModelNotPulledError(Exception):
    """A configured model string has not been `ollama pull`-ed locally."""

    def __init__(self, model: str):
        self.model = model
        bare_model = model.split("/", 1)[-1]
        super().__init__(f"Model '{model}' is not available locally. Run `ollama pull {bare_model}`.")


class EvalsNotSupportedError(Exception):
    """`evals` crashes tlm 0.0.3's pipeline factory, which builds
    SemanticEvaluationScoreGenerator with a `model=` argument the component
    does not accept. Reproduced with a bare TLM().score(evals=[...]), so it is
    upstream and not fixable here. CONFIG_REFERENCE.md has the detail.
    """

    def __init__(self):
        super().__init__(
            "evals is broken in the installed tlm version (trustworthy-llm==0.0.3): "
            "a bug in tlm's own pipeline factory, not in this wrapper. See "
            "EvalsNotSupportedError's docstring and CONFIG_REFERENCE.md."
        )


class EmptyGenerationError(Exception):
    """The generator returned no content.

    Reachable on a filtered, aborted or truncated-to-nothing completion. Named
    here because it otherwise surfaces as a TypeError from a logging line, which
    points at the wrong thing entirely.
    """

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model {model!r} returned an empty completion. Retrying usually succeeds.")


class JudgeModelNotLocalError(Exception):
    """The judge model tlm resolved is not an Ollama model.

    tlm's fallback when it cannot see DEFAULT_MODEL is the hosted
    gpt-4.1-mini, keyed from OPENAI_API_KEY, so every judge call would leave
    the machine carrying the question and the answer, without warning. Checked
    at construction because it fails open. Pitfall 1 in the package README.
    """

    def __init__(self, resolved_model: str):
        self.resolved_model = resolved_model
        super().__init__(
            f"tlm resolved its judge model to {resolved_model!r}, which is not a local Ollama model. "
            "Every scoring call would leave this machine. Set DEFAULT_MODEL=ollama/<model> in your "
            "environment or .env BEFORE the process starts (tlm caches this on first import), or pass "
            "LocalTLM(require_local_judge=False) if you really do want a hosted judge."
        )


class RagNotSupportedError(Exception):
    """`context=` crashes tlm 0.0.3 for the same reason `evals` does.

    A non-None context selects tlm's RAG workflow, which injects default RAG
    evals of its own, which hit the same pipeline-factory bug. It therefore
    fails even when the caller passed no evals, which is why it is a separate
    error from EvalsNotSupportedError.
    """

    def __init__(self):
        super().__init__(
            "Passing context= (RAG scoring) is broken in the installed tlm version "
            "(trustworthy-llm==0.0.3): a non-None context selects tlm's RAG workflow, which injects "
            "default RAG evals, which hit the same upstream pipeline-factory bug as evals. Not fixable "
            "in this wrapper. See RagNotSupportedError's docstring and CONFIG_REFERENCE.md."
        )


class JudgeCallFailedError(Exception):
    """A judge call failed and tlm's scoring pipeline crashed on the failure.

    tlm turns any judge-side exception into a CompletionFailure and then reads
    .per_field_metadata on it unconditionally, so any judge failure takes the
    whole score down: a timeout, a restart, an evicted model, an overloaded
    server. drop_params removes one trigger, not the defect, and the graceful
    path below it is unreachable. Naming it is all this package can do; a retry
    usually succeeds. Pitfall 2 in the package README.
    """

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(
            "A judge call failed and tlm's scoring pipeline crashed on the failure rather than "
            "degrading (a known upstream defect in trustworthy-llm==0.0.3: CompletionFailure has no "
            "per_field_metadata). Usual causes are Ollama restarting, the judge model being evicted, a "
            "timeout, or an overloaded server. Retrying the request normally succeeds. See "
            "JudgeCallFailedError's docstring."
        )


# Substrings that identify a transport failure rather than a rejected request.
# Matched against the message because the exception type does not separate the
# two: see this module's docstring.
_UNREACHABLE_MARKERS = (
    "connection error",
    "connection refused",
    "failed to connect",
    "connect call failed",
    "max retries",
    "timed out",
    "timeout",
)


def translate_ollama_error(error: APIError, model: str) -> Exception | None:
    """Name an Ollama failure, or return None when it cannot be named.

    Returning None matters as much as the two hits. Anything reaching this
    function used to come back as OllamaUnavailableError, so a request Ollama
    understood and refused - a context longer than the model's window, a
    parameter it rejects, a 500 from the model itself - was reported as "is
    the server running?", sending the reader to check a server that was never
    down. The caller re-raises the original error instead, which says less but
    says nothing false.
    """
    message = str(error).lower()
    if "not found" in message:
        return ModelNotPulledError(model)
    if any(marker in message for marker in _UNREACHABLE_MARKERS):
        return OllamaUnavailableError()
    return None
