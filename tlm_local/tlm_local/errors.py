"""Typed exceptions for Ollama/local-model failure modes, so callers can show
a clean message instead of a raw litellm/Ollama traceback. litellm raises
different exception types for "server down" vs "model not pulled" depending
on which provider prefix is used (APIConnectionError for both, via the
`ollama`/`ollama_chat` providers; NotFoundError vs InternalServerError via
the `openai` provider pointed at Ollama's OpenAI-compatible endpoint).

All three share `openai.APIError` as a common ancestor - NOT
`litellm.exceptions.APIError`, a confusing gotcha: litellm defines its own,
*different* class also named APIError in litellm.exceptions, which is not
actually in this inheritance chain despite the identical name (verified
directly: different module, different id(), issubclass() is False against
it). We catch the real ancestor, openai.APIError, and disambiguate the same
way in both provider cases: by inspecting the message text (also verified
empirically).
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
    """Passing `evals` to `TLM.score()` crashes in `trustworthy-llm==0.0.3`,
    reproduced independent of tlm_local (a bare `TLM().score(evals=[...])`
    call crashes the same way): tlm's pipeline factory
    (`tlm/pipeline/factory.py`, `SemanticEvaluationScoreGenerator`
    construction) passes `model=config.model` into a component whose
    `__init__` doesn't accept that argument, raising
    `TypeError: Component.__init__() got an unexpected keyword argument 'model'`.
    Not something callers can work around, and not fixable here without
    forking tlm - see CONFIG_REFERENCE.md.
    """

    def __init__(self):
        super().__init__(
            "evals is broken in the installed tlm version (trustworthy-llm==0.0.3): "
            "a bug in tlm's own pipeline factory, not in this wrapper. See "
            "EvalsNotSupportedError's docstring and CONFIG_REFERENCE.md."
        )


class JudgeModelNotLocalError(Exception):
    """The judge model `tlm` actually resolved is not an Ollama model.

    `tlm` reads its judge model from the DEFAULT_MODEL env var into its own
    lru_cached Settings on first import. When that variable is not visible at
    that moment, `tlm` falls back to its own default, `gpt-4.1-mini`
    (tlm/config/models.py), with provider "openai" and an api_key auto-filled
    from OPENAI_API_KEY (tlm/config/defaults.py). Every judge call then leaves
    the machine, carrying the question and the answer, and nothing in `tlm`
    warns about it.

    That silently defeats this package's whole premise, so LocalTLM checks the
    resolved value at construction rather than letting it happen. The check is
    skippable via LocalTLM(require_local_judge=False) for the deliberate case
    of scoring against a hosted judge.
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
    """Passing `context` to score() crashes in `trustworthy-llm==0.0.3`.

    A non-None `context` makes tlm select WorkflowType.RAG (tlm/api.py), which
    makes it inject DEFAULT_RAG_EVALS when the caller passed no evals
    (tlm/inference.py). Those injected evals then hit the same broken pipeline
    factory as user-supplied ones (see EvalsNotSupportedError): tlm builds
    SemanticEvaluationScoreGenerator with a `model=` argument the component
    does not accept.

    So RAG scoring is unreachable in this version for the same upstream reason
    evals are, but it fails even when the caller passed no evals at all, which
    is why it needs its own error rather than being folded into
    EvalsNotSupportedError.
    """

    def __init__(self):
        super().__init__(
            "Passing context= (RAG scoring) is broken in the installed tlm version "
            "(trustworthy-llm==0.0.3): a non-None context selects tlm's RAG workflow, which injects "
            "default RAG evals, which hit the same upstream pipeline-factory bug as evals. Not fixable "
            "in this wrapper. See RagNotSupportedError's docstring and CONFIG_REFERENCE.md."
        )


class JudgeCallFailedError(Exception):
    """A judge call failed and `tlm`'s scoring pipeline crashed on the failure.

    `tlm` converts any judge-side exception into a CompletionFailure object
    that carries only `error` and `type` (tlm/types/base.py). Its
    self-reflection scoring component then reads `.per_field_metadata` on every
    completion unconditionally (tlm/components/scores/
    self_reflection_score_computation.py), which that object does not have, so
    the whole score dies with an unrelated AttributeError.

    `litellm.drop_params = True` removes only ONE trigger for this (the
    rejected `logprobs` parameter). Every other cause still hits it: a timeout,
    Ollama restarting, the model being evicted from memory, a dropped
    connection, a 500 from an overloaded server. Since the graceful NaN path
    further down is unreachable, this cannot be fixed from the wrapper without
    forking `tlm`; LocalTLM.score() translates the AttributeError into this
    typed error so callers can at least tell "a judge call failed" from a bug
    in their own code.
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
