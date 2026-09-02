"""LocalTLM: a thin async-safe wrapper around cleanlab/tlm's TLM class,
fixing every pitfall found running it against a local Ollama server, and
exposing tlm's full Config surface (not just quality_preset). See the
package README for the pitfalls, and CONFIG_REFERENCE.md for what each `tlm`
parameter actually does.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass

from .config import LocalTLMConfig  # import order: must run before `import tlm` below, see config.py

import litellm
from openai import APIError

from tlm import TLM
from tlm.config.schema import Config
from tlm.types import Eval

from .errors import (
    EvalsNotSupportedError,
    JudgeCallFailedError,
    JudgeModelNotLocalError,
    RagNotSupportedError,
    translate_ollama_error,
)

logger = logging.getLogger(__name__)

# tlm's own five presets. Only "medium" and "high" have been measured on this
# project's hardware/models:
#   medium: self-reflection only, ~59-90s per scored answer (this project's default)
#   high:   + consistency-sampling (4 completions), ~94-179s, AND recalibrates
#           scores rather than just adding strictness - a genuinely good
#           answer measured here dropped from ~0.9 to ~0.72. Not a drop-in
#           swap for "slower but otherwise the same".
# "base"/"low"/"best" are valid tlm presets, verified to run without error on
# this project (see tests/test_client.py), but not benchmarked for score
# quality here.
KNOWN_QUALITY_PRESETS = ("base", "low", "medium", "high", "best")
VALIDATED_QUALITY_PRESETS = ("medium", "high")

# Lower-level tlm.config.schema.Config fields that quality_preset normally
# derives automatically. Accepted through score()'s **advanced_tlm_config
# escape hatch for callers who want finer control than the presets offer -
# see CONFIG_REFERENCE.md for what each one does. Kept as an explicit
# allowlist so a typo surfaces as a clear error here rather than a confusing
# one from inside tlm's own pydantic validation.
ADVANCED_CONFIG_FIELDS = frozenset(
    {
        "num_reference_completions",
        "num_consistency_completions",
        "observed_consistency_temperature",
        "self_reflection_temperature",
        "num_self_reflection_completions",
        "use_prompt_evaluation",
        "prompt_evaluation_temperature",
        "semantic_evaluation_temperature",
    }
)


@dataclass(frozen=True)
class Generation:
    answer: str
    messages: list[dict]
    raw_response: dict
    # Named after tlm's field, which is a misnomer: it carries a mean token
    # probability in [0, 1], not a perplexity. See _mean_token_probability.
    perplexity: float | None = None
    usage: dict | None = None  # prompt/completion/total token counts, from litellm


@dataclass(frozen=True)
class ScoreResult:
    trust_score: float
    raw: dict  # full tlm InferenceResult
    explanation: str | None = None
    """tlm's stated reason for the score, when it has one.

    Above tlm's EXPLAINABILITY_THRESHOLD (0.8) this is always the fixed string
    "Did not find a reason to doubt trustworthiness."; below it, it only carries
    real critique when reasoning_effort is not "none", which is the default for
    a QA workflow. See docs/SCORING.md before relying on it.

    The per-signal sub-scores are NOT here: tlm computes them and drops them.
    That same doc has the logging recipe to see them.
    """


def _mean_token_probability(raw_response: dict) -> float | None:
    """Mean token probability, mean(exp(logprob)), over the generated answer.

    Despite the name of the tlm field this feeds, a mean probability is
    exactly what that field wants. tlm fills it for its own completions with
    get_parsed_answer_tokens_confidence (tlm/utils/parse_utils.py:145-161),
    which averages exp(logprob) per token and clips to 1.0, and the helper it
    uses, _logprob_to_probability, is documented as converting "to probability
    0-1 scale". A real perplexity is exp(-mean logprob), a value of at least 1,
    and feeding one here would put the signal far outside the 0-1 range the
    weighted average assumes - silently, since nothing validates it. The
    misnomer is upstream; the value below was checked to match tlm's own
    computation numerically.

    Only populated when the completion actually carries per-token logprobs,
    which requires generate()'s `openai` provider route (see its docstring);
    None otherwise, and tlm's scoring math renormalizes over whatever signals
    are actually available rather than penalizing a missing one.
    """
    try:
        logprobs = raw_response["choices"][0].get("logprobs")
        content = logprobs.get("content") if logprobs else None
        if not content:
            return None
        probabilities = [math.exp(token["logprob"]) for token in content]
        return sum(probabilities) / len(probabilities) if probabilities else None
    except (KeyError, IndexError, TypeError, AttributeError):
        logger.debug("Could not extract logprobs from raw_response, perplexity signal unavailable", exc_info=True)
        return None


class LocalTLM:
    """Generates and scores chat answers against local Ollama models."""

    def __init__(self, config: LocalTLMConfig | None = None, *, require_local_judge: bool = True):
        """Build a client and assert the stack really is local.

        Two things have to happen here rather than lazily, because both are
        about state `tlm` and litellm read out of the process environment
        instead of from arguments:

        - ollama_api_base is exported so judge calls reach the same host as
          generation (see LocalTLMConfig.export_ollama_api_base).
        - the judge model tlm actually resolved is checked, because tlm's own
          fallback is the hosted gpt-4.1-mini and it fails open, not closed.

        require_local_judge=False skips only the second check, for the
        deliberate case of scoring against a hosted judge.
        """
        self.config = config or LocalTLMConfig()
        self.config.export_ollama_api_base()
        if require_local_judge and not self.config.judge_is_local:
            raise JudgeModelNotLocalError(self.config.judge_model)
        logger.debug(
            "LocalTLM ready: generator=%s judge=%s ollama_api_base=%s",
            self.config.generator_model,
            self.config.judge_model,
            self.config.ollama_api_base,
        )

    async def generate(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> Generation:
        """Generate an answer with the configured (or given) generator model.

        temperature/top_p/seed/stop are optional passthroughs to litellm, left
        unset by default so the provider's own defaults apply. `seed` is the one
        worth knowing about: without it, re-running the same prompt gives a
        different answer and a different score, so no benchmark is reproducible.
        Note that `litellm.drop_params = True` is set package-wide, so a
        parameter the endpoint does not support is dropped rather than refused.

        Routes through litellm's `openai` provider pointed at Ollama's
        OpenAI-compatible endpoint (`<ollama_api_base>/v1`), not the
        `ollama`/`ollama_chat` providers: verified directly (via
        litellm._turn_on_debug()) that both of those target Ollama's
        *native* API and silently drop the `logprobs` parameter into an
        `options` bag where it isn't a recognized key - even though Ollama's
        OpenAI-compatible endpoint genuinely supports it (confirmed with
        real per-token data over raw HTTP). Real logprobs let us populate
        tlm's otherwise-unavailable perplexity signal (see
        _mean_token_probability above and the package README's "Known
        limitations" section for the full story).
        """
        model = model or self.config.generator_model
        bare_model = model.split("/", 1)[-1]  # strip an "ollama/" prefix if present
        # Omitted rather than passed as None, so the provider's own default
        # applies instead of an explicit null it may or may not accept.
        optional = {"temperature": temperature, "top_p": top_p, "seed": seed, "stop": stop}
        optional = {key: value for key, value in optional.items() if value is not None}
        logger.debug("generate: model=%s max_tokens=%d optional=%s", model, max_tokens, sorted(optional))
        try:
            response = await litellm.acompletion(
                model=f"openai/{bare_model}",
                api_base=f"{self.config.ollama_api_base}/v1",
                api_key="ollama",  # dummy value: unchecked by Ollama, but litellm's openai provider requires non-empty
                messages=messages,
                max_tokens=max_tokens,
                logprobs=True,
                **optional,
            )
        except APIError as e:
            translated = translate_ollama_error(e, model)
            if translated is None:
                # Not a failure this package can name; relabelling it would
                # point the reader at the wrong cause.
                logger.warning("generate failed for model=%s with an unrecognized API error: %s", model, e)
                raise
            logger.warning("generate failed for model=%s: %s", model, translated)
            raise translated from e

        raw_response = response.model_dump()
        answer = raw_response["choices"][0]["message"]["content"]
        perplexity = _mean_token_probability(raw_response)
        logger.debug("generate: got %d chars, perplexity=%s", len(answer), perplexity)
        return Generation(
            answer=answer,
            messages=messages,
            raw_response=raw_response,
            perplexity=perplexity,
            usage=raw_response.get("usage"),
        )

    async def score(
        self,
        messages: list[dict],
        raw_response: dict,
        *,
        model: str | None = None,
        context: str | None = None,
        perplexity: float | None = None,
        quality_preset: str | None = None,
        reasoning_effort: str | None = None,
        similarity_measure: str | None = None,
        constrain_outputs: list[str] | None = None,
        evals: list[Eval] | None = None,
        **advanced_tlm_config: object,
    ) -> ScoreResult:
        """Score an already-generated answer for trustworthiness.

        TLM.score()/.create() are synchronous and call
        self._event_loop.run_until_complete(...) internally. Calling that
        directly from a coroutine - which already has a running event loop -
        raises "RuntimeError: This event loop is already running" (confirmed
        by reproducing the crash). Running it in a worker thread avoids the
        conflict: no loop is running there yet, so TLM can safely create its
        own.

        raw_response must be a dict as produced by a litellm ModelResponse's
        .model_dump() (not a real openai.types.chat.ChatCompletion instance -
        litellm's ModelResponse isn't one, so it never gets the automatic
        `{"chat_completion": ...}` wrapping TLM.score() applies to real
        ChatCompletion objects). We wrap it in that same shape ourselves
        below, since TLM.score() expects that exact structure to extract the
        answer text internally.

        perplexity, if given (see generate()'s return value), is passed
        through as a top-level "perplexity" key: verified directly in tlm's
        source (tlm/types/completion.py, Completion.from_completion_dict)
        that TLM.score() never computes this itself from logprobs for an
        externally-supplied response - it only reads a literal "perplexity"
        key if the caller provides one.

        quality_preset/reasoning_effort/similarity_measure override
        self.config's values for this call only - lets one LocalTLM instance
        serve different scoring depths per request (e.g. a frontend
        selector) without reconstructing the client. Only "medium" and
        "high" are validated on this project for quality_preset (see
        VALIDATED_QUALITY_PRESETS); every parameter here is documented in
        CONFIG_REFERENCE.md, including which ones simply have no effect
        outside certain presets/workflows.

        constrain_outputs restricts answers to a fixed set of values
        (multiple-choice/classification workflows) - not relevant to a plain
        QA chatbot, exposed for completeness.

        evals lets you attach custom semantic evaluation criteria on top of
        the core trustworthiness score (see tlm.types.Eval); re-exported
        from this package for convenience. Currently raises
        EvalsNotSupportedError on any non-empty value: a reproducible bug in
        tlm==0.0.3's own pipeline factory, not something this wrapper can
        work around - see that exception's docstring.

        **advanced_tlm_config accepts the lower-level tlm Config fields that
        quality_preset normally derives automatically (see
        ADVANCED_CONFIG_FIELDS) for callers who want finer control than the
        presets offer, e.g. score(..., quality_preset="medium",
        num_consistency_completions=2). Unknown keys raise ValueError
        immediately rather than failing inside tlm's own validation.
        """
        model = model or self.config.generator_model
        quality_preset = quality_preset or self.config.quality_preset
        reasoning_effort = reasoning_effort or self.config.reasoning_effort
        similarity_measure = similarity_measure or self.config.similarity_measure

        if quality_preset not in KNOWN_QUALITY_PRESETS:
            raise ValueError(f"Unknown quality_preset {quality_preset!r}, expected one of {KNOWN_QUALITY_PRESETS}")
        unknown_fields = advanced_tlm_config.keys() - ADVANCED_CONFIG_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Unknown tlm Config field(s) {sorted(unknown_fields)}; expected one of "
                f"{sorted(ADVANCED_CONFIG_FIELDS)} (see CONFIG_REFERENCE.md)"
            )

        response_payload: dict = {"chat_completion": raw_response}
        if perplexity is not None:
            response_payload["perplexity"] = perplexity

        tlm_config = Config(
            quality_preset=quality_preset,
            reasoning_effort=reasoning_effort,
            similarity_measure=similarity_measure,
            constrain_outputs=constrain_outputs,
            **advanced_tlm_config,
        )
        logger.debug(
            "score: model=%s quality_preset=%s has_perplexity=%s", model, quality_preset, perplexity is not None
        )

        def _run() -> dict:
            tlm_instance = TLM(config=tlm_config)
            try:
                return tlm_instance.score(
                    model=model,
                    messages=messages,
                    context=context,
                    evals=evals,
                    response=response_payload,
                )
            except TypeError as e:
                # Same tlm==0.0.3 pipeline-factory bug reached two ways, both
                # reproduced independent of tlm_local. Order matters: a non-None
                # context makes tlm inject its own RAG evals, so this fires with
                # evals=None and the evals check below would miss it.
                if "unexpected keyword argument 'model'" in str(e):
                    if context is not None:
                        raise RagNotSupportedError() from e
                    if evals:
                        raise EvalsNotSupportedError() from e
                raise
            except AttributeError as e:
                # A failed judge call crashes tlm's own scoring pipeline. Narrow
                # the match to that exact upstream defect so real AttributeErrors
                # still surface: see JudgeCallFailedError's docstring.
                if "per_field_metadata" in str(e):
                    raise JudgeCallFailedError(e) from e
                raise

        result = await asyncio.to_thread(_run)
        raw_score = result["trustworthiness_score"]
        # Every judge call failed but none crashed the pipeline: tlm averages zero
        # surviving signals, which reaches us as None or NaN depending on where it
        # gave up. float(None) would raise a bare TypeError three frames from here,
        # and a NaN would survive all the way into the caller's JSON as an
        # out-of-spec literal.
        if raw_score is None:
            raise JudgeCallFailedError(ValueError("tlm returned trustworthiness_score=None"))
        trust_score = float(raw_score)
        if math.isnan(trust_score):
            raise JudgeCallFailedError(ValueError("tlm returned trustworthiness_score=NaN"))
        logger.debug("score: trust_score=%.3f", trust_score)
        return ScoreResult(trust_score=trust_score, raw=result, explanation=result.get("explanation"))

    async def generate_and_score(
        self,
        messages: list[dict],
        *,
        generator_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        quality_preset: str | None = None,
        reasoning_effort: str | None = None,
        similarity_measure: str | None = None,
        constrain_outputs: list[str] | None = None,
        evals: list[Eval] | None = None,
        **advanced_tlm_config: object,
    ) -> tuple[Generation, ScoreResult]:
        """Convenience: generate then score in one call. All keyword
        arguments beyond generator_model/max_tokens are forwarded to
        score() - see its docstring for what each one does.
        """
        generation = await self.generate(
            messages,
            model=generator_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
        )
        score_result = await self.score(
            generation.messages,
            generation.raw_response,
            model=generator_model,
            perplexity=generation.perplexity,
            quality_preset=quality_preset,
            reasoning_effort=reasoning_effort,
            similarity_measure=similarity_measure,
            constrain_outputs=constrain_outputs,
            evals=evals,
            **advanced_tlm_config,
        )
        return generation, score_result
