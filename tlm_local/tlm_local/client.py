"""LocalTLM: a thin async-safe wrapper around cleanlab/tlm's TLM class,
fixing every pitfall found running it against a local Ollama server, and
exposing tlm's full Config surface (not just quality_preset). See the
package README for the pitfalls, and CONFIG_REFERENCE.md for which `tlm`
parameters are inert or broken.
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

# All five run; only medium and high are benchmarked for score quality here,
# and high is not a slower medium - it reweights. See docs/SCORING.md.
KNOWN_QUALITY_PRESETS = ("base", "low", "medium", "high", "best")
VALIDATED_QUALITY_PRESETS = ("medium", "high")

# Config fields quality_preset normally derives, reachable through score()'s
# **advanced_tlm_config. An allowlist rather than a passthrough so a typo fails
# here, not inside tlm's pydantic validation. CONFIG_REFERENCE.md flags the traps.
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
    perplexity: float | None = None  # tlm's name; a probability, not a perplexity
    usage: dict | None = None  # prompt/completion/total token counts, from litellm


@dataclass(frozen=True)
class ScoreResult:
    trust_score: float
    raw: dict  # full tlm InferenceResult
    explanation: str | None = None
    """tlm's stated reason for the score. A fixed string above tlm's 0.8
    threshold, and real critique only when reasoning_effort is not "none".
    Carries no per-signal breakdown; docs/SCORING.md explains both.
    """


def _mean_token_probability(raw_response: dict) -> float | None:
    """Mean token probability, mean(exp(logprob)), over the generated answer.

    Not a perplexity, and deliberately so: tlm's field of that name wants a
    probability in [0, 1], matching what it computes for its own completions
    (tlm/utils/parse_utils.py:145-161). Passing a real perplexity would be the
    wrong scale, silently. docs/SCORING.md has the derivation.

    None when the response carries no logprobs, which tlm renormalizes around
    rather than penalizing.
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

        Both steps here are eager because both concern process-wide state that
        tlm and litellm read from the environment, not from arguments: the
        Ollama host is exported so judge calls reach it, and the judge model
        tlm resolved is checked, since its fallback is a hosted model and it
        fails open. Pitfalls 1 and 3 in the package README.

        require_local_judge=False skips only the judge check.
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

        temperature/top_p/seed/stop are passed to litellm only when set, so the
        provider's defaults otherwise apply. `seed` is what makes a run
        reproducible; without it the same prompt gives a different answer and a
        different score. Unsupported parameters are dropped, not refused, since
        litellm.drop_params is set package-wide.

        Routes through litellm's `openai` provider against
        `<ollama_api_base>/v1` rather than the `ollama` providers, which is the
        only route returning real logprobs, and so the only one that can feed
        the perplexity signal. Pitfall 7 in the package README.

        Raises OllamaUnavailableError or ModelNotPulledError for the two named
        failures; any other API error surfaces unchanged.
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

        raw_response is a litellm ModelResponse dump, which TLM.score() cannot
        read as-is; it is wrapped here into the shape it expects. perplexity,
        from generate(), is passed as the top-level key tlm reads, since tlm
        never derives it for a supplied response. The call runs in a worker
        thread because TLM.score() drives its own event loop. Pitfalls 4, 5
        and 7 in the package README.

        quality_preset/reasoning_effort/similarity_measure override
        self.config for this call only, so one client can serve several
        scoring depths. **advanced_tlm_config reaches the lower-level Config
        fields in ADVANCED_CONFIG_FIELDS; an unknown key raises ValueError
        before any network call. CONFIG_REFERENCE.md documents every
        parameter that is inert or broken.

        Raises EvalsNotSupportedError, RagNotSupportedError or
        JudgeCallFailedError on the three upstream defects; see their
        docstrings.
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
