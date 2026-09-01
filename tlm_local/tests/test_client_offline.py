"""Offline coverage for the generate()/score() happy path.

Every test in this file runs with Ollama down. `tlm`'s TLM class and
litellm's acompletion are replaced at tlm_local.client's own boundary
(never inside tlm), so the pitfall fixes this package exists for stay
pinned on a machine with no local model server, which is every CI machine.
The integration tests in test_client.py cover the same path for real but
skip themselves when Ollama is unreachable, which left every fix below
asserted nowhere.

Nothing here carries a requires_ollama marker, and nothing here should
grow one.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from openai import APIError

from tlm_local import (
    JudgeCallFailedError,
    JudgeModelNotLocalError,
    LocalTLM,
    LocalTLMConfig,
    ModelNotPulledError,
    OllamaUnavailableError,
    RagNotSupportedError,
)

# Fixed instead of read from the environment: the .env this repo ships sets
# GENERATOR_MODEL and OLLAMA_API_BASE, and CI has no .env at all, so any
# assertion on what generate() sends has to pin its own values.
OLLAMA_API_BASE = "http://ollama.test:11434"
GENERATOR_MODEL = "ollama/ministral-3:3b"
BARE_GENERATOR_MODEL = "ministral-3:3b"

ANSWER = "4"
TRUST_SCORE = 0.87
# exp(0.0) = 1.0 and exp(-1.0) ~= 0.368, so the mean token probability is ~0.684
LOGPROBS = {"content": [{"logprob": 0.0}, {"logprob": -1.0}]}
EXPECTED_PERPLEXITY = 0.6839


def _completion_payload(*, logprobs: dict | None = None) -> dict:
    """The dict shape generate() reads back out of a litellm ModelResponse,
    and the shape score() takes as its raw_response argument.
    """
    return {
        "id": "chatcmpl-offline",
        "model": BARE_GENERATOR_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER},
                "logprobs": logprobs,
                "finish_reason": "stop",
            }
        ],
    }


class _FakeModelResponse:
    """Stand-in for litellm's ModelResponse: generate() only ever calls
    .model_dump() on what acompletion returns.
    """

    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


class _FakeOllamaAPIError(APIError):
    """A genuine openai.APIError (the ancestor generate() catches, not the
    same-named litellm.exceptions.APIError) built without an httpx.Request,
    since only its message text is ever inspected.
    """

    def __init__(self, message: str):
        Exception.__init__(self, message)
        self.message = message


@pytest.fixture
def tlm_client(monkeypatch) -> LocalTLM:
    """A client with a pinned config and no construction-time judge check.

    require_local_judge=False keeps this file independent of ambient state:
    with this repo's .env, tlm resolves a local judge and the check passes,
    but on a machine where no .env is discoverable tlm falls back to the
    hosted gpt-4.1-mini and construction would raise. Setting
    OLLAMA_API_BASE to the value the config already carries makes the
    constructor's export a no-op, so nothing leaks into other tests.
    """
    monkeypatch.setenv("OLLAMA_API_BASE", OLLAMA_API_BASE)
    config = LocalTLMConfig(ollama_api_base=OLLAMA_API_BASE, generator_model=GENERATOR_MODEL)
    return LocalTLM(config, require_local_judge=False)


@pytest.fixture
def tlm_class(monkeypatch) -> MagicMock:
    """Replaces tlm's TLM class inside tlm_local.client, so scoring reaches
    no judge model. The Config object score() built is recorded as
    tlm_class.call_args.kwargs["config"].
    """
    fake = MagicMock(name="TLM")
    fake.return_value.score.return_value = {"trustworthiness_score": TRUST_SCORE, "explanation": "looks right"}
    monkeypatch.setattr("tlm_local.client.TLM", fake)
    return fake


@pytest.fixture
def judge_call(tlm_class) -> MagicMock:
    """The TLM.score() call itself, which is what most assertions inspect."""
    return tlm_class.return_value.score


@pytest.fixture
def acompletion(monkeypatch) -> AsyncMock:
    """Replaces litellm.acompletion inside tlm_local.client, so generation
    reaches no Ollama server.
    """
    fake = AsyncMock(return_value=_FakeModelResponse(_completion_payload(logprobs=LOGPROBS)))
    monkeypatch.setattr("tlm_local.client.litellm.acompletion", fake)
    return fake


@pytest.fixture
def messages() -> list[dict]:
    return [{"role": "user", "content": "2+2?"}]


@pytest.fixture
def raw_response() -> dict:
    return _completion_payload(logprobs=LOGPROBS)


class TestLitellmDropParams:
    def test_importing_tlm_local_enables_drop_params(self):
        """litellm's own default is False, and with it False every single judge
        call crashes: tlm always sends a `logprobs` key Ollama rejects, and
        tlm's handling of that rejection dies on an unrelated AttributeError.
        config.py sets this as an import-time side effect, so importing
        anything from the package is what has to be enough.
        """
        # given / when - tlm_local was imported at the top of this module

        # then
        assert litellm.drop_params is True


class TestScoreResponsePayload:
    """The exact `response` structure TLM.score() is handed. A litellm
    ModelResponse dump is neither of the two types it accepts, so getting
    this wrong means silent misparsing rather than an error.
    """

    async def test_wraps_the_raw_response_in_a_chat_completion_key(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given / when
        await tlm_client.score(messages, raw_response)

        # then
        payload = judge_call.call_args.kwargs["response"]
        assert payload == {"chat_completion": raw_response}
        assert payload["chat_completion"] is raw_response

    async def test_adds_perplexity_as_a_top_level_key(self, tlm_client, judge_call, messages, raw_response):
        """tlm never derives perplexity itself for a caller-supplied response;
        it only reads a literal top-level "perplexity" key.
        """
        # given / when
        await tlm_client.score(messages, raw_response, perplexity=0.42)

        # then
        payload = judge_call.call_args.kwargs["response"]
        assert payload == {"chat_completion": raw_response, "perplexity": 0.42}

    async def test_omits_perplexity_when_it_is_none(self, tlm_client, judge_call, messages, raw_response):
        # given / when - no perplexity available, e.g. a completion with no logprobs
        await tlm_client.score(messages, raw_response, perplexity=None)

        # then - absent, not present-and-null: tlm's scoring math renormalizes
        # over the signals actually supplied
        assert "perplexity" not in judge_call.call_args.kwargs["response"]


class TestScoreHappyPath:
    async def test_returns_the_trust_score_and_the_full_tlm_result(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given / when
        result = await tlm_client.score(messages, raw_response)

        # then
        assert result.trust_score == pytest.approx(TRUST_SCORE)
        assert result.raw == judge_call.return_value

    async def test_forwards_the_messages_and_generator_model_to_the_judge(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given / when
        await tlm_client.score(messages, raw_response)

        # then - the model scored is the one that generated, prefix intact:
        # unlike generate(), tlm builds its own litellm call from this
        kwargs = judge_call.call_args.kwargs
        assert kwargs["messages"] is messages
        assert kwargs["model"] == GENERATOR_MODEL

    async def test_per_call_overrides_and_advanced_fields_reach_the_tlm_config(
        self, tlm_client, tlm_class, messages, raw_response
    ):
        """Per-call overrides let one client serve several scoring depths, and
        **advanced_tlm_config is the escape hatch to the lower-level fields a
        preset normally derives. Both only work if they land on the Config.
        """
        # given / when
        await tlm_client.score(messages, raw_response, quality_preset="high", num_consistency_completions=2)

        # then
        config = tlm_class.call_args.kwargs["config"]
        assert config.quality_preset == "high"
        assert config.num_consistency_completions == 2

    async def test_runs_the_judge_off_the_calling_event_loop_thread(
        self, tlm_client, judge_call, messages, raw_response
    ):
        """TLM.score() is synchronous and calls run_until_complete() on its own
        loop internally, so calling it from a coroutine raises "RuntimeError:
        This event loop is already running". A worker thread has no running
        loop yet, which is the whole fix.
        """
        # given
        judge_thread = {}

        def _record_thread(**kwargs):
            judge_thread["ident"] = threading.get_ident()
            return {"trustworthiness_score": TRUST_SCORE}

        judge_call.side_effect = _record_thread

        # when
        await tlm_client.score(messages, raw_response)

        # then
        assert judge_thread["ident"] != threading.get_ident()


class TestGenerateRouting:
    """generate() deliberately routes through litellm's `openai` provider
    pointed at Ollama's OpenAI-compatible endpoint. The `ollama`/
    `ollama_chat` providers target Ollama's native API instead, where the
    logprobs request is silently dropped, which costs the perplexity signal.
    """

    async def test_uses_the_openai_provider_with_the_bare_model_name(self, tlm_client, acompletion, messages):
        # given / when
        await tlm_client.generate(messages)

        # then
        model = acompletion.call_args.kwargs["model"]
        assert model == f"openai/{BARE_GENERATOR_MODEL}"
        assert not model.startswith("ollama/")

    async def test_targets_the_openai_compatible_endpoint_with_a_non_empty_api_key(
        self, tlm_client, acompletion, messages
    ):
        # given / when
        await tlm_client.generate(messages)

        # then - the key is a dummy Ollama ignores, but litellm's openai
        # provider refuses to build the call without one
        kwargs = acompletion.call_args.kwargs
        assert kwargs["api_base"] == f"{OLLAMA_API_BASE}/v1"
        assert kwargs["api_base"].endswith("/v1")
        assert kwargs["api_key"]

    async def test_asks_for_logprobs(self, tlm_client, acompletion, messages):
        # given / when
        await tlm_client.generate(messages)

        # then
        assert acompletion.call_args.kwargs["logprobs"] is True

    async def test_returns_the_answer_and_a_perplexity_derived_from_the_logprobs(
        self, tlm_client, acompletion, messages
    ):
        # given / when
        generation = await tlm_client.generate(messages)

        # then
        assert generation.answer == ANSWER
        assert generation.messages is messages
        assert generation.perplexity == pytest.approx(EXPECTED_PERPLEXITY, abs=1e-3)

    async def test_translates_an_api_error_into_a_typed_local_error(self, tlm_client, acompletion, messages):
        # given - what litellm's openai provider raises for an unpulled model
        acompletion.side_effect = _FakeOllamaAPIError("litellm.NotFoundError: model 'ministral-3:3b' not found")

        # when / then
        with pytest.raises(ModelNotPulledError):
            await tlm_client.generate(messages)

    async def test_translates_a_connection_failure_into_ollama_unavailable(self, tlm_client, acompletion, messages):
        # given
        acompletion.side_effect = _FakeOllamaAPIError(
            "litellm.InternalServerError: OpenAIException - Connection error."
        )

        # when / then
        with pytest.raises(OllamaUnavailableError):
            await tlm_client.generate(messages)


class TestGenerateAndScore:
    async def test_passes_the_generated_perplexity_through_to_the_judge(
        self, tlm_client, acompletion, judge_call, messages
    ):
        """The end-to-end reason pitfalls 4 and 2 both exist: real logprobs are
        only available on the openai-provider route, and the value computed
        from them only reaches tlm as an explicit top-level key.
        """
        # given / when
        generation, score = await tlm_client.generate_and_score(messages)

        # then
        assert generation.answer == ANSWER
        assert score.trust_score == pytest.approx(TRUST_SCORE)
        assert judge_call.call_args.kwargs["response"]["perplexity"] == pytest.approx(EXPECTED_PERPLEXITY, abs=1e-3)


class TestScoreErrorTranslation:
    async def test_context_raises_rag_not_supported_error(self, tlm_client, messages, raw_response):
        """Unmocked on purpose: a non-None context makes tlm inject its own RAG
        evals, which crash while the pipeline is being built, before any judge
        call goes out. Nothing has to be stubbed for this to reproduce.
        """
        # given / when / then
        with pytest.raises(RagNotSupportedError):
            await tlm_client.score(messages, raw_response, context="Rest 2-3 minutes.", quality_preset="base")

    async def test_a_failed_judge_call_raises_judge_call_failed_error(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given - tlm turns any judge-side exception into a CompletionFailure and
        # then reads an attribute that object does not have
        judge_call.side_effect = AttributeError("'CompletionFailure' object has no attribute 'per_field_metadata'")

        # when / then
        with pytest.raises(JudgeCallFailedError):
            await tlm_client.score(messages, raw_response)

    async def test_an_unrelated_attribute_error_still_surfaces(self, tlm_client, judge_call, messages, raw_response):
        # given - the translation is matched narrowly so real bugs are not
        # relabelled as judge failures
        judge_call.side_effect = AttributeError("'NoneType' object has no attribute 'strip'")

        # when / then
        with pytest.raises(AttributeError, match="strip"):
            await tlm_client.score(messages, raw_response)

    async def test_a_none_trustworthiness_score_raises_judge_call_failed_error(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given - every judge call failed without crashing the pipeline, so tlm
        # averaged zero surviving signals
        judge_call.return_value = {"trustworthiness_score": None}

        # when / then - not the bare TypeError float(None) would raise
        with pytest.raises(JudgeCallFailedError):
            await tlm_client.score(messages, raw_response)

    async def test_a_nan_trustworthiness_score_raises_judge_call_failed_error(
        self, tlm_client, judge_call, messages, raw_response
    ):
        # given
        judge_call.return_value = {"trustworthiness_score": float("nan")}

        # when / then - a NaN would otherwise survive into the caller's JSON as
        # an out-of-spec literal
        with pytest.raises(JudgeCallFailedError):
            await tlm_client.score(messages, raw_response)


class TestJudgeModelCheck:
    """tlm resolves its judge model from DEFAULT_MODEL on first import and
    falls back to the hosted gpt-4.1-mini, silently, when it cannot see one.
    LocalTLM checks at construction rather than letting every scoring call
    leave the machine.
    """

    def test_refuses_to_construct_when_the_resolved_judge_is_not_local(self, monkeypatch):
        # given - tlm's own fallback
        monkeypatch.setattr(LocalTLMConfig, "judge_model", property(lambda self: "gpt-4.1-mini"))

        # when / then
        with pytest.raises(JudgeModelNotLocalError, match="gpt-4.1-mini"):
            LocalTLM()

    def test_constructs_when_the_resolved_judge_is_an_ollama_model(self, monkeypatch):
        # given
        monkeypatch.setattr(LocalTLMConfig, "judge_model", property(lambda self: "ollama/qwen2.5:7b"))

        # when
        client = LocalTLM()

        # then
        assert client.config.judge_is_local is True

    def test_require_local_judge_false_allows_a_hosted_judge(self, monkeypatch):
        # given - the deliberate opt-out, for scoring against a hosted judge
        monkeypatch.setattr(LocalTLMConfig, "judge_model", property(lambda self: "gpt-4.1-mini"))

        # when
        client = LocalTLM(require_local_judge=False)

        # then
        assert client.config.judge_is_local is False
