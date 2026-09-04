import os
import socket
from unittest.mock import MagicMock, patch

import pytest

from tlm_local import Eval, EvalsNotSupportedError, LocalTLM
from tlm_local.client import KNOWN_QUALITY_PRESETS, _mean_token_probability


def _ollama_reachable(host: str = "localhost", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


requires_ollama = pytest.mark.skipif(not _ollama_reachable(), reason="Ollama server not reachable on localhost:11434")


class TestMeanTokenProbability:
    def test_returns_mean_token_probability_when_logprobs_present(self):
        # given - two tokens, both with logprob=0.0 (i.e. probability 1.0)
        raw_response = {"choices": [{"logprobs": {"content": [{"logprob": 0.0}, {"logprob": 0.0}]}}]}

        # when
        result = _mean_token_probability(raw_response)

        # then
        assert result == pytest.approx(1.0)

    def test_averages_across_tokens_with_different_probabilities(self):
        # given - exp(0) = 1.0, exp(-1) ~= 0.368, mean ~= 0.684
        raw_response = {"choices": [{"logprobs": {"content": [{"logprob": 0.0}, {"logprob": -1.0}]}}]}

        # when
        result = _mean_token_probability(raw_response)

        # then
        assert result == pytest.approx(0.6839, abs=1e-3)

    def test_returns_none_when_logprobs_is_null(self):
        # given - what litellm's ollama/ollama_chat providers actually
        # return: the key is present but the value is None
        raw_response = {"choices": [{"logprobs": None}]}

        # when / then
        assert _mean_token_probability(raw_response) is None

    def test_returns_none_when_response_shape_is_unexpected(self):
        # given
        raw_response = {}

        # when / then
        assert _mean_token_probability(raw_response) is None

    def test_returns_none_when_content_list_is_empty(self):
        # given
        raw_response = {"choices": [{"logprobs": {"content": []}}]}

        # when / then
        assert _mean_token_probability(raw_response) is None


class TestScoreDoesNotLeakEventLoops:
    """tlm never closes the loop TLM() takes, and score() builds one per call,
    so an unowned loop leaks descriptors until a long-lived server runs out.
    """

    @pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="no /dev/fd to count descriptors with")
    async def test_repeated_scoring_holds_its_descriptor_count(self):
        # given - a client whose judge call is stubbed, so this stays offline
        client = LocalTLM(require_local_judge=False)
        messages = [{"role": "user", "content": "hi"}]
        raw_response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        fake = MagicMock()
        fake.return_value.score.return_value = {"trustworthiness_score": 0.9}

        def open_descriptors() -> int:
            return len(os.listdir("/dev/fd"))

        with patch("tlm_local.client.TLM", fake):
            await client.score(messages, raw_response)  # warm the thread pool
            before = open_descriptors()

            # when
            for _ in range(10):
                await client.score(messages, raw_response)

            # then
            assert open_descriptors() == before


class TestLocalTLMScoreValidation:
    """Pre-flight validation in score() - runs before any network call, so
    these fail fast and need no Ollama. require_local_judge=False for the
    reason the tlm_client fixture in test_client_offline.py gives; the judge
    check itself is covered by TestJudgeModelCheck.
    """

    async def test_rejects_unknown_quality_preset(self):
        # given
        client = LocalTLM(require_local_judge=False)
        messages = [{"role": "user", "content": "hi"}]
        raw_response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}

        # when / then
        with pytest.raises(ValueError, match="Unknown quality_preset"):
            await client.score(messages, raw_response, quality_preset="ultra")

    async def test_rejects_unknown_advanced_config_field(self):
        # given
        client = LocalTLM(require_local_judge=False)
        messages = [{"role": "user", "content": "hi"}]
        raw_response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}

        # when / then
        with pytest.raises(ValueError, match="Unknown tlm Config field"):
            await client.score(messages, raw_response, not_a_real_field=1)


@requires_ollama
class TestLocalTLMIntegration:
    """Real calls against a locally running Ollama server. Requires the
    models configured via GENERATOR_MODEL/DEFAULT_MODEL (env, or
    tlm_local's own defaults) to already be pulled - skipped automatically
    if Ollama isn't reachable at all, but a missing *model* still surfaces
    as a real ModelNotPulledError failure here, not a skip.
    """

    @pytest.mark.parametrize("preset", KNOWN_QUALITY_PRESETS)
    async def test_score_succeeds_for_every_known_quality_preset(self, preset):
        # given
        client = LocalTLM()
        messages = [{"role": "user", "content": "2+2?"}]
        raw_response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "4"}}]}

        # when
        result = await client.score(messages, raw_response, quality_preset=preset)

        # then
        assert 0.0 <= result.trust_score <= 1.0

    async def test_generate_and_score_returns_a_scored_answer(self):
        # given
        client = LocalTLM()
        messages = [{"role": "user", "content": "2+2?"}]

        # when
        generation, score = await client.generate_and_score(messages)

        # then
        assert generation.answer
        assert 0.0 <= score.trust_score <= 1.0

    async def test_score_with_evals_raises_evals_not_supported_error(self):
        """Documents a real upstream tlm==0.0.3 bug (see EvalsNotSupportedError's
        docstring) rather than letting a confusing internal TypeError leak
        through - this test should start failing (in a good way) the day a
        newer tlm release fixes it, which is the point of pinning it here."""
        # given
        client = LocalTLM()
        messages = [{"role": "user", "content": "hi"}]
        raw_response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        evals = [Eval(name="politeness", criteria="Is the response polite?", response_identifier="Response")]

        # when / then
        with pytest.raises(EvalsNotSupportedError):
            await client.score(messages, raw_response, quality_preset="base", evals=evals)
