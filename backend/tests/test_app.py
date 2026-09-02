"""What the showcase app owns: trust labels and request validation.

Everything Ollama- and tlm-related belongs to tlm_local and is tested there.
Nothing here reaches a model, so the whole file runs with Ollama down.
"""

from __future__ import annotations

import pytest
from app.config import MAX_QUESTION_CHARS, MAX_TOKENS_LIMIT, TRUST_THRESHOLDS, label_for_score


class TestLabelForScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, "Reliable"),
            (0.8, "Reliable"),
            (0.79, "Needs checking"),
            (0.5, "Needs checking"),
            (0.49, "Unreliable"),
            (0.0, "Unreliable"),
        ],
    )
    def test_labels_each_band_including_its_boundary(self, score, expected):
        """The boundaries are the point: both thresholds are inclusive, so a
        score sitting exactly on one lands in the upper band.
        """
        # given / when / then
        assert label_for_score(score) == expected

    def test_bands_are_driven_by_the_thresholds_not_by_literals(self):
        """calibrate.py exists to replace these values, so the function has to
        follow them rather than repeat them.
        """
        # given / when / then
        assert label_for_score(TRUST_THRESHOLDS["reliable"]) == "Reliable"
        assert label_for_score(TRUST_THRESHOLDS["needs_checking"]) == "Needs checking"


class TestChatValidation:
    """All of these must fail before any model call: a 400 the caller can act
    on, rather than a 500 out of tlm's own validation minutes later.
    """

    @pytest.mark.parametrize("question", ["", "   ", "\n\t "])
    def test_rejects_a_blank_question(self, client, question):
        # given / when
        response = client.post("/chat", json={"question": question})

        # then
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_rejects_a_question_over_the_character_limit(self, client):
        # given - it is not sent once, but to the generator and then into each
        # of the six judge prompts
        # when
        response = client.post("/chat", json={"question": "a" * (MAX_QUESTION_CHARS + 1)})

        # then
        assert response.status_code == 400
        assert str(MAX_QUESTION_CHARS) in response.json()["detail"]

    @pytest.mark.parametrize("max_tokens", [0, -1, MAX_TOKENS_LIMIT + 1])
    def test_rejects_max_tokens_outside_the_bounds(self, client, max_tokens):
        # given / when
        response = client.post("/chat", json={"question": "hi", "max_tokens": max_tokens})

        # then
        assert response.status_code == 400
        assert "max_tokens" in response.json()["detail"]

    def test_rejects_a_quality_preset_this_app_does_not_serve(self, client):
        # given - valid for tlm, but never benchmarked on this project
        # when
        response = client.post("/chat", json={"question": "hi", "quality_preset": "best"})

        # then
        assert response.status_code == 400
        assert "medium, high" in response.json()["detail"]

    @pytest.mark.parametrize("measure", ["embedding_small", "embedding_large"])
    def test_refuses_the_embedding_similarity_measures(self, client, measure):
        """The one tlm code path that calls hosted OpenAI. Accepting it from an
        HTTP request would let a caller punch a hole in the local guarantee, so
        this is a security boundary rather than a taste.
        """
        # given / when
        response = client.post("/chat", json={"question": "hi", "similarity_measure": measure})

        # then
        assert response.status_code == 400
        assert "OpenAI" in response.json()["detail"]

    def test_rejects_an_unknown_reasoning_effort(self, client):
        # given / when
        response = client.post("/chat", json={"question": "hi", "reasoning_effort": "extreme"})

        # then
        assert response.status_code == 400
        assert "reasoning_effort" in response.json()["detail"]
