import os
from dataclasses import FrozenInstanceError

import pytest

from tlm_local.config import LocalTLMConfig


def _tlm_resolved_judge_model() -> str:
    """The judge model tlm itself has cached, read the same way the property does.

    Deliberately not hardcoded: it depends on whether a .env was discoverable
    when tlm was first imported, which differs between a dev machine and CI.
    What the tests pin is that the property agrees with tlm, not what tlm picked.
    """
    from tlm.config.defaults import get_settings

    return str(get_settings().DEFAULT_MODEL)


ENV_VARS = (
    "OLLAMA_API_BASE",
    "GENERATOR_MODEL",
    "TLM_QUALITY_PRESET",
    "TLM_REASONING_EFFORT",
    "TLM_SIMILARITY_MEASURE",
    "DEFAULT_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test in this file controls its own env vars explicitly - clear
    them first so a real .env picked up in CI/local dev can't leak in.
    """
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestLocalTLMConfig:
    def test_defaults_when_no_env_vars_set(self):
        # given - no env vars set (see _clean_env fixture)

        # when
        config = LocalTLMConfig()

        # then
        assert config.ollama_api_base == "http://localhost:11434"
        assert config.generator_model == "ollama/ministral-3:3b"
        assert config.quality_preset == "medium"
        assert config.reasoning_effort is None
        assert config.similarity_measure is None

    def test_reads_ollama_api_base_from_env(self, monkeypatch):
        # given
        monkeypatch.setenv("OLLAMA_API_BASE", "http://example.internal:9999")

        # when
        config = LocalTLMConfig()

        # then
        assert config.ollama_api_base == "http://example.internal:9999"

    def test_reads_quality_preset_from_env(self, monkeypatch):
        # given
        monkeypatch.setenv("TLM_QUALITY_PRESET", "high")

        # when
        config = LocalTLMConfig()

        # then
        assert config.quality_preset == "high"

    def test_reads_reasoning_effort_and_similarity_measure_from_env(self, monkeypatch):
        # given
        monkeypatch.setenv("TLM_REASONING_EFFORT", "high")
        monkeypatch.setenv("TLM_SIMILARITY_MEASURE", "jaccard")

        # when
        config = LocalTLMConfig()

        # then
        assert config.reasoning_effort == "high"
        assert config.similarity_measure == "jaccard"

    def test_judge_model_reports_what_tlm_resolved_not_the_env_var(self, monkeypatch):
        """The whole point of the property, and the reason it does not read
        os.environ: tlm caches its Settings on first import, so a DEFAULT_MODEL
        exported afterwards is visible to os.environ but changes no judge call.
        Reporting the env var here would tell the caller a comfortable lie.
        """
        # given - tlm has already cached its settings by the time any test runs,
        # so this assignment is exactly the "too late" case
        monkeypatch.setenv("DEFAULT_MODEL", "ollama/some-model-tlm-never-saw:70b")

        # when
        config = LocalTLMConfig()

        # then
        assert config.judge_model != "ollama/some-model-tlm-never-saw:70b"
        assert config.judge_model == _tlm_resolved_judge_model()

    def test_judge_model_survives_default_model_being_unset(self):
        # given - DEFAULT_MODEL not set (see _clean_env fixture)

        # when
        config = LocalTLMConfig()

        # then - still tlm's cached value, not an invented local-looking default
        assert config.judge_model == _tlm_resolved_judge_model()

    def test_judge_is_local_tracks_the_resolved_model(self):
        # given
        config = LocalTLMConfig()

        # when / then
        assert config.judge_is_local == config.judge_model.startswith("ollama/")

    def test_judge_is_local_is_false_for_a_hosted_judge(self, monkeypatch):
        """Guards the failure mode that matters: tlm's own fallback judge is the
        hosted gpt-4.1-mini, so judge_is_local has to come out False for it.
        """
        # given
        monkeypatch.setattr(LocalTLMConfig, "judge_model", property(lambda self: "gpt-4.1-mini"), raising=True)

        # when
        config = LocalTLMConfig()

        # then
        assert config.judge_is_local is False

    def test_export_ollama_api_base_publishes_the_configured_host(self, monkeypatch):
        """Judge calls are built inside tlm and carry no api_base, so litellm
        resolves the host from this env var. Without the export, generation and
        scoring would target different servers.
        """
        # given
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
        config = LocalTLMConfig(ollama_api_base="http://gpu-box:11434")

        # when
        config.export_ollama_api_base()

        # then
        assert os.environ["OLLAMA_API_BASE"] == "http://gpu-box:11434"

    def test_export_ollama_api_base_overrides_a_differing_env_var(self, monkeypatch):
        """An explicit constructor argument beats the ambient env var, which is
        the opposite of the usual precedence and is the point: the two can only
        differ when the caller passed the value explicitly, and honoring the env
        var instead would send generation and scoring to different hosts. This
        also covers the case that broke the first version of this method, where
        .env had already populated the variable at import time.
        """
        # given
        monkeypatch.setenv("OLLAMA_API_BASE", "http://from-dotenv:11434")
        config = LocalTLMConfig(ollama_api_base="http://gpu-box:11434")

        # when
        config.export_ollama_api_base()

        # then
        assert os.environ["OLLAMA_API_BASE"] == "http://gpu-box:11434"

    def test_config_is_immutable(self):
        # given
        config = LocalTLMConfig()

        # when / then
        with pytest.raises(FrozenInstanceError):
            config.quality_preset = "high"
