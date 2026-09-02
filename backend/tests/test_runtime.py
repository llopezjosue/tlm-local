"""Two behaviours that were bugs once, pinned so they cannot come back quietly.

Neither is visible in a response body, which is why neither had a test and why
both are worth one: a regression in either produces a working app that leaks or
overloads. Nothing here reaches a model.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading

from conftest import BACKEND


class TestJudgePayloadDoesNotReachTheLogs:
    """tlm logs the entire message payload of every judge call at INFO - the
    question, the answer, the parsed fields - six times per scored request
    (tlm/utils/completion_utils.py). Configuring the root logger at INFO, which
    is the obvious thing to write, puts all of it on stderr and from there into
    journald, container logs or CI output.

    Checked in a subprocess, and that is the whole point. logging.basicConfig is
    a no-op once the root logger has handlers, and pytest installs some, so an
    in-process check cannot tell a working configuration from a missing one: it
    passes either way. A fresh interpreter is the only place the assertion bites.
    """

    def test_importing_the_app_leaves_tlm_below_info(self):
        # given - a fresh interpreter, as uvicorn starts one
        probe = (
            "import sys; sys.path.insert(0, '.'); import app.main, logging; "
            "print(logging.getLogger('tlm.utils.completion_utils').isEnabledFor(logging.INFO))"
        )

        # when
        result = subprocess.run([sys.executable, "-c", probe], cwd=BACKEND, capture_output=True, text=True, timeout=120)

        # then
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("False"), (
            f"tlm's payload logger is enabled for INFO, so every question and answer "
            f"reaches stderr six times per request. stdout={result.stdout!r}"
        )


class TestScoredRequestsQueue:
    """One scored request already fans out to about seven concurrent Ollama
    calls, so two at once push judge calls toward timeouts - and a failed judge
    call takes the whole score down rather than degrading. Requests therefore
    wait instead of competing.
    """

    def test_never_more_than_max_concurrent_chats_in_flight(self, main_module, monkeypatch):
        # given - a generator slow enough for overlap to be observable
        in_flight = 0
        peak = 0
        log_threads: list[int] = []

        async def slow_generate(messages, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return _Generation()

        async def instant_score(*args, **kwargs):
            return _ScoreResult()

        monkeypatch.setattr(main_module.tlm_client, "generate", slow_generate)
        monkeypatch.setattr(main_module.tlm_client, "score", instant_score)
        monkeypatch.setattr(main_module, "_log_score", lambda *a, **k: log_threads.append(threading.get_ident()))

        # when - four callers arrive at once
        request = main_module.ChatRequest(question="2+2?")

        async def four_at_once():
            await asyncio.gather(*(main_module.chat(request) for _ in range(4)))

        asyncio.run(four_at_once())

        # then
        assert peak <= main_module.MAX_CONCURRENT_CHATS
        assert peak >= 1, "the stub never ran, so this asserted nothing"

    def test_the_score_log_is_written_off_the_event_loop(self, main_module, monkeypatch):
        """A blocking open() in a coroutine stalls every other request, and with
        requests queued there is more waiting behind it.
        """
        # given
        log_threads: list[int] = []

        async def fake_generate(messages, **kwargs):
            return _Generation()

        async def fake_score(*args, **kwargs):
            return _ScoreResult()

        monkeypatch.setattr(main_module.tlm_client, "generate", fake_generate)
        monkeypatch.setattr(main_module.tlm_client, "score", fake_score)
        monkeypatch.setattr(main_module, "_log_score", lambda *a, **k: log_threads.append(threading.get_ident()))

        # when
        asyncio.run(main_module.chat(main_module.ChatRequest(question="2+2?")))

        # then
        assert log_threads, "the logging stub never ran"
        assert log_threads[0] != threading.get_ident()


class _Generation:
    answer = "4"
    messages: list[dict] = []
    raw_response: dict = {}
    perplexity = 0.75
    usage = {"prompt_tokens": 1, "completion_tokens": 1}


class _ScoreResult:
    trust_score = 0.9
    raw: dict = {}
    explanation = None


class TestStartupRefusesUnusableLimits:
    """MAX_CONCURRENT_CHATS=0 builds a semaphore with no permits: the app starts,
    reports nothing, and every request hangs forever, which reads as a slow model
    rather than a misconfiguration.
    """

    def test_a_zero_limit_stops_the_app_from_starting(self):
        # given - a fresh interpreter, since the check runs at import
        probe = "import sys; sys.path.insert(0, '.'); import app.main"

        # when
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=BACKEND,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "MAX_CONCURRENT_CHATS": "0"},
        )

        # then
        assert result.returncode != 0
        assert "MAX_CONCURRENT_CHATS is 0" in result.stderr
