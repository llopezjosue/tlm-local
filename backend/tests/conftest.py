"""Makes `app` importable and keeps these tests independent of the machine.

app.main builds a LocalTLM at import, which resolves a judge model and raises
when it is not local. That depends on whether a .env was discoverable when tlm
was first imported, so importing app.main for real would pass here and fail on
a machine without one. The client is stubbed instead: what these tests cover is
this app's own validation and labelling, not the wrapper, which
tlm_local/tests covers directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def main_module():
    """app.main with its LocalTLM replaced, imported once."""
    import tlm_local

    stub = MagicMock(name="LocalTLM")
    stub.return_value.config.quality_preset = "medium"
    stub.return_value.config.generator_model = "ollama/test-generator"
    stub.return_value.config.judge_model = "ollama/test-judge"
    original = tlm_local.LocalTLM
    tlm_local.LocalTLM = stub
    try:
        import app.main as main

        yield main
    finally:
        tlm_local.LocalTLM = original


@pytest.fixture
def client(main_module):
    from fastapi.testclient import TestClient

    return TestClient(main_module.app)
