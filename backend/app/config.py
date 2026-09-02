"""App-specific config: persona, trust labels, limits, log location. Generic
Ollama/tlm wiring lives in the tlm_local package.

Reads the environment, so it must be imported after tlm_local has loaded .env.
main.py does that; see tlm_local/config.py for why the order matters.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = REPO_ROOT / "logs" / "scores.jsonl"

# One scored request already fans out to about seven concurrent Ollama calls, so
# requests queue rather than compete; raising this without raising
# OLLAMA_NUM_PARALLEL only moves the contention. Documented in env.example.
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "1"))

# A latency bound rather than a safety one: the six judge calls all re-read the
# answer, so its length is multiplied through the scoring pass.
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))

# The question is unbounded otherwise, and it is not sent once: it goes to the
# generator, then into each of the six judge prompts.
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "4000"))

# No persona hardcoded: this backend is a generic example, not tied to a topic.
# env.example ships the demo's sport coach as an illustration.
DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# Placeholders, so the demo has a badge to render. They are not measurements and
# no measurement would transfer: a threshold belongs to a generator, a judge, a
# preset and a domain. scripts/calibrate.py derives them for yours, docs/SCORING.md
# says why each axis moves them. This 0.8 is also unrelated to the 0.8 in
# frontend/app.js, which is tlm's own EXPLAINABILITY_THRESHOLD.
TRUST_THRESHOLDS = {"reliable": 0.8, "needs_checking": 0.5}


def label_for_score(score: float) -> str:
    if score >= TRUST_THRESHOLDS["reliable"]:
        return "Reliable"
    if score >= TRUST_THRESHOLDS["needs_checking"]:
        return "Needs checking"
    return "Unreliable"
