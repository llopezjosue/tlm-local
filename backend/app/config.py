"""App-specific config: presentation concerns (assistant persona, trust
label thresholds, score log location). Generic Ollama/tlm wiring lives in
the tlm_local package.

Reads SYSTEM_PROMPT from the environment - relies on tlm_local already
having loaded .env by the time this module is imported (main.py imports
tlm_local first; see tlm_local/config.py for why that ordering matters).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = REPO_ROOT / "logs" / "scores.jsonl"

# No persona hardcoded here: this backend is a generic local-tlm chat example,
# not tied to any one topic. Set SYSTEM_PROMPT in .env to give it one (the
# sport/fitness coach used in this repo's demo is just the example value in
# env.example).
# One scored request already sends about seven concurrent calls to Ollama (one
# generation plus the six self-reflection judge calls tlm fires together), so it
# oversubscribes the OLLAMA_NUM_PARALLEL=4 this repo recommends on its own. Two
# such requests at once do not halve throughput, they push judge calls into
# timeouts, and a failed judge call is the one thing tlm cannot degrade from:
# it takes the whole score down with it (JudgeCallFailedError). Requests
# therefore queue rather than compete. Raise this only alongside
# OLLAMA_NUM_PARALLEL, and re-measure.
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "1"))

# Upper bound on a request's max_tokens. Not a safety limit so much as a
# latency one: the six judge calls all re-read the answer, so generation length
# is multiplied through the whole scoring pass.
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# Provisional - scripts/calibrate.py replaces these with measured values.
#
# The 0.8 here is this app's presentation choice and is unrelated to the 0.8 in
# frontend/app.js, which is tlm's own EXPLAINABILITY_THRESHOLD. They are equal
# by coincidence, so recalibrating these must not touch that one.
TRUST_THRESHOLDS = {"reliable": 0.8, "needs_checking": 0.5}


def label_for_score(score: float) -> str:
    if score >= TRUST_THRESHOLDS["reliable"]:
        return "Reliable"
    if score >= TRUST_THRESHOLDS["needs_checking"]:
        return "Needs checking"
    return "Unreliable"
