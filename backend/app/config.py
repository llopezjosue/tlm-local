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
DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# Provisional - Phase 4 tunes these empirically against a real test question set.
TRUST_THRESHOLDS = {"reliable": 0.8, "needs_checking": 0.5}


def label_for_score(score: float) -> str:
    if score >= TRUST_THRESHOLDS["reliable"]:
        return "Reliable"
    if score >= TRUST_THRESHOLDS["needs_checking"]:
        return "Needs checking"
    return "Unreliable"
