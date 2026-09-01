"""Builds the chat messages sent to the generator model. The persona lives
entirely in app.config.SYSTEM_PROMPT (configurable via the SYSTEM_PROMPT env
var) - nothing sport-specific or otherwise topic-specific is hardcoded here.
"""
from __future__ import annotations

from app.config import SYSTEM_PROMPT


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
