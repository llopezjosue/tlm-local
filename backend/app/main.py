"""FastAPI backend: POST /chat generates an answer locally (persona set via
SYSTEM_PROMPT, see app.config) and attaches a TLM trustworthiness score. All
Ollama/tlm plumbing and error translation live in the tlm_local package;
this module only wires it up and owns the user-facing messages.
"""
from __future__ import annotations  # noqa: I001

import asyncio
import json
import logging
import time

# The I001 ignore above guards this order: tlm_local must be imported before
# app.config, which reads SYSTEM_PROMPT out of the environment that tlm_local's
# import loads .env into. Sorting alphabetically puts app first and silently
# drops the persona back to the generic default. tlm_local/pyproject.toml and
# scripts/ruff.toml carry the same ignore for the same reason.
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tlm_local import (
    VALIDATED_QUALITY_PRESETS,
    JudgeCallFailedError,
    LocalTLM,
    ModelNotPulledError,
    OllamaUnavailableError,
    ReasoningEffort,
)

from app.config import (
    LOG_FILE,
    MAX_CONCURRENT_CHATS,
    MAX_QUESTION_CHARS,
    MAX_TOKENS_LIMIT,
    REPO_ROOT,
    label_for_score,
)
from app.generator import build_messages

# Root stays at WARNING on purpose: tlm logs the full message payload of every
# judge call at INFO - the user's question and the model's answer, six times per
# scored request - so configuring the root logger at INFO puts all of it on
# stderr, and from there into journald, container logs or CI output. Only this
# app's own logger is raised. docs/SCORING.md's sub-score recipe is unaffected:
# it opts one tlm logger in by name.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("reliable_chat")
logger.setLevel(logging.INFO)

app = FastAPI(title="reliable-chat")
tlm_client = LocalTLM()

# Whatever the wrapper resolved from TLM_QUALITY_PRESET; a request may still
# override it. Checked at import so an unsupported preset fails once at startup
# rather than turning every call into a 400.
DEFAULT_QUALITY_PRESET = tlm_client.config.quality_preset
if DEFAULT_QUALITY_PRESET not in VALIDATED_QUALITY_PRESETS:
    raise RuntimeError(
        f"TLM_QUALITY_PRESET is {DEFAULT_QUALITY_PRESET!r}, which this app does not serve. "
        f"Set it to one of: {', '.join(VALIDATED_QUALITY_PRESETS)}."
    )

_chat_slots = asyncio.Semaphore(MAX_CONCURRENT_CHATS)  # see app.config

REASONING_EFFORTS = tuple(effort.value for effort in ReasoningEffort)
# embedding_small and embedding_large are excluded on purpose: they are the one
# code path in tlm that calls hosted OpenAI for embeddings, so accepting them
# from an HTTP request would let a caller punch a hole in this app's whole
# point. The remaining three are local. See tlm_local/CONFIG_REFERENCE.md.
LOCAL_SIMILARITY_MEASURES = ("jaccard", "code", "statement")


class ChatRequest(BaseModel):
    """Everything but `question` is optional and falls back to the wrapper's
    own configuration, which reads the environment. See env.example.
    """

    question: str
    quality_preset: str | None = None
    # Scoring knobs. reasoning_effort is the one that changes what you get back:
    # it is "none" by default, which leaves `explanation` a fixed string.
    reasoning_effort: str | None = None
    similarity_measure: str | None = None
    # Generation knobs. seed is what makes a run reproducible.
    max_tokens: int = 1024
    temperature: float | None = None
    seed: int | None = None


class ChatResponse(BaseModel):
    response: str
    trust_score: float
    trust_label: str
    duration_s: float
    quality_preset: str
    # Optional detail, absent when the underlying call did not produce it.
    explanation: str | None = None
    perplexity: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generator_model: str | None = None
    judge_model: str | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question cannot be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400, detail=f"The question must be at most {MAX_QUESTION_CHARS} characters."
        )

    quality_preset = payload.quality_preset or DEFAULT_QUALITY_PRESET
    if quality_preset not in VALIDATED_QUALITY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"quality_preset '{quality_preset}' is not supported here. "
                f"Tested and available values: {', '.join(VALIDATED_QUALITY_PRESETS)}."
            ),
        )
    # Validated here rather than left to tlm's pydantic Config, which would
    # surface a bad value as a 500 instead of telling the caller what to fix.
    if payload.reasoning_effort is not None and payload.reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(
            status_code=400,
            detail=f"reasoning_effort must be one of: {', '.join(REASONING_EFFORTS)}.",
        )
    if payload.similarity_measure is not None and payload.similarity_measure not in LOCAL_SIMILARITY_MEASURES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"similarity_measure must be one of: {', '.join(LOCAL_SIMILARITY_MEASURES)}. "
                "The embedding_* measures are refused: they are the one tlm code path that calls "
                "hosted OpenAI, which would break this app's fully-local guarantee."
            ),
        )
    if not 1 <= payload.max_tokens <= MAX_TOKENS_LIMIT:
        raise HTTPException(
            status_code=400, detail=f"max_tokens must be between 1 and {MAX_TOKENS_LIMIT}."
        )

    start = time.monotonic()  # before the wait: the caller experienced the queuing too
    try:
        async with _chat_slots:
            messages = build_messages(question)
            generation = await tlm_client.generate(
                messages,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                seed=payload.seed,
            )
            score_result = await tlm_client.score(
                generation.messages,
                generation.raw_response,
                perplexity=generation.perplexity,
                quality_preset=quality_preset,
                reasoning_effort=payload.reasoning_effort,
                similarity_measure=payload.similarity_measure,
            )
    except (OllamaUnavailableError, ModelNotPulledError, JudgeCallFailedError) as e:
        logger.error("Chat request failed: %s", e)
        raise HTTPException(status_code=503, detail=_user_facing_error(e)) from e
    except Exception:
        logger.exception("Unexpected error while handling /chat request")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating or scoring the answer.",
        ) from None

    trust_label = label_for_score(score_result.trust_score)
    duration_s = time.monotonic() - start
    _log_score(
        question,
        generation.answer,
        score_result.trust_score,
        trust_label,
        duration_s,
        quality_preset,
        explanation=score_result.explanation,
        perplexity=generation.perplexity,
    )

    usage = generation.usage or {}
    return ChatResponse(
        response=generation.answer,
        trust_score=score_result.trust_score,
        trust_label=trust_label,
        duration_s=round(duration_s, 1),
        quality_preset=quality_preset,
        explanation=score_result.explanation,
        perplexity=generation.perplexity,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        generator_model=tlm_client.config.generator_model,
        judge_model=tlm_client.config.judge_model,
    )


def _user_facing_error(e: Exception) -> str:
    if isinstance(e, OllamaUnavailableError):
        return "Could not reach the local Ollama server. Check that it is running (`ollama serve`)."
    if isinstance(e, ModelNotPulledError):
        bare_model = e.model.split("/", 1)[-1]
        return f"Model '{e.model}' is not available locally. Run `ollama pull {bare_model}`."
    if isinstance(e, JudgeCallFailedError):
        # Transient by nature (a judge call dropped, the model got evicted), so
        # the message says to retry rather than reporting a broken setup.
        return "A scoring call to the judge model failed. This is usually transient, please try again."
    return str(e)


def _log_score(
    question: str,
    answer: str,
    trust_score: float,
    trust_label: str,
    duration_s: float,
    quality_preset: str,
    *,
    explanation: str | None = None,
    perplexity: float | None = None,
) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "trust_score": trust_score,
        "trust_label": trust_label,
        "duration_s": round(duration_s, 1),
        "quality_preset": quality_preset,
        "explanation": explanation,
        "perplexity": perplexity,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Mounted last, after /chat: Starlette matches routes in declaration order, so
# this only ever serves paths /chat doesn't already own. Same origin as the
# API means the frontend needs no CORS configuration at all.
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "frontend"), html=True), name="frontend")
