# tlm-local

A small wrapper around [`cleanlab/tlm`](https://github.com/cleanlab/tlm) that fixes the rough edges of running it fully locally against [Ollama](https://ollama.com): no API key, no external network call.

Not an official Cleanlab project. `tlm` (PyPI: `trustworthy-llm`) is Apache-2.0 licensed; so is this wrapper.

## Why this exists

Getting `tlm` to work correctly and safely against a local Ollama server surfaced several real, non-obvious pitfalls, each one found by testing, not by reading documentation:

1. **Import-order hazard, and it fails open**: `tlm` builds and caches its own settings object (which reads the judge model from `DEFAULT_MODEL`) the moment `tlm` is first imported in a process. Load your `.env` after that point and it's silently too late for the rest of the process's life. `tlm_local` loads `.env` as an import-time side effect, so importing it first guarantees correct ordering. What makes this worth guarding rather than just documenting: `tlm`'s own fallback judge is the hosted `gpt-4.1-mini`, with `api_key` auto-filled from `OPENAI_API_KEY`, so losing the ordering does not raise, it quietly starts sending your prompts and answers to OpenAI. `LocalTLM` therefore asserts at construction that the judge model `tlm` actually resolved is `ollama/`-prefixed, and raises `JudgeModelNotLocalError` otherwise. `LocalTLMConfig.judge_model` reports what `tlm` resolved, not what the environment says, because after caching those two can disagree and the environment is the one that lies.
2. **A crash disguised as a missing feature, only partly curable**: `tlm`'s self-reflection/consistency judge calls always request a `logprobs` parameter. Ollama's chat API rejects unrecognized parameters outright, and `tlm`'s own error handling doesn't check for that specific failure: it crashes with an unrelated `AttributeError` deep in its scoring pipeline, on every single call, unless `litellm.drop_params = True` is set first. Be clear about what that fixes: `drop_params` removes one *trigger*, not the defect. Any judge-side failure still takes the whole score down, which `score()` can name as `JudgeCallFailedError` but not prevent. See that exception's docstring.
3. **The judge's Ollama host is not yours to pass**: only `generate()` builds its own litellm call, so only it can carry `api_base`. Judge and consistency calls are built inside `tlm`, which never sets `api_base` on them (verified: they go out with `api_base=None`), and `tlm`'s own `Config.api_base` is not honored on that path either. litellm then resolves the host from the `OLLAMA_API_BASE` env var, so `LocalTLM(LocalTLMConfig(ollama_api_base="http://gpu-box:11434"))` would generate on the remote box and score on localhost. `LocalTLM` exports the configured host into that variable at construction to close the gap.
4. **The exact `response` shape `TLM.score()` expects**: `.score()` accepts either a real `openai.types.chat.ChatCompletion` object or a dict. A `litellm.ModelResponse` (what you get calling Ollama through litellm) is neither: pass it directly and it silently fails to parse. It has to be wrapped as `{"chat_completion": <the raw dict>}` first.
5. **Sync-inside-async deadlock**: `TLM.score()`/`.create()` are synchronous and call `run_until_complete()` on their own event loop internally. Call that directly from an already-running event loop (e.g. a FastAPI route) and it raises `RuntimeError: This event loop is already running`. It has to run in a worker thread instead.
6. **Indistinguishable Ollama failures, and a same-name class trap**: litellm raises different exception types depending on the provider prefix (`APIConnectionError` for both failure modes via `ollama`/`ollama_chat`; `NotFoundError` vs `InternalServerError` via `openai`), so you have to inspect the message text to tell "down" from "model not pulled" apart in both cases. Their real common ancestor is `openai.APIError`, *not* `litellm.exceptions.APIError`: litellm defines its own, different class also named `APIError`, which just happens to share the name but isn't actually in the inheritance chain (`issubclass()` against it is `False`; confirmed by comparing `id()`/`__module__` of the two). Caught this only by explicitly testing the error paths, not just the happy path.
7. **Real logprobs exist, but not through the route you'd expect**: `tlm`'s perplexity signal needs real per-token log-probabilities. Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) genuinely returns them, verified directly with real per-token data. But litellm's `ollama`/`ollama_chat` providers both target Ollama's *native* API instead (confirmed via `litellm._turn_on_debug()`), where `logprobs` isn't a recognized option and silently gets dropped, regardless of provider prefix. `generate()` routes through litellm's `openai` provider pointed at `<ollama_api_base>/v1` instead. tlm never derives the value itself for a supplied response, so `generate()` computes it and `score()` passes it through as the literal `"perplexity"` key tlm reads. Note that field name is a misnomer: it wants a probability in `[0, 1]`, not a perplexity. See [`../docs/SCORING.md`](../docs/SCORING.md).

None of this is exotic or Ollama-specific misconfiguration. It's what you hit on the very first real attempt to wire `tlm` up to a fully local stack.

## Usage

```python
from tlm_local import LocalTLM

tlm_client = LocalTLM()  # reads OLLAMA_API_BASE / GENERATOR_MODEL / DEFAULT_MODEL / TLM_QUALITY_PRESET from env

messages = [{"role": "user", "content": "How long should I rest between squat sets?"}]
generation, score = await tlm_client.generate_and_score(messages)

print(generation.answer)
print(score.trust_score)  # 0.0-1.0
```

Or generate and score separately if you need to do something with the raw answer in between (e.g. apply your own trust-label thresholds, as the `backend/` example app in this repo does). Pass `generation.perplexity` through explicitly so `score()` still gets the real logprobs-derived signal:

```python
generation = await tlm_client.generate(messages)
score = await tlm_client.score(generation.messages, generation.raw_response, perplexity=generation.perplexity)
```

`quality_preset` can also be overridden per call (e.g. from a request parameter, a frontend selector) instead of being fixed at construction time. A single `LocalTLM` instance can serve different scoring depths:

```python
from tlm_local import VALIDATED_QUALITY_PRESETS  # ("medium", "high") - the only ones benchmarked for score quality here

generation, score = await tlm_client.generate_and_score(messages, quality_preset="high")
```

### The full scoring API

`score()`/`generate_and_score()` expose tlm's whole tunable surface, not just
`quality_preset`. Everything beyond `messages`/`raw_response` is optional and falls
back to `self.config` or tlm's own defaults:

```python
score = await tlm_client.score(
    generation.messages,
    generation.raw_response,
    perplexity=generation.perplexity,
    quality_preset="high",  # base|low|medium|high|best
    reasoning_effort="high",  # none|low|medium|high - explanation length, not score value
    similarity_measure="statement",  # jaccard|embedding_small|embedding_large|code|statement
    constrain_outputs=None,  # list[str], for multiple-choice/classification workflows
    evals=None,  # list[Eval] - see "Known limitations" below
)
```

Rarer, lower-level `tlm.config.schema.Config` fields (the ones each `quality_preset`
normally derives automatically, e.g. `num_consistency_completions`) are reachable
through a validated escape hatch instead of one named parameter per field:

```python
await tlm_client.score(messages, raw_response, quality_preset="medium", num_consistency_completions=2)
```

An unknown keyword here (typo or otherwise) raises `ValueError` immediately, before any
network call, rather than failing deep inside tlm's own validation. See
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) for which of these parameters are inert,
broken, or do something other than what upstream's docs say.

## Configuration

Reads from the environment (see `env.example` at the repo root) or pass a `LocalTLMConfig` explicitly:

| Env var | Default | Notes |
|---|---|---|
| `OLLAMA_API_BASE` | `http://localhost:11434` | |
| `GENERATOR_MODEL` | `ollama/ministral-3:3b` | Passed to `.score()`/`.create()` directly |
| `DEFAULT_MODEL` | `ollama/qwen2.5:7b` | Read by `tlm` itself for the judge role, see the import-order note above: this must be set before first use |
| `TLM_QUALITY_PRESET` | `medium` | `base`\|`low`\|`medium`\|`high`\|`best` |
| `TLM_REASONING_EFFORT` | unset (tlm default) | `none`\|`low`\|`medium`\|`high` |
| `TLM_SIMILARITY_MEASURE` | unset (tlm default) | `jaccard`\|`embedding_small`\|`embedding_large`\|`code`\|`statement` |

Every one of these can also be overridden per call, as shown above.

## Known limitations

Each of these is a property of `trustworthy-llm==0.0.3`, not a decision taken here.

- **`evals` and RAG scoring are broken upstream.** A non-empty `evals` raises `EvalsNotSupportedError`; passing `context=` raises `RagNotSupportedError`, because tlm injects its own RAG evals and hits the same pipeline-factory bug. Both reproduce with a bare `TLM()` call, no wrapper involved.
- **Only the QA workflow is reachable.** Classification, binary classification and structured-output scoring are not wrapped, and neither is `TLM.create()`, so `num_reference_completions` is accepted but inert.
- **`TLM` leaks an event loop per instance.** `TLM.__init__` adopts the thread's event loop or creates one, and tlm never closes it. `score()` builds a `TLM` per call, so this package creates and disposes of that loop itself; without it a long-running server leaked three file descriptors per scored request. Measured: 60 descriptors over 20 calls before, none after.
- **A failed judge call takes the whole score down.** tlm reads `.per_field_metadata` on a failure object that has no such attribute, so any judge-side exception aborts scoring. `score()` names it `JudgeCallFailedError`; there is no partial score, no retry and no backoff. Retry at your call site.
- **The score carries no decomposition**, and `quality_preset` changes its composition rather than its strictness, so a threshold calibrated at `medium` does not hold at `high`. Both are covered in [`../docs/SCORING.md`](../docs/SCORING.md), along with why `tlm`'s per-model weights fall back to a generic bucket for any local model.
