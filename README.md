# tlm-local

A Python wrapper for running [`cleanlab/tlm`](https://github.com/cleanlab/tlm)'s trustworthiness scoring fully locally against [Ollama](https://ollama.com): no API key, no external network call, no Cleanlab account. `tlm` is an open-source library; getting it to work correctly and safely against local models surfaced several real, non-obvious pitfalls (silent crashes, wrong configuration, missing scoring signals) that this wrapper fixes.

That "no external network call" is enforced, not just intended. Left to itself, `tlm` falls back to a hosted `gpt-4.1-mini` judge with an `api_key` taken from `OPENAI_API_KEY`, so a misplaced `.env` silently ships your prompts and answers to OpenAI instead of failing. `LocalTLM` checks the judge model `tlm` actually resolved at construction and raises rather than let that happen. See pitfall 1 in [the package README](tlm_local/README.md).

**What this covers, so the name does not overpromise.** `tlm`'s QA scoring path: score an answer you already have, or generate and score in one call. Not all of `tlm`. Its RAG scoring and custom `evals` are broken upstream in `trustworthy-llm==0.0.3`, and raise a typed error here rather than failing obscurely; the classification, binary-classification and structured-output workflows are not wrapped, and neither is `TLM.create()`. The full list is in [Known limitations](tlm_local/README.md#known-limitations-not-yet-fixed).

The repo has two parts:
- **[`tlm_local/`](tlm_local/)**, the actual deliverable: a standalone, separately-installable Python package that wraps `tlm` and fixes everything needed to run it safely against Ollama. See [its README](tlm_local/README.md) for the full list of pitfalls it handles. No knowledge of chatbots or any topic, fully reusable on its own, in any project.
- **`backend/` + `frontend/`**, a small chat app used purely as a showcase/testbed to exercise the wrapper end-to-end. Not a product in its own right. No hardcoded persona: the assistant's topic/tone is set entirely by `SYSTEM_PROMPT` (see Configuration below).

Not an official Cleanlab project, see [Relationship to cleanlab/tlm](#relationship-to-cleanlabtlm).

Worth knowing about:
- [docs/SCORING.md](docs/SCORING.md) explains what the trust score actually measures in a given configuration: how the five sub-scores are weighted and renormalized, the six criteria the judge model is asked to apply, and what each `quality_preset` catches or misses. Read this before tuning thresholds or changing presets.
- [scripts/calibrate.py](scripts/calibrate.py) runs a question set through the wrapper and, once you have labelled the answers, sweeps candidate thresholds and reports precision/recall plus the count of wrong answers that would be shown as trustworthy. The trust labels this repo ships are still provisional guesses, so if you are deciding whether the scores are good enough for your own use, start here. See [scripts/README.md](scripts/README.md).

## Using the wrapper

```python
from tlm_local import LocalTLM

tlm_client = LocalTLM()  # reads OLLAMA_API_BASE / GENERATOR_MODEL / DEFAULT_MODEL / TLM_QUALITY_PRESET from env

messages = [{"role": "user", "content": "How long should I rest between squat sets?"}]
generation, score = await tlm_client.generate_and_score(messages)

print(generation.answer)
print(score.trust_score)  # 0.0-1.0
```

Install it into your own project with `pip install -e ./tlm_local` (or point at this path from your own `requirements.txt`/`pyproject.toml`). See [`tlm_local/README.md`](tlm_local/README.md) for the full API (per-call `quality_preset` override, error types, configuration).

## Running the showcase chat app

Exercises the wrapper end-to-end with a real chat UI and a trust badge on every answer.

```bash
# 1. Install Ollama: https://ollama.com/download (macOS, Windows, Linux),
#    or `brew install ollama` on macOS/Linux with Homebrew.

# 2. Start it with OLLAMA_NUM_PARALLEL=4 set (see Configuration below for why)
OLLAMA_NUM_PARALLEL=4 ollama serve &      # macOS/Linux (bash/zsh)
# $env:OLLAMA_NUM_PARALLEL=4; ollama serve  # Windows (PowerShell)

# 3. Pull the two local models
ollama pull ministral-3:3b
ollama pull qwen2.5:7b

# 4. Python environment. 3.11 is the floor: litellm imports `typing.NotRequired`,
#    which only exists from 3.11 on. Installs tlm_local in editable mode, plus
#    FastAPI/uvicorn.
python3.11 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 5. Configure
cp env.example .env
# edit .env - see Configuration below

# 6. Run (also serves the frontend, same origin, no CORS needed)
cd backend
uvicorn app.main:app --port 8000 --reload
```

On Apple Silicon, run Ollama natively rather than inside Docker: Docker Desktop on macOS runs in a Linux VM and can't reach the GPU through Metal, so inference is much slower. Not an issue on Linux/Windows setups without that constraint.

Open **http://localhost:8000/** for the chat UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long should I rest between squat sets?", "quality_preset": "medium"}'
```

A scored request takes roughly 60-90s at `quality_preset=medium` (more at `high`). This is a full-local tradeoff, not a bug.

## Configuration

Everything lives in `.env` (copy `env.example` to start). No API key required anywhere. Read directly by `tlm_local`, so this applies whether you're running the showcase app or using the wrapper standalone in your own project:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_API_BASE` | `http://localhost:11434` | Where your local Ollama server is running. |
| `GENERATOR_MODEL` | `ollama/ministral-3:3b` | Model that produces chat answers. |
| `DEFAULT_MODEL` | `ollama/qwen2.5:7b` | Model `tlm` uses as judge (self-reflection/consistency). Must be set before the process starts: `tlm` caches it on first import. |
| `TLM_QUALITY_PRESET` | `medium` | Scoring depth: `medium` (self-reflection + perplexity) or `high` (adds consistency-sampling, more latency, stricter/recalibrated scores). Overridable per-call in the wrapper, or per-request in the showcase app (chat UI dropdown / API's `quality_preset` field). |

`SYSTEM_PROMPT` is specific to the showcase app (`backend/`), not the wrapper: it sets the chatbot's persona/topic. `env.example` ships this repo's demo persona (a sport/fitness coach) as a working example; replace it with anything, or delete the line for a plain generic assistant.

`OLLAMA_NUM_PARALLEL=4` (set when starting `ollama serve`, not in `.env`) matters a lot for latency: without it, a scored request can take 2-3x longer, because Ollama processes one request at a time per loaded model by default, which serializes the six judge calls `tlm` sends concurrently.

## Relationship to `cleanlab/tlm`

This is an independent, separate repository, **not an official Cleanlab project**. It depends on the `tlm` library as a normal PyPI dependency (`trustworthy-llm` on PyPI, `import tlm` in code) rather than vendoring or forking its source. `tlm` is licensed Apache-2.0, same as this repository; full credit for the trustworthiness-scoring approach belongs to the [Cleanlab](https://github.com/cleanlab/tlm) team.

## License

Apache-2.0, see [LICENSE](LICENSE).
