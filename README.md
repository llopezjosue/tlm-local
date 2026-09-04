# tlm-local

[`cleanlab/tlm`](https://github.com/cleanlab/tlm) answers a question most LLM stacks leave open: **how much can you trust the answer you just got?** It hands the answer to a second model, interrogates it from six angles, folds in the generator's own confidence in its tokens, and returns a score between 0 and 1. You get to flag the answers worth checking instead of reading every one of them.

It is open source and needs no Cleanlab account, but everything around it assumes hosted APIs. **This wrapper runs it entirely locally against [Ollama](https://ollama.com)** : no API key, no account, nothing leaving your machine. Two `ollama pull` commands and a `pip install`, and you are scoring answers from your own models.

```python
generation, score = await LocalTLM().generate_and_score(messages)

print(generation.answer)
print(score.trust_score)   # 0.0-1.0
```

Go to [Running the showcase chat app](#running-the-showcase-chat-app) for a chat UI with a trust badge on every answer, or [Using the wrapper](#using-the-wrapper) to drop the package into your own project.

The repo has two parts:
- **[`tlm_local/`](tlm_local/)**, the deliverable: a standalone, separately-installable package that wraps `tlm` and fixes everything needed to run it against Ollama. Not tied to chatbots or to any topic, reusable on its own.
- **`backend/` + `frontend/`**, a small chat app that exercises the wrapper end to end. Not a product in its own right, and no hardcoded persona: `SYSTEM_PROMPT` sets the topic and tone.

## What running it locally actually takes

Enough that the wrapper is worth having. Seven non-obvious pitfalls, each found by testing rather than by reading documentation: scoring that crashes on every call until one litellm flag is set, a signal that silently never arrives because the request is routed to the wrong endpoint, an event-loop conflict, and a configuration failure that sends your prompts to a hosted model without saying so. [The package README](tlm_local/README.md) lists all seven and what each one costs you.

Worth knowing about, once you are running:

- [docs/SCORING.md](docs/SCORING.md) — how to read the number: what contributes at each preset, why a threshold does not transfer between configurations, and the two things the score will not tell you.
- [docs/SAFETY_NOTES.md](docs/SAFETY_NOTES.md) — the risks found by review and left unfixed on purpose. The first one is measured: a wrong answer can be pushed across the Reliable threshold by text placed in the question.
- [scripts/calibrate.py](scripts/calibrate.py) derives trust-label thresholds for **your** configuration. This repo ships none on purpose, because none would transfer: a cut-off belongs to a generator, a judge, a preset and a domain, and swapping the judge alone moved two answers from 0.842 to 0.342 here.

Not an official Cleanlab project, see [Relationship to cleanlab/tlm](#relationship-to-cleanlabtlm).

## Using the wrapper

```python
import asyncio

from tlm_local import LocalTLM


async def main():
    # reads OLLAMA_API_BASE / GENERATOR_MODEL / DEFAULT_MODEL / TLM_QUALITY_PRESET from env
    tlm_client = LocalTLM()

    messages = [{"role": "user", "content": "How long should I rest between squat sets?"}]
    generation, score = await tlm_client.generate_and_score(messages)

    print(generation.answer)
    print(score.trust_score)  # 0.0-1.0


asyncio.run(main())
```

Install it into your own project with `pip install -e ./tlm_local` (or point at this path from your own `requirements.txt`/`pyproject.toml`). See [`tlm_local/README.md`](tlm_local/README.md) for the full API (per-call `quality_preset` override, error types, configuration).

## Running the showcase chat app

Exercises the wrapper end-to-end with a real chat UI and a trust badge on every answer.

```bash
# 1. Install Ollama: https://ollama.com/download

# 2. Start it. See Configuration below for OLLAMA_NUM_PARALLEL, which is worth
#    setting if an accelerator serves your models and worth leaving alone if not.
ollama serve &

# 3. Pull the two local models
ollama pull ministral-3:3b
ollama pull qwen2.5:7b

# 4. Python environment. 3.11 is the floor: litellm imports `typing.NotRequired`,
#    which only exists from 3.11 on. Check yours with `python3 --version`; if it
#    is older, `uv python install 3.11` fetches one without admin rights and
#    without touching the system, or use python.org or your package manager.
#    Installs tlm_local in editable mode, plus FastAPI/uvicorn.
python3 -m venv .venv        # substitute your 3.11+ interpreter if python3 is older
source .venv/bin/activate    # bash/zsh
# .venv\Scripts\activate     # PowerShell
pip install -r requirements.txt

# 5. Configure
cp env.example .env
# edit .env - see Configuration below

# 6. Run (also serves the frontend, same origin, no CORS needed)
cd backend
uvicorn app.main:app --port 8000 --reload
```

If you run Ollama in a container, check it can actually reach your accelerator. A container that silently falls back to the CPU makes every scored request several times slower, and nothing reports it.

### Running the tests

67 tests, none of which need Ollama or a `.env` — the integration tests that do skip
themselves. Useful for checking the wrapper on a machine before pulling any model:

```bash
pip install -e "./tlm_local[dev]"          # pytest and ruff, not needed to run the app
python -m pytest tlm_local/tests backend/tests -q
```

Open **http://localhost:8000/** for the chat UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long should I rest between squat sets?", "quality_preset": "medium"}'
```

How long a scored request takes is a property of your hardware, not of this code: it is one generation plus six judge calls, all local. On a machine whose accelerator serves the models, roughly 60-90s at `quality_preset=medium`; on CPU alone, several minutes. `high` costs two to three times more either way. This is the full-local tradeoff, not a bug.

## Configuration

Everything lives in `.env` (copy `env.example` to start). No API key required anywhere.

Read by the `tlm_local` package, so these apply whether you run the showcase app or import the wrapper into your own project:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_API_BASE` | `http://localhost:11434` | Where your local Ollama server is running. |
| `GENERATOR_MODEL` | `ollama/ministral-3:3b` | Model that produces chat answers. |
| `DEFAULT_MODEL` | `ollama/qwen2.5:7b` | Model `tlm` uses as judge. Must be set before the process starts: `tlm` caches it on first import, and falls back to a hosted model when it sees nothing, which `LocalTLM` refuses to construct against. |
| `TLM_QUALITY_PRESET` | `medium` | Scoring depth: `medium` is self-reflection plus perplexity, `high` and `best` add consistency sampling. `base` and `low` are identical to `medium`. Overridable per call, or per request in the showcase app. |
| `TLM_REASONING_EFFORT` | unset (`none` for this workflow) | Whether the judge reasons before rating. Anything but `none` changes the prompt it receives, so it changes the score as well as the length of `explanation`. |
| `TLM_SIMILARITY_MEASURE` | unset (`statement` for this workflow) | How the consistency signal compares alternate answers. Inert below `high`, since consistency runs no completions at `medium`. |

Read by the showcase app only:

| Variable | Default | What it does |
|---|---|---|
| `SYSTEM_PROMPT` | a generic assistant | The chatbot's persona and topic. `env.example` ships this repo's demo coach as a working example; replace it, or delete the line. It reaches the judge as well as the generator. |
| `MAX_CONCURRENT_CHATS` | `1` | How many scored requests run at once; the rest queue. One request already fans out to about seven concurrent Ollama calls, so raising this without raising `OLLAMA_NUM_PARALLEL` only moves the contention. |
| `MAX_QUESTION_CHARS` | `4000` | Upper bound on a question's length. It is not sent once: it goes to the generator, then into each of the six judge prompts. |
| `MAX_TOKENS_LIMIT` | `4096` | Upper bound on a request's `max_tokens`. All six judge calls re-read the answer, so its length is multiplied through the scoring pass. |

`OLLAMA_NUM_PARALLEL` (set when starting `ollama serve`, not in `.env`) is worth understanding before copying a number from anywhere, this README included. Ollama serves one request at a time per loaded model by default, which serializes the six judge calls `tlm` sends concurrently.

Whether raising it helps depends on what is doing the work. Where an accelerator serves the model, it sits idle between those calls and raising the value fills the gap: measured here, a scored request went from 112-210s to about 60s at `OLLAMA_NUM_PARALLEL=4`. Where the CPU serves the model, a single inference already saturates the cores, so concurrent requests split the same compute and win nothing, while each one reserves its own KV cache. On a memory-tight machine that is a straight loss, and swapping during inference is far worse than waiting. Leave it alone, or try 2, and measure.

## Relationship to `cleanlab/tlm`

This is an independent, separate repository, **not an official Cleanlab project**. It depends on the `tlm` library as a normal PyPI dependency (`trustworthy-llm` on PyPI, `import tlm` in code) rather than vendoring or forking its source. `tlm` is licensed Apache-2.0, same as this repository; full credit for the trustworthiness-scoring approach belongs to the [Cleanlab](https://github.com/cleanlab/tlm) team.

## License

Apache-2.0, see [LICENSE](LICENSE).
