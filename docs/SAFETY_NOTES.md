# Safety notes

Structural risks found by review that are **not** fixed in the code, because fixing them
means a design decision rather than a patch. Each entry says what the exposure is, why it
was left alone, and what closing it would cost.

Scope: this is a local POC. Several entries only matter once it stops being one, and they
are marked as such. What is *not* a problem here, checked rather than assumed: no `eval`,
`exec`, `pickle`, `subprocess` or shell use anywhere; no SQL; no hardcoded secret (the
`api_key="ollama"` in `client.py` is a dummy Ollama ignores, required non-empty by
litellm); no deserialisation of untrusted data. `.env` is gitignored and no key of any
kind is needed at runtime.

## 1. The trust score can be steered by the text it is scoring

The user's question reaches the judge. `LocalTLM.score()` passes the whole `messages`
list to `TLM.score()`, and each of the six self-reflection templates embeds it in the
judge prompt alongside the answer (see [SCORING.md](SCORING.md) for the six criteria).
The judge is an instruction-following model reading attacker-supplied text.

So a question can carry text aimed at the judge rather than at the generator — the
familiar prompt-injection shape, except the target is the component whose whole job is to
say whether the output can be trusted. A successful steer does not produce a wrong
answer; it produces a wrong answer **labelled Reliable**, which is worse, because the
badge is the reason a reader would stop checking.

Untested here. Worth knowing it is untested rather than assuming a 7B judge resists it:
[SCORING.md](SCORING.md) already documents that the six angles all interrogate the same
model, so they do not vote independently, and a single systematic misjudgement is
reproduced six times rather than averaged away.

Not fixed because there is no good fix at this layer. Sanitising the question would
change what the generator sees, and delimiter-based defences are known to be weak. The
honest mitigations are outside the wrapper: treat the score as advisory, and do not build
an automatic action on top of a Reliable label.

## 2. Every question and answer is written to disk, permanently

`_log_score()` in `backend/app/main.py` appends the question, the answer, the score and
the timestamp to `logs/scores.jsonl` on every request. No rotation, no retention limit,
no redaction, no opt-out.

That is deliberate for a calibration testbed and would be wrong in anything user-facing.
`logs/` is gitignored, so it does not leave the machine on its own, but it accumulates
indefinitely and holds whatever users typed.

Related and already fixed: `logging.basicConfig(level=logging.INFO)` used to configure
the *root* logger, and tlm logs the full message payload of every judge call at INFO, so
the same content also reached stderr six times per request. Root is now WARNING and only
the app's own logger is raised.

## 3. Nothing has a timeout, and requests now queue behind each other

No `timeout` is passed to `litellm.acompletion()`, and `TLM.score()` runs through
`asyncio.to_thread`, which cannot be cancelled. A hung Ollama therefore holds a request
open indefinitely, and a client that disconnects leaves the worker thread running.

This got sharper with the concurrency fix: requests now queue on a semaphore
(`MAX_CONCURRENT_CHATS`, default 1), so one stuck request blocks every later one instead
of just itself. Queuing is still the right trade — the alternative was judge calls
timing out and taking whole scores down — but the failure mode moved rather than
disappeared.

Not fixed because a timeout needs a number, and the right number is not knowable without
the calibration run: a scored request legitimately takes anywhere from about a minute to
several, depending on whether an accelerator or the CPU serves the models, and the
hardware has not been fixed yet. Guessing low would abort good requests.

## 4. Error handling depends on upstream strings that are not pinned

`trustworthy-llm>=0.0.3` has no upper bound, and three of the typed errors are recognised
by substring: `per_field_metadata`, `unexpected keyword argument 'model'`, `not found`. A
`0.0.4` that rephrases any of those turns `JudgeCallFailedError` and `RagNotSupportedError`
back into raw `AttributeError` and `TypeError`, and nothing fails at install time to say
so.

The safety-relevant part is that the degradation is silent and in the wrong direction: the
errors exist precisely to stop an upstream defect surfacing as an unrelated crash.

Not fixed here because pinning is a scope call for this repo, not a code fix — see the
project's own position on production tooling in a local POC.

## 5. Things that only matter beyond a local POC

- **No authentication or rate limiting on `/chat`.** Fine bound to localhost, not fine the
  moment uvicorn is started with `--host 0.0.0.0`. The queue makes it worse as a denial-of-
  service target, since one slot is shared by everyone.
- **`SYSTEM_PROMPT=""` yields an empty persona**, not the default: `os.environ.get` treats
  set-but-empty as set. Deleting the line works, emptying it does not.
- **Automating this at scale would multiply the exposures above**, not just the cost. Each
  scored request is seven model calls, one durable log line containing user text, and one
  trust label. A batch runner built on `scripts/calibrate.py` inherits all three, and the
  label is the one that would propagate into downstream decisions.

## What was fixed instead

Low-risk, high-confidence corrections applied with tests after each: the stderr leak of
questions and answers, the unguarded import order in `backend/app/main.py`, the unbounded
question length, and the unchained 500. See the git history for each.
