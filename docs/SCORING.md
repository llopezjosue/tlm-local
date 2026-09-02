# What the `trust_score` actually measures, per configuration

This page answers one question: **for a given configuration, what is the final score
made of, and what does it catch or miss?**

It does not describe the parameters one by one, that is what
[`tlm_local/CONFIG_REFERENCE.md`](../tlm_local/CONFIG_REFERENCE.md) is for. This page
covers the combinations. Everything below is verified against the installed
`trustworthy-llm==0.0.3` source, with file paths given each time.

## In one table

| Config | What actually contributes to the score | What that measures |
|---|---|---|
| `medium` (current) + perplexity | self-reflection **70 %**, perplexity **30 %** | the judge's opinion on factual correctness, plus the generator's confidence in its own tokens |
| `medium` without perplexity | self-reflection **100 %** | the judge's opinion, alone |
| `high` / `best` + perplexity | self-reflection **47 %**, consistency **32 %**, perplexity **20 %**, indicator **1 %** | same, plus how stable the answer is under resampling |
| `base` / `low` | **strictly identical to `medium`** on the `score()` path | same as `medium`, these presets reduce neither self-reflection nor cost |

`prompt_eval` has a weight of 0 outside RAG, so it never contributes here.

## The formula

`tlm/utils/scoring/trustworthiness_scoring_utils.py:48-152`. The final score is a
**weighted average of 5 sub-scores**, using the weights from
`tlm/config/score_weights.py`:

| Sub-score | Raw weight |
|---|---|
| `self_reflection` | 0.47 |
| `consistency` | 0.32 |
| `perplexity` | 0.20 |
| `indicator` | 0.01 |
| `prompt_eval` | 0.00 |

**`perplexity` is a misnomer upstream, and the distinction decides what you may
feed it.** tlm fills that field for its own completions with
`get_parsed_answer_tokens_confidence` (`tlm/utils/parse_utils.py:145-161`), which
averages `exp(logprob)` per token and clips to 1.0, using a helper documented as
converting "to probability 0-1 scale". The field therefore holds a **mean token
probability in [0, 1]**. A real perplexity is `exp(-mean logprob)`, always at
least 1, and supplying one would put the value far outside the range this
weighted average assumes, with nothing to validate it. `tlm_local` supplies the
mean probability, checked to match tlm's own computation numerically.

Two fallbacks are what land us on this particular set of weights:

1. **By workflow**: `COMPONENT_SCORE_WEIGHTS` only contains the keys `DEFAULT` and
   `RAG`. Our workflow is QA, absent from the table, so we fall back to `DEFAULT`
   (line 161).
2. **By model**: the table explicitly calibrates GPT-3.5/4/4o/4o-mini, o1-preview and
   three Claude 3 models. Our Ollama models are absent, so we fall back to the
   `DEFAULT_MODEL` bucket, which is `gpt-4.1-mini` (line 164, and
   `tlm/config/models.py:144`). The weights above are therefore `gpt-4.1-mini`'s,
   applied to `ministral-3:3b` and `qwen2.5:7b`.

**The key mechanism is renormalization.** A sub-score that was not computed is `NaN`,
it gets dropped from the list, and `np.average` recomputes the mean over the remaining
weights only (lines 133-150). An absent signal is therefore not penalized, it is
ignored, and **the remaining signals see their relative weight go up**.

Hence the 70/30 at `medium`: only `self_reflection` (0.47) and `perplexity` (0.20)
survive, summing to 0.67, so 0.47/0.67 = 70 % and 0.20/0.67 = 30 %.

### Verified against this project's own data

An A/B test run on this project measured the following on a fixed answer whose
self-reflection score was 1.0:

| Injected perplexity | Measured `trust_score` | Predicted by the formula |
|---|---|---|
| absent | 1.0 | 1.0 |
| 0.99 | 0.997 | (0.47x1.0 + 0.20x0.99) / 0.67 = 0.9970 |
| 0.05 | 0.716 | (0.47x1.0 + 0.20x0.05) / 0.67 = 0.7164 |

All three match to three decimal places. The weights, the renormalization and the 70/30
split are therefore confirmed empirically, not just read off the source.

## The 6 self-reflection criteria

This is the core of the score, and it is absent from every other doc in the repo. The
judge (`DEFAULT_MODEL`, currently `qwen2.5:7b`) receives the same question/answer pair
**6 times**, from 6 different angles. Table
`SELF_REFLECTION_TEMPLATES_BY_WORKFLOW[WorkflowType.QA]`,
`tlm/templates/reflection_completion_templates.py:905-913`:

| Template | What the judge is asked | Scale | Mapping |
|---|---|---|---|
| `ReflectionCertainty` | estimated probability that the answer is correct | 0-100 | `x/100` |
| `ReflectionKnowledgeGap` | produce the evidence, justify why it can be trusted, then rate its own confidence | 0-10 | `x/10` |
| `ReflectionArgument` | build the adversarial case (why the answer might be wrong), then rate | 0-100 | `x/100` |
| `ReflectionBinaryCorrectness` | true/false, instructed to verify every claim rigorously | True/False | logprobs of the choice token |
| `ReflectionTrustworthiness` | same question, graded rating | 1-5 | 5→1.0, 4→0.75, 3→0.5, 2→0.25, 1→0 |
| `ReflectionCorrectness` | verify the accuracy of each claim and look for tricky aspects in the question | 0-10 | `x/10` |

The 6 ratings are mapped into 0-1 (`tlm/templates/score_mapping.py`) then aggregated by
**plain nan-safe mean**, with no weighting between templates
(`tlm/utils/scoring/self_reflection_scoring_utils.py:39`).

Details that matter for interpretation:

- Every prompt presents the answer as coming from an "untrustworthy" or "unreliable AI
  Assistant". The adversarial bias is deliberate and baked into the prompts.
- `ReflectionBinaryCorrectness` is the **only** one of the 6 that uses the choice
  token's logprobs (`use_logprobs=True`, line 314). It therefore produces a continuous
  confidence rather than a hard 0/1. The other five are discrete ratings.
- **The criterion is always factual correctness.** None of the 6 looks at relevance to
  the question, completeness, tone, or adherence to the `SYSTEM_PROMPT`. None consults
  an external source. The score measures what the judge believes it knows.

### The direct consequence for false negatives

All 6 angles interrogate the same model. On a **systematic error** (the generator is
wrong in the same way with the same confidence every time), the 6 prompts reproduce the
same misjudgement 6 times, and averaging corrects nothing. This is exactly the false
negative this project hit in practice: an answer whose headline figure was plausible,
but whose very next sentence confused grams with kilograms, still scored 0.842,
"Reliable". It also explains why moving the judge up to `qwen2.5:14b` caught it
(0.342, "Unreliable") while sampling more would not.

## Config by config

### `medium` (the project's current default)

`num_consistency_completions: 0`, 6 self-reflection templates
(`tlm/config/presets.py:129`).

- **Composition**: self-reflection 70 %, perplexity 30 %.
- **Catches**: errors that `qwen2.5:7b` can recognize as wrong, plus a penalty when the
  generator hesitates at the token level.
- **Misses**: systematic errors shared by generator and judge, and anything that is not
  a question of factual correctness.
- **Measured cost**: roughly 59-90s per scored answer on the accelerator-backed
  machine these figures come from; several times that on CPU alone.

### `high` and `best`

`num_consistency_completions: 4` (`high`) or `8` (`best`), 6 self-reflection templates
unchanged (`tlm/config/presets.py:127-128`).

- **Composition**: all 4 signals are active, weights sum to 1.00, so there is no
  renormalization. self-reflection 47 %, consistency 32 %, perplexity 20 %,
  indicator 1 %.
- **What consistency measures**: 4 (or 8) extra answers are generated and compared
  against the reference answer by similarity. For QA the default measure is `statement`
  (`tlm/types/base.py`, `SimilarityMeasure.for_workflow()`), so an LLM judge comparing
  the claims, plus a jaccard term weighted 0.05. That is **stability**, not accuracy.
- **Why scores drop, mechanically**: it is not "stricter". Self-reflection's weight goes
  from 70 % down to 47 %, and 33 % of the score shifts onto a different question ("does
  the model answer the same way if you run it again?"). A correct but freely worded
  answer can therefore lose points without any error having been detected. This is the
  origin of the 0.9 to 0.72 drop measured on this project.
- **Practical consequence**: adopting `high` means recalibrating the `label_for_score()`
  thresholds, not just accepting more latency. The current thresholds (0.8 / 0.5) are
  tuned against a 70/30 composition that no longer exists at `high`.
- **`indicator`** is a plain exact-string-equality ratio
  (`tlm/utils/scoring/indicator_scoring_utils.py:12-15`), so it is almost always 0 on
  free text. Its 0.01 weight is consistent with how useless it is here.

### `base` and `low`

**On the `score()` path, `base`, `low` and `medium` are strictly the same pipeline.**
Verified field by field:

- All 6 self-reflection templates run in all three cases, including for `base`, which
  asks for 0. See caveat 2 below for the mechanism.
- `num_consistency_completions` is 0 in all three cases
  (`tlm/config/presets.py:129-136`).
- `min_self_reflection_completions`, `min_consistency_completions` and
  `alternate_reference_temperature`, the only other fields these presets set, are
  **declared and read by no component** (verified by grep across the whole package).
  They are dead fields, like `self_reflection_temperature`.
- `min_reference_completions` is the only one of that family actually consumed
  (`pipeline/factory.py:84`), but only by `ReferenceCompletionGenerator`, which is not
  instantiated when you supply the answer: the `score()` path uses
  `ReferenceCompletionFormatter` instead (`pipeline/factory.py:79-89`).
- `base` and `low` force `reasoning_effort=none`, which is already the default for QA
  (caveat 1), so it has no effect.

In other words, choosing `base` or `low` over `medium` changes neither the score nor the
cost. On the same input, any gap observed between these three presets is run-to-run
variance, not a preset effect. That is consistent with the measurements taken here, where
`base` and `low` both return exactly 1.0.

Worth noting, to remove an ambiguity: `ObservedConsistencyCompletionGenerator` is added
to the pipeline **unconditionally** (`pipeline/factory.py:50-58`), including at
`medium`. It is its `count=0` that makes it inert, not its absence. Consistency does
therefore run on the `score()` path as soon as the count is non-zero, it is not reserved
for `.create()`.

## Three caveats verified in the source

None of these three are documented anywhere in the repo, and two contradict
[`CONFIG_REFERENCE.md`](../tlm_local/CONFIG_REFERENCE.md).

### 1. `reasoning_effort` is `none` by default, under all 5 presets

`tlm/config/base.py:68-70` sets `reasoning_default = ReasoningEffort.NONE` for every
workflow other than `STRUCTURED_OUTPUT_SCORING`. No preset overrides it, except `low`
and `base`, which redundantly set it back to `NONE`.

But each self-reflection template carries **two prompt variants** (see each `create()`
in `reflection_completion_templates.py`): `_PROMPT_TEMPLATE` with no `<think>` block,
and `_PROMPT_TEMPLATE_WITH_REASONING`, which asks the judge to reason step by step
**before** producing its rating. With `reasoning_effort=none`, the variant without
`<think>` is what gets sent.

**So today, the judge rates with no deliberation at all.** It sees the question/answer
pair and emits a number directly.

This qualifies the claim in `CONFIG_REFERENCE.md` that `reasoning_effort` "does not
change the score value, only the explanation length": true of the word counter, false of
the mechanism. Changing `reasoning_effort` changes the **prompt sent to the judge**, so
it changes the distribution of its output. On a 7B model, an adversarial reasoning step
before rating is very likely to move the number.

It is also the cheapest untested lever on this project: unlike `high` (4x the calls) or
`qwen2.5:14b` (2x the model), raising `reasoning_effort` does not change the number of
calls, only their content. Worth testing against the two known failure cases.

### 2. `num_self_reflection_completions` cannot be set to 0 or 1

`tlm/components/completions/self_reflection_completion_generator.py:25-26`:

```python
if num_completions > 1:
    completion_templates = completion_templates[:num_completions]
```

The guard is `> 1`. So `-1` (medium, low, high, best), `0` (base) and `1` all fall
through untruncated, and the **6 templates run in all three cases**.

This is the mechanism behind the surprise already recorded empirically in the repo
("`quality_preset=base` does NOT disable self-reflection"): `base` does ask for
`num_self_reflection_completions: 0` (`tlm/config/presets.py:135`), but 0 is not `> 1`,
so nothing gets truncated. It is also impossible to request a single template.

### 3. Four config fields are dead, one of them exposed by `tlm_local`

`self_reflection_temperature` is declared in `tlm/config/base.py:37` and
`tlm/config/schema.py:44`, and is **read by no component** (verified by grep across the
whole package: no occurrence outside the config declarations). The self-reflection
generator hardcodes the temperature to `temperature=0.0`
(`self_reflection_completion_generator.py:50`).

It is the only one of the 8 `ADVANCED_CONFIG_FIELDS` in that state, so `tlm_local`
exposes it as a usable knob even though passing it produces neither an error nor an
effect. The other 7 are properly consumed in `tlm/pipeline/factory.py`.

Same situation for three fields `tlm_local` does not expose, but which appear in the
preset tables and can suggest that `base`/`low` do something:
`min_self_reflection_completions`, `min_consistency_completions` and
`alternate_reference_temperature` are declared and read nowhere.

## What a trust threshold belongs to

Everything above is why this repository ships **no calibrated thresholds**, and why
none would help you if it did.

A cut-off like "0.8 means reliable" is not a property of `tlm`, of this wrapper, or
of trustworthiness scoring in general. It is a property of one configuration
measured on one set of questions. Five things move it, and the first two move it
more than the preset everyone thinks of:

| What you change | Why the score moves | Measured here |
|---|---|---|
| **The judge model** | It is the score. At `medium`, ~70 % of the number is this model's opinion, and a bigger model does not merely score higher, it discriminates differently | Two answers containing real errors scored 0.842 and 0.769 with `qwen2.5:7b`, and 0.342 and 0.525 with `qwen2.5:14b` |
| **The generator model** | Perplexity is the generator's confidence in its own tokens, ~30 % of the score at `medium`. A different generator has a different confidence profile on the same answer | see the formula above |
| **`quality_preset`** | It changes which signals exist, so the surviving weights are renormalized. Not added strictness, a different question | A good answer went from ~0.9 at `medium` to ~0.72 at `high` |
| **`reasoning_effort`** | Anything but `none` sends the judge a prompt that reasons before rating, which changes its output distribution (caveat 1 above) | untested |
| **Your questions and your domain** | The judge scores what it believes it knows. Its reliability is not uniform across subjects, so a threshold calibrated on one domain does not carry to another | — |

So the values in `backend/app/config.py` are placeholders that let the demo render
a badge, not measurements, and `docs/test_questions.md` ships an example question
set matched to the example persona rather than a benchmark.

Getting real numbers for **your** configuration is what
[`scripts/calibrate.py`](../scripts/calibrate.py) is for; `scripts/README.md`
covers the workflow. It records the whole configuration on every row and groups by
it, so two judges swept out of one file do not average into a threshold that fits
neither.

## Checking this yourself on a given answer

**The per-signal sub-scores are not retrievable from the API.** They are computed, then
dropped: each score component adds its array to the pipeline's `ExecutionContext`
(`tlm/components/scores/*.py`), and `tlm/inference.py:60-70` builds the returned
`InferenceResult` from only six fields, none of which is a sub-score:

| Field on `ScoreResult.raw` | What you actually get here |
|---|---|
| `trustworthiness_score` | the final number, nothing about its composition |
| `response` | the answer you supplied |
| `usage` | always `{}` on the `score()` path: `response_assembly.py:61-75` fills it only for `InferenceType.PROMPT`, and supplying a response forces `SCORE` (`pipeline/factory.py:77`) |
| `metadata` | only ever `per_field_score`, and only for the structured-output workflow, which this wrapper cannot reach |
| `evals` | broken upstream, see the `EvalsNotSupportedError` note |
| `explanation` | one of three canned strings by default, see below |

`explanation` is worth understanding before relying on it
(`tlm/utils/explainability_utils.py`). Above `EXPLAINABILITY_THRESHOLD` it is the fixed
string `"Did not find a reason to doubt trustworthiness."`. Below it, `tlm` tries to
surface the `<think>` text of the *lowest-scoring* judge template, but with
`reasoning_effort=none`, the default for QA (caveat 1 above), no template ever produces a
`<think>` block, so it degrades to `"Cannot verify that this response is correct."` or
`"The prompt/response appear atypical or vague."`. Raising `reasoning_effort` is what
makes this field carry actual judge reasoning. That is a second, independent reason to
test that lever.

**The recipe that does work: turn on `tlm`'s own INFO logging.**
`_generate_total_scores` logs all five sub-score arrays before combining them
(`tlm/utils/scoring/trustworthiness_scoring_utils.py:80-85`):

```python
import logging
logging.getLogger("tlm.utils.scoring.trustworthiness_scoring_utils").setLevel(logging.INFO)
logging.basicConfig()  # or your app's handler

generation, score = await tlm_client.generate_and_score(messages, quality_preset="high")
# stderr now carries:
#   -- Consistency scores: [...]
#   -- Indicator scores: [...]
#   -- Self reflection scores: [...]
#   -- Perplexity scores: [...]
#   -- Prompt eval scores: [...]
```

This is how to confirm how a `high` score decomposes, and how to tell a genuine judge
doubt from a parse failure, which also scores 0.5. It is a log-scraping workaround, not
an API. Getting these values as data would require patching `tlm` to carry them onto
`InferenceResult`.

**Beware of 0.5, it is the silent fallback value.** The mappings do `.get(x, 0.5)` on an
unrecognized value (`score_mapping.py`), and a parse failure scores
`SELF_REFLECTION_PARSE_FAILURE_SCORE = 0.5` (`tlm/config/defaults.py:63`). A 7B judge
that does not respect the expected XML format therefore yields 0.5 without raising
anything. A score near 0.5 can mean "the judge is unsure" or "the judge could not answer
in the required format", and `trust_score` alone cannot distinguish them.

## See also

- [`tlm_local/CONFIG_REFERENCE.md`](../tlm_local/CONFIG_REFERENCE.md): `tlm`'s 17
  parameters, one by one, with their real defaults.
