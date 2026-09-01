# Reference: every `tlm` `Config` parameter

`tlm`'s own published docs (`cleanlab.github.io/tlm/api/config/`) are auto-generated
from the docstrings below, one line per parameter, no practical guidance. This
document adds what we've actually verified against the source (`tlm/config/schema.py`,
`tlm/config/presets.py`, `tlm/types/base.py`) and empirically on this project. Source
version: `trustworthy-llm==0.0.3`.

For the **combined** effect of these parameters (what a given configuration actually
measures, how the five sub-scores are weighted and renormalized, and what each preset
catches or misses), see [`../docs/SCORING.md`](../docs/SCORING.md). This document covers
the parameters individually; that one covers the combinations.

`LocalTLM.score()`/`.generate_and_score()` expose `quality_preset`, `reasoning_effort`,
`similarity_measure`, `constrain_outputs` and `evals` as named parameters, plus every
lower-level knob below through a validated `**advanced_tlm_config` escape hatch (see
`tlm_local/client.py`, `ADVANCED_CONFIG_FIELDS`). `provider`/`api_key`/`api_version` are
the only fields genuinely not applicable to this wrapper's local-only, BYOK-free design.

## All 17 parameters at a glance

| Parameter | Where | Values | Default | What it does | Status here |
|---|---|---|---|---|---|
| `quality_preset` | `Config` | `base`\|`low`\|`medium`\|`high`\|`best` | `medium` | Speed vs scoring depth | Exposed, named parameter (default `medium`) |
| `reasoning_effort` | `Config` | `none`\|`low`\|`medium`\|`high` | `None`, resolved to `none` for QA | Max explanation word count (0/50/125/375), **and** switches the judge prompt to a variant that reasons before scoring, see caveat below | Exposed, named parameter; `explanation` still not surfaced in `ScoreResult` |
| `similarity_measure` | `Config` | `jaccard`\|`embedding_small`\|`embedding_large`\|`code`\|`statement` | `None` (auto per workflow) | How `consistency` compares alternate answers | Exposed, named parameter; no-op while consistency=0 (i.e. below `quality_preset=high`) |
| `constrain_outputs` | `Config` | `list[str]` \| `None` | `None` | Restricts outputs (multiple-choice/classification) | Exposed, named parameter; not applicable to a QA workflow |
| `num_reference_completions` | `Config` | `int` \| `None` | preset-derived | Completions generated (generator role, `.create()` only) | Exposed via `**advanced_tlm_config` |
| `num_consistency_completions` | `Config` | `int` \| `None` | preset-derived (0/0/0/4/8) | Alternate samples for `consistency` | Exposed via `**advanced_tlm_config`, normally left to `quality_preset` |
| `observed_consistency_temperature` | `Config` | `float` \| `None` | `None` | Sampling temperature for comparison completions | Exposed via `**advanced_tlm_config` |
| `self_reflection_temperature` | `Config` | `float` \| `None` | `None` | Nothing: declared but read by no component in `0.0.3`, self-reflection is hardcoded to `temperature=0.0` | Exposed via `**advanced_tlm_config`, but a dead field upstream, see below |
| `num_self_reflection_completions` | `Config` | `int` \| `None` | preset-derived (`-1` = all) | How many self-reflection templates run, but only values `> 1` truncate: `-1`, `0` and `1` all run all 6 | Exposed via `**advanced_tlm_config`; always 6 in practice, even when overridden to 0 or 1, see below |
| `use_prompt_evaluation` | `Config` | `bool` \| `None` | `None` | Enables the prompt-eval score (RAG) | Exposed via `**advanced_tlm_config`; not applicable (no RAG) |
| `prompt_evaluation_temperature` | `Config` | `float` \| `None` | `None` | Temperature for prompt-eval | Exposed via `**advanced_tlm_config`; not applicable |
| `semantic_evaluation_temperature` | `Config` | `float` \| `None` | `None` | Temperature for scoring custom `evals` | Exposed via `**advanced_tlm_config`; moot while `evals` itself is broken upstream, see below |
| `provider` | `Config` | `str` \| `None` | `None` | Provider name | Not applicable (we route via `model=` directly) |
| `api_base` | `Config` | `str` \| `None` | `None` | Nothing on the scoring path: judge calls go out with `api_base=None` regardless, and litellm resolves the host from `OLLAMA_API_BASE` | Not exposed; `LocalTLM` exports `ollama_api_base` into that env var instead, see below |
| `api_key` | `Config` | `str` \| `None` | `None` | API key | Not applicable (BYOK, zero keys) |
| `api_version` | `Config` | `str` \| `None` | `None` | API version | Not applicable |
| `evals` | Not on `Config`, passed to `.score()`/`.create()` per call | `list[Eval]` \| `None` | `None` | Custom semantic evaluation criteria (name/criteria/identifiers) | Exposed, named parameter; any non-empty value raises `EvalsNotSupportedError`, a real upstream bug, see below |

## `quality_preset` (`QualityPreset`: `base`\|`low`\|`medium`\|`high`\|`best`, default `medium`)

Controls `num_consistency_completions` and (partially) `num_self_reflection_completions`
via internal preset tables. What we verified empirically on this project (not just
read from the source, which turned out to be an unreliable predictor on its own,
see below):

| Preset | `num_consistency_completions` (QA workflow) | Self-reflection |
|---|---|---|
| `base` | 0 | Still runs all 6 QA templates (confirmed empirically, contradicts a static reading of the preset tables, which suggest 0) |
| `low` | 0 | Same as `base`/`medium`, does not reduce self-reflection cost |
| `medium` (our default) | 0 | 6 templates |
| `high` | 4 | 6 templates |
| `best` | 8 | 6 templates |

Measured on this project's hardware/models: `medium` ~59-90s per scored answer,
`high` ~94-179s, and `high` recalibrates scores rather than just adding strictness
(a genuinely good answer measured here dropped from ~0.9 to ~0.72). Full story in
`../docs/JOURNAL.md`.

## `reasoning_effort` (`ReasoningEffort`: `none`\|`low`\|`medium`\|`high`, default `None`)

Controls the max word count of the score's **explanation** field, verified in
`tlm/config/presets.py` (`REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS`): `none`=0,
`low`=50, `medium`=125, `high`=375 words. It does **not** change the number of
completions/calls made.

It does, however, change the score, contrary to what this section originally claimed.
Each self-reflection template carries two prompt variants (see each `create()` in
`tlm/templates/reflection_completion_templates.py`): `_PROMPT_TEMPLATE` with no
`<think>` block, and `_PROMPT_TEMPLATE_WITH_REASONING`, which asks the judge to reason
step by step *before* producing its rating. Any value other than `none` selects the
second. The judge's prompt changes, so its output distribution changes.

Worth knowing: for a QA workflow the effective default is `none`, not "unset".
`tlm/config/base.py:68-70` resolves it to `ReasoningEffort.NONE` for every workflow
except `STRUCTURED_OUTPUT_SCORING`, and no preset overrides it (`low`/`base` set it to
`NONE` redundantly). So by default the judge scores with no deliberation step at all,
under every `quality_preset`. See [`../docs/SCORING.md`](../docs/SCORING.md) for why
this is the cheapest untested lever on this project.

Exposed as `score(..., reasoning_effort=...)`, but note `LocalTLM`'s `ScoreResult` only
extracts `trustworthiness_score` from `tlm`'s `InferenceResult` today; the full
`InferenceResult` (including `explanation`) is available on `ScoreResult.raw` for any
caller that wants it, but neither the backend showcase nor the frontend currently read
or display it. Tuning this has zero visible effect until something downstream consumes
`explanation`.

## `similarity_measure` (`SimilarityMeasure`: `jaccard`\|`embedding_small`\|`embedding_large`\|`code`\|`statement`, default `None`)

Controls how the **consistency** signal compares alternate generated answers to each
other. Only has any effect when `num_consistency_completions > 0`, i.e. `quality_preset=high`/`best`
- irrelevant at our default `medium`. When left `None`, `tlm` picks a default per
workflow type (`tlm/types/base.py`, `SimilarityMeasure.for_workflow()`, verified
directly):

| Workflow | Default `similarity_measure` |
|---|---|
| QA (our workflow) | `statement` |
| Classification | `embedding_small` |
| Binary classification | `embedding_large` |
| RAG | `code` |
| Structured output scoring | `jaccard` |

Worth knowing: `embedding_small`/`embedding_large` are the two values that trigger
the one hardcoded, non-BYOK OpenAI API call found during Phase 0 (`tlm/utils/openai_utils.py`,
a raw `AsyncOpenAI` client for embedding similarity). Since our workflow is QA, the
default is `statement` (no OpenAI call), not embedding-based - this stays true even
if we adopt `quality_preset=high`, as long as we don't explicitly override
`similarity_measure` to one of the embedding options ourselves.

## `constrain_outputs` (`list[str] | None`, default `None`)

Restricts responses to a fixed set of allowed values (multiple-choice / classification
workflows). Exposed as `score(..., constrain_outputs=[...])` for completeness; not
applicable to our QA chatbot use case.

## `evals` (`list[Eval] | None`) - not a `Config` field, passed to `.score()`/`.create()` per call

Lets you define custom semantic evaluation criteria on top of the core trustworthiness
score (e.g. "check the response is polite", "check it mentions X"). `Eval` is a
pydantic model (`tlm/types/base.py`, re-exported as `tlm_local.Eval`):

| Field | Meaning |
|---|---|
| `name` | Name of the evaluation |
| `criteria` | Semantic description of what to assess |
| `query_identifier` | Label for the user query in the prompt sent to the judge, or `None` if not needed |
| `context_identifier` | Label for the context, or `None` if not needed |
| `response_identifier` | Label for the response, or `None` if not needed |

`LocalTLM.score()` accepts an `evals` parameter, but any non-empty value currently
raises `EvalsNotSupportedError`: a genuine bug in `trustworthy-llm==0.0.3` itself,
reproduced independent of `tlm_local` with a bare `TLM().score(evals=[...])` call.
`tlm`'s pipeline factory (`tlm/pipeline/factory.py`) constructs its
`SemanticEvaluationScoreGenerator` component with `model=config.model`, an argument
that component's own `__init__` (`tlm/components/semantic_evaluation_score_generator.py`)
doesn't accept, so it falls through to `Component.__init__(**kwargs)`, which rejects it
too: `TypeError: Component.__init__() got an unexpected keyword argument 'model'`. Not
something a caller can work around, and not fixable here without forking `tlm`; revisit
if a future `trustworthy-llm` release fixes it (see `tests/test_client.py`, which pins
this exact failure so a fix upstream surfaces as a newly-failing test here).

## Lower-level knobs normally set automatically by `quality_preset`

`num_reference_completions`, `num_consistency_completions`, `observed_consistency_temperature`,
`self_reflection_temperature`, `num_self_reflection_completions`, `use_prompt_evaluation`,
`prompt_evaluation_temperature`, `semantic_evaluation_temperature`. Each is normally
derived from `quality_preset` via internal tables (`tlm/config/presets.py`), but can be
set individually for finer control than the presets offer, e.g. `medium` preset behavior
but with `num_consistency_completions=2` instead of 0:

```python
await tlm_client.score(messages, raw_response, quality_preset="medium", num_consistency_completions=2)
```

Exposed through `score()`'s `**advanced_tlm_config` (validated against
`tlm_local.ADVANCED_CONFIG_FIELDS`; an unknown keyword raises `ValueError` immediately
rather than failing inside `tlm`'s own pydantic validation). Not benchmarked for score
quality here beyond the preset-level defaults.

Two upstream caveats verified in `0.0.3`, both detailed in
[`../docs/SCORING.md`](../docs/SCORING.md):

- **`self_reflection_temperature` is dead.** Declared on `Config`
  (`tlm/config/base.py:37`, `tlm/config/schema.py:44`) and read by no component
  anywhere in the package. Self-reflection completions are hardcoded to
  `temperature=0.0` (`self_reflection_completion_generator.py:50`). Setting it produces
  neither an error nor an effect. The other seven fields listed above *are* consumed,
  in `tlm/pipeline/factory.py`.
- **`num_self_reflection_completions` cannot be lowered to 0 or 1.** The truncation
  guard is `if num_completions > 1`
  (`self_reflection_completion_generator.py:25-26`), so `-1`, `0` and `1` all fall
  through untruncated and run all 6 QA templates. This is the mechanism behind
  `quality_preset=base` not disabling self-reflection despite requesting 0.

## `provider` / `api_base` / `api_key` / `api_version` on `Config`

These exist on `Config` (`ModelProviderSchema`) but setting them does not do what the
names suggest, so `tlm_local` routes around them instead. An earlier version of this
section claimed the generator role gets `model`/`api_base` passed as `**openai_kwargs`
to `.score()`/`.create()`; that was wrong, and the code contradicts it. What actually
happens:

- **`api_base` is not honored on the scoring path at all.** Judge and consistency calls
  are built inside `tlm` from `completion_params` plus `ModelProvider`
  (`tlm/utils/completion_utils.py`), and they go out with `api_base=None`. Setting
  `Config(api_base=...)` does not change that (verified by instrumenting
  `tlm.utils.completion_utils.acompletion`: the captured judge calls still carried
  `api_base=None`). litellm resolves the Ollama host from the `OLLAMA_API_BASE` env var
  instead, defaulting to `localhost:11434`. `LocalTLM` therefore exports
  `LocalTLMConfig.ollama_api_base` into that variable at construction, which is the only
  route that actually reaches the judge. Only `generate()` can pass `api_base` directly,
  because it is the one call `tlm_local` builds itself.
- **`api_key` / `provider`** are what make the fallback dangerous rather than merely
  broken: `tlm`'s defaults are provider `openai` and an `api_key` auto-filled from
  `OPENAI_API_KEY` (`tlm/config/defaults.py`), paired with the `gpt-4.1-mini` judge
  default. That is why `LocalTLM` validates the resolved judge model instead of trusting
  configuration, see `JudgeModelNotLocalError`.
- **The judge model itself** is configured only through the `DEFAULT_MODEL` env var and
  `tlm`'s own cached `Settings` (see the main README's import-order note), never through
  these `Config` fields.

All four are deliberately absent from `ADVANCED_CONFIG_FIELDS`: allowing them would
suggest a control that does not exist.
