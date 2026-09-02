# Scoring parameters

`score()` and `generate_and_score()` accept `tlm`'s configuration surface. What each
parameter means is upstream's business: see [`cleanlab.github.io/tlm/api/config/`](https://cleanlab.github.io/tlm/api/config/).

What follows is only what upstream does not tell you, verified against the installed
`trustworthy-llm==0.0.3` source: which of those parameters do nothing, do something
unexpected, or fail outright. Read [`../docs/SCORING.md`](../docs/SCORING.md) first for
what actually moves the score.

## Named parameters

| Parameter | Status here |
|---|---|
| `quality_preset` | Works. `base` and `low` are identical to `medium` on the `score()` path, so only `medium`, `high` and `best` differ |
| `reasoning_effort` | Works, and it is not cosmetic: upstream describes it as an explanation-length cap, but any value other than `none` switches the judge to a prompt that reasons before rating, so it changes the score too. `none` is the effective default for QA under every preset |
| `similarity_measure` | Inert below `quality_preset=high`: it only affects the consistency signal, which runs 0 completions at `medium`. The `embedding_*` values are the one code path in `tlm` that calls hosted OpenAI |
| `constrain_outputs` | Works. Multiple-choice and classification workflows only |
| `evals` | **Raises `EvalsNotSupportedError`.** Broken upstream: the pipeline factory builds `SemanticEvaluationScoreGenerator` with a `model=` argument the component does not accept. Reproducible with a bare `TLM().score(evals=[...])` |
| `context` | **Raises `RagNotSupportedError`.** A non-`None` context selects the RAG workflow, which injects its own default evals and hits the same bug — so it fails even with no evals passed |

## The escape hatch

Lower-level fields normally derived by `quality_preset` are reachable through
`**advanced_tlm_config`, validated against `ADVANCED_CONFIG_FIELDS` so a typo fails here
rather than inside `tlm`'s own validation:

```python
await client.score(messages, raw_response, quality_preset="medium", num_consistency_completions=2)
```

Two traps in that set:

- **`num_self_reflection_completions` cannot be reduced.** The truncation guard is
  `if num_completions > 1`, so `-1`, `0` and `1` all fall through and the six templates
  run regardless. Asking for fewer is not possible.
- **`self_reflection_temperature` is dead upstream.** Declared, read by no component;
  the generator hardcodes `temperature=0.0`. Passing it produces neither an error nor an
  effect.

`provider`, `api_key` and `api_version` are not exposed: this wrapper is local and
key-free by design.
