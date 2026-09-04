# Reading the trust score

What the number is made of, and what you cannot conclude from it. Verified against
`trustworthy-llm==0.0.3`.

## What contributes, per preset

| Preset | Composition | What it measures |
|---|---|---|
| `medium` (default) | self-reflection **70 %**, perplexity **30 %** | the judge's opinion on factual correctness, plus the generator's confidence in its own tokens |
| `high` / `best` | self-reflection **47 %**, consistency **32 %**, perplexity **20 %**, indicator **1 %** | same, plus how stable the answer is under resampling |
| `base` / `low` | identical to `medium` on the `score()` path | they reduce neither the work nor the score |

`tlm` averages five weighted sub-scores. A signal that was not computed is dropped and
the average renormalizes over the rest, so absent signals are ignored rather than
penalized, hence the 70/30 at `medium`, where only self-reflection (weight 0.47) and
perplexity (0.20) survive: 0.47/0.67 and 0.20/0.67.

Moving to `high` is therefore **not** "the same score, stricter". A third of the weight
shifts onto a different question, whether the model answers the same way twice, so a
correct but freely worded answer can lose points with no error detected. Measured here:
a good answer went from ~0.9 to ~0.72.

## The judge is most of the score

At `medium`, ~70 % of the number is one model's opinion. `tlm` asks it the same
question/answer pair **six times** from six angles (certainty, knowledge gap,
adversarial argument, binary correctness, graded trustworthiness, claim-by-claim
correctness), maps each to 0-1 and takes a plain mean.

All six interrogate **the same model**, so they do not vote independently. On a
systematic error, where the generator is wrong in the same way with the same
confidence every time, the six prompts reproduce that misjudgement and averaging
corrects nothing.
That is why scaling the judge up fixed the false negative recorded in this project (an
answer confusing grams with kilograms scored 0.842 "Reliable" under `qwen2.5:7b`, 0.342
under `qwen2.5:14b`) while sampling more would not have.

None of the six looks at relevance, completeness, tone, or adherence to your system
prompt. None consults an external source. The score is what the judge believes it knows.

## A threshold belongs to a configuration

This is why this repository ships **no calibrated thresholds**, and why none would help
you if it did. Five things move a cut-off, and the first two move it more than the
preset everyone thinks of:

| What you change | Measured effect |
|---|---|
| **The judge model** | two answers with real errors: 0.842 and 0.769 under `qwen2.5:7b`, 0.342 and 0.525 under `qwen2.5:14b` |
| **The generator model** | perplexity is its confidence in its own tokens, ~30 % of the score at `medium` |
| **`quality_preset`** | a good answer, ~0.9 at `medium`, ~0.72 at `high` |
| **`reasoning_effort`** | anything but `none` sends the judge a prompt that reasons before rating, changing its output distribution. Untested here |
| **Your questions and domain** | the judge's reliability is not uniform across subjects |

[`scripts/calibrate.py`](../scripts/calibrate.py) derives thresholds for your own
configuration and records which one produced them.

## Two things the number will not tell you

**0.5 is ambiguous.** It is both "the judge is unsure" and the fallback when the judge
does not answer in the expected format (`SELF_REFLECTION_PARSE_FAILURE_SCORE`). A
smaller judge that mangles the output format scores 0.5 without anything being raised,
and `trust_score` alone cannot distinguish the two.

**The score carries no decomposition.** `tlm` computes the five sub-scores and drops
them, so a score built from one surviving judge call looks like one built from six. They
can only be read out of `tlm`'s own logs, which is scraping, not an API:

```python
logging.getLogger("tlm.utils.scoring.trustworthiness_scoring_utils").setLevel(logging.INFO)
logging.basicConfig()  # stderr then carries the per-signal arrays
```

Also worth knowing: the field `tlm` calls `perplexity` holds a **mean token probability**
in [0, 1], not a perplexity. Feeding it a real perplexity (`exp(-mean logprob)`, at least
1) would be the wrong scale, silently. `tlm_local` supplies what the field actually
wants.
