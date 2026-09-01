# scripts/

## `calibrate.py`

Replaces the guessed trust-label thresholds with measured ones.

`backend/app/config.py` ships `TRUST_THRESHOLDS = {"reliable": 0.8, "needs_checking": 0.5}`
marked provisional, and the results table in `docs/test_questions.md` is empty. Those
numbers have never been checked against data. This script is how you check them.

Three steps, because the middle one needs a human:

```bash
# 1. score the questions (roughly an hour per preset for 20 questions)
python scripts/calibrate.py run --preset medium --out logs/calib.jsonl

# 2. open logs/calib.jsonl and set "correct" on each row to true or false
#    (leave it null where you cannot tell). It is the first key on every line.

# 3. sweep the cut-offs
python scripts/calibrate.py analyze logs/calib.jsonl
```

Needs Ollama running for step 1, same as the app. Step 3 needs nothing.

The questions come from `docs/test_questions.md`, parsed out of its two markdown
tables, so there is a single source of truth for the test set. Edit that file to
change the questions. The parser is coupled to the table layout, and will say so
rather than run on an empty set if the layout changes.

### Reading the output

The sweep prints, for each cut-off from 0.05 to 0.95, how many correct answers
would be called reliable (`ok`) and how many **incorrect** ones would be
(`bad`), plus precision, recall and F1. `bad` is the column that matters most:
it counts wrong answers a user would be told to trust, the failure mode this
project has already hit twice in practice.

The recommendation minimizes `bad` first, then takes the best F1, then the
lowest cut-off so recall is not given away for nothing. The lower threshold is
placed at or below the weakest correct answer, so nothing known to be right ends
up in the bottom bucket.

Thresholds are per preset and do not transfer: at `high`, self-reflection drops
from about 70% to 47% of the score's weight. See `docs/SCORING.md`.

### Caveats

- One run per question by default, so a single score carries whatever run-to-run
  variance the judge has. Repeat the run and compare before trusting a tight
  margin.
- No resume: re-running appends, and `analyze` will then see duplicate rows for
  the same question. Delete the file or use a fresh `--out` to start over.
- No synthetic results are committed. Fake numbers that look like measurements
  are worse than a visibly empty results table.

### Linting

`ruff.toml` repeats `tlm_local/pyproject.toml`'s settings, because ruff resolves
config from the checked file's directory upwards and the repo root carries none.
Without it, `ruff check scripts/` would quietly apply ruff's own defaults.
