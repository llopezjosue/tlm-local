"""Calibrate the trust-label thresholds against a real question set.

Two steps, because the middle one is yours. `run` scores every question and
writes a JSONL with `correct` left null; you label the answers by hand; then
`analyze` sweeps candidate cut-offs and tells you where to put them.

The thresholds in backend/app/config.py are provisional guesses. This is how to
replace them with measured ones. Expect roughly an hour per preset for 20
questions.

    python scripts/calibrate.py run --preset medium --out logs/calib.jsonl
    # edit the "correct" field in each row: true, false, or null if unclear
    python scripts/calibrate.py analyze logs/calib.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = REPO_ROOT / "docs" / "test_questions.md"
SWEEP_GRID = [round(0.05 * n, 2) for n in range(1, 20)]

# Questions live in docs/test_questions.md, not here: duplicating them would give
# the project two sources of truth for the same test set. The price is that this
# parser is coupled to that file's table layout.
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|")


def load_questions() -> list[dict]:
    category = None
    questions = []
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            category = "factual" if "factual" in heading else "trap" if "trap" in heading else None
        elif category and (match := ROW.match(line)):
            questions.append({"id": match.group(1), "question": match.group(2), "category": category})
    if not questions:
        sys.exit(f"No questions parsed from {QUESTIONS_FILE}. Did its table layout change?")
    return questions


async def run(presets: list[str], out: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from tlm_local import LocalTLM  # noqa: E402  imported late so `analyze` needs no tlm/.env/Ollama

    from app.generator import build_messages  # noqa: E402  reuse the app's own prompt assembly

    questions = load_questions()
    client = LocalTLM()
    out.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions) * len(presets)
    done = 0
    started = time.monotonic()

    with out.open("a", encoding="utf-8") as handle:
        for preset in presets:
            for question in questions:
                done += 1
                began = time.monotonic()
                try:
                    generation, score = await client.generate_and_score(
                        build_messages(question["question"]), quality_preset=preset
                    )
                    row = {
                        "correct": None,  # first key on the line: this is what you edit
                        "id": question["id"],
                        "category": question["category"],
                        "preset": preset,
                        "question": question["question"],
                        "answer": generation.answer,
                        "trust_score": score.trust_score,
                        "duration_s": round(time.monotonic() - began, 1),
                    }
                except Exception as error:  # noqa: BLE001  one bad question must not lose the batch
                    row = {"correct": None, "id": question["id"], "preset": preset, "error": str(error)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                elapsed = time.monotonic() - started
                print(
                    f"[{done}/{total}] {preset} q{question['id']}: "
                    f"{row.get('trust_score', row.get('error'))} "
                    f"({elapsed / 60:.0f} min elapsed)",
                    flush=True,
                )

    print(f"\nWrote {out}. Now set `correct` to true or false on each row, then run:")
    print(f"  python scripts/calibrate.py analyze {out}")


def sweep(correct: list[float], incorrect: list[float]) -> list[dict]:
    """Precision, recall and false negatives for calling an answer reliable at each cut-off.

    The false-negative count (a wrong answer shown as trustworthy) is the number
    the journal's two documented failures are about, so it gets its own column.
    """
    rows = []
    for cut in SWEEP_GRID:
        hits = sum(1 for s in correct if s >= cut)
        false_positives = sum(1 for s in incorrect if s >= cut)
        called = hits + false_positives
        precision = hits / called if called else 0.0
        recall = hits / len(correct) if correct else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {"cut": cut, "ok": hits, "bad": false_positives, "precision": precision, "recall": recall, "f1": f1}
        )
    return rows


def analyze(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [r for r in rows if "trust_score" in r]
    if not scored:
        sys.exit(f"No scored rows in {path}.")

    for preset in sorted({r["preset"] for r in scored}):
        group = [r for r in scored if r["preset"] == preset]
        correct = [r["trust_score"] for r in group if r.get("correct") is True]
        incorrect = [r["trust_score"] for r in group if r.get("correct") is False]

        print(f"\n=== preset={preset} ===")
        print(
            f"{len(group)} scored, {len(correct)} correct, {len(incorrect)} incorrect, "
            f"{len(group) - len(correct) - len(incorrect)} unlabelled"
        )
        if not correct or not incorrect:
            print("Need at least one correct and one incorrect answer to sweep. Label more rows.")
            continue
        print(f"correct   min {min(correct):.2f} max {max(correct):.2f}")
        print(f"incorrect min {min(incorrect):.2f} max {max(incorrect):.2f}")

        table = sweep(correct, incorrect)
        print("\n cut   ok  bad  precision  recall     f1     (bad = wrong answers called reliable)")
        for row in table:
            print(
                f"{row['cut']:.2f}  {row['ok']:3d}  {row['bad']:3d}     "
                f"{row['precision']:.3f}   {row['recall']:.3f}  {row['f1']:.3f}"
            )

        # Keeping wrong answers out of the "reliable" bucket comes first; among
        # the cut-offs that manage it, take the best F1, then the lowest cut-off
        # so recall is not given away for nothing.
        clean = [r for r in table if r["bad"] == 0]
        best = min(clean, key=lambda r: (-r["f1"], r["cut"])) if clean else max(table, key=lambda r: r["f1"])
        # The lower cut-off has to sit at or below the weakest correct answer, so
        # no answer known to be right lands in the bottom bucket. Nearest grid
        # point is not enough: rounding up would strand it there.
        below = [c for c in SWEEP_GRID if c <= min(correct)]
        floor = max(below) if below else SWEEP_GRID[0]
        print(f"\nrecommended: reliable >= {best['cut']:.2f}, needs_checking >= {floor:.2f}")
        if not below:
            print(f"A correct answer scored {min(correct):.2f}, below the sweep grid; no floor keeps it out.")
        if not clean:
            print("No cut-off keeps every wrong answer out of the reliable bucket; showing best F1 instead.")
        print("Current values live in backend/app/config.py (TRUST_THRESHOLDS).")
        print("Thresholds do not transfer between presets: see docs/SCORING.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="score every question and write an unlabelled results file")
    runner.add_argument("--preset", action="append", dest="presets", metavar="NAME", help="repeatable, default medium")
    runner.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "calibration.jsonl")

    analyzer = sub.add_parser("analyze", help="sweep thresholds over a labelled results file")
    analyzer.add_argument("results", type=Path)

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args.presets or ["medium"], args.out))
    else:
        analyze(args.results)


if __name__ == "__main__":
    main()
