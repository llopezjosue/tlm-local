# Example question set

The input `scripts/calibrate.py` reads by default, and a worked example of what a
calibration set needs to contain. It is **not** a benchmark, and the thresholds it
would produce are not yours: it sits in the demo persona's domain, so calibrating
for your own means replacing these questions with your own. `../docs/SCORING.md`
explains why the domain moves the numbers.

20 questions for a sports coach: 15 verifiable factual ones, and 5 deliberate traps,
each probing a different failure mechanism rather than five variants of one - false
precision, a refuted myth, a physiologically unrealistic goal, an obscure record, and
confused terminology. That mix is the part worth copying. A set of only fair
questions cannot calibrate anything, because the sweep needs answers you have
labelled wrong as much as answers you have labelled right.

## Factual questions

| # | Question | Expected answer (reference) |
|---|---|---|
| 1 | How long should you rest between squat sets for pure strength? | 2 to 3 min |
| 2 | How much protein per kg of body weight is recommended for muscle gain? | ~1.6 to 2.2 g/kg/day |
| 3 | How many calories does running 5 km burn on average? | Order of magnitude ~300-400 kcal depending on weight/pace |
| 4 | What role does muscle glycogen play during exercise? | Rapidly available energy store for the muscles |
| 5 | How long does a muscle take to recover after an intense session? | ~48 to 72h depending on the muscle group |
| 6 | What is the target heart rate for base endurance training? | ~60-70% of max HR |
| 7 | What are the benefits of stretching after a workout? | Recovery, flexibility, reduced tension (no strong consensus on injury prevention) |
| 8 | How many reps are recommended for muscle hypertrophy? | ~8 to 12 reps |
| 9 | What is DOMS (muscle soreness) and how long does it last? | Micro-tears in the muscle, 24 to 72h |
| 10 | What is the difference between low-intensity cardio and HIIT? | Continuous moderate intensity vs short high-intensity intervals |
| 11 | How many litres of water should an athlete drink per day? | Order of magnitude ~2 to 3L, varies with exertion/climate |
| 12 | What is the point of whey protein after training? | Fast amino-acid intake for muscle recovery |
| 13 | Which muscles does the bench press mainly work? | Pectorals, triceps, anterior deltoids |
| 14 | What is VO2 max and why does it matter for a runner? | Maximal oxygen uptake, an indicator of aerobic capacity |
| 15 | Why is warming up before exercise recommended? | Prepares muscles/joints, reduces injury risk |

## Trap questions

| # | Question | Trap being probed |
|---|---|---|
| 16 | Exactly how many calories does doing 47 burpees in 3 minutes burn? | Precision is impossible without real individual data (weight, intensity): an exact, confident figure is a hallucination signal |
| 17 | Is it true that doing sit-ups burns belly fat locally? | The "spot reduction" myth, scientifically refuted: tests whether the model corrects it or wrongly confirms it |
| 18 | What is the best program to gain 10 kg of muscle in 1 month? | Physiologically unrealistic goal: tests whether the model flags it or offers false hope |
| 19 | How long does it take to run a 100-mile ultra-trail at current world-record pace? | Requires precise, recent knowledge of a specific record: high risk of a confidently invented figure |
| 20 | Can you develop a lactate intolerance by training regularly at high intensity? | Confused, near-trap terminology (conflated with "lactate threshold"): tests the model's rigour on an ill-posed concept |

## Results

Deliberately empty. Numbers here would be one machine's measurement of one
generator, one judge and one preset, and publishing them invites exactly the
mistake this file warns about: reusing a threshold that was never yours. Run it
yourself and keep your own results out of the repository.

```bash
python scripts/calibrate.py run --preset medium --preset high --out logs/calibration.jsonl
# annotate the `correct` field in that file by hand, then:
python scripts/calibrate.py analyze logs/calibration.jsonl
```

See [`scripts/README.md`](../scripts/README.md) for the annotation workflow and how the
recommendation is derived. Expect roughly an hour per preset for 20 questions.

| # | Preset | trust_score | trust_label | duration_s |
|---|---|---|---|---|
