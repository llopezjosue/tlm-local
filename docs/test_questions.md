# Test set (Phase 4)

Goal: empirically confirm that the trust score reacts differently to a good answer and
a bad one, and calibrate the `Reliable`/`Needs checking`/`Unreliable` thresholds against
real data rather than the provisional values set in Phase 2 (`>=0.8` / `>=0.5`).

20 questions, within the demo persona's domain (sports coach): 15 verifiable factual
ones, 5 deliberate traps designed to expose overconfidence in the small generator model
(false precision, myth, unrealistic goal, obscure fact, confused terminology). Results
(score, latency, preset) get filled into the table below as the real runs happen.

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

Still empty: the batch was started once at `quality_preset=medium` and stopped after
3/20 questions, and has not been re-run.

Run it with the calibration harness, which carries this same question set as its default
and does the threshold sweep this phase exists for:

```bash
python scripts/calibrate.py run --preset medium --preset high --out logs/calibration.jsonl
# annotate the `correct` field in that file by hand, then:
python scripts/calibrate.py analyze logs/calibration.jsonl
```

See [`scripts/README.md`](../scripts/README.md) for the annotation workflow and how the
recommendation is derived. Expect roughly an hour per preset for 20 questions.

| # | Preset | trust_score | trust_label | duration_s |
|---|---|---|---|---|
