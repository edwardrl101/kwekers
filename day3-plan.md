# Day 3 plan

## Purpose

Day 3 is an optimization and integration day. The team already has a strong,
simple BM25-plus-freshness reference (`0.8553`) and a more capable Member 1
pipeline whose last measured full-set score was lower (`0.820121`). The goal is
to identify which parts of the full pipeline help, remove or disable the parts
that hurt, and finish with the simplest measured configuration that preserves
or improves the `0.8553` floor.

The priority is evidence, not additional capability. Every change should be
isolated, reversible, tested, and evaluated with the same fixed data split.

## What success looks like

By the end of Day 3, the team should have:

1. A verified exact-route integration that uses all active constraints in one
   intersection rather than unioning single-constraint results.
2. Cross-turn recommendation freshness, with deliberate reset behavior for
   intent-override sessions.
3. A six-configuration ablation table with overall and per-scenario metrics.
4. A clear ship/reject decision for exact, bucket, and dense evidence.
5. A working BM25-plus-freshness fallback that is not endangered by
   experiments.
6. Focused tests proving the final behavior, including a real
   `ExactRoute`-through-`Agent` integration test.

The expected outcome is approximately `0.85-0.87` from freshness plus removal
of harmful fusion terms. A small, high-confidence exact set may push the score
higher by improving MRR, but it should ship only if measurement confirms this.

## Current facts and baseline

The relevant reported results are:

| System | Technical score | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Member 4 BM25-only + freshness | **0.8553** | 0.995 | 0.634 | 2.62 |
| Member 1 pre-merge full pipeline | 0.820121 | 0.945 | 0.597405 | 2.58 |

The `0.820121` result is historical. It was measured before the current exact
route/API mismatch and must not be described as the score of the present
checkout. The present code must be re-evaluated after integration is repaired.

Technical score is:

```text
0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

This makes any top-10 hit valuable, while rank and earlier hits remain important.
The final system must always return exactly 10 valid, unique product IDs and a
non-null question attribute.

## Resolution of the exact-route contradiction

The contradiction has been resolved and repaired in the current checkout:

- Member 3's `query()` implementation was cherry-picked as `a7dc2aa`.
- `starter/agent.py::_route_exact()` now passes all active constraints to
  `query()` in one call, producing an AND intersection.
- A real `ExactRoute`-through-`Agent` test proves that exact evidence reaches
  fusion and promotes the matching BM25 candidate.
- Unmatched active constraints now force an empty intersection instead of
  being silently skipped.
- Exact results are sorted before the 500-result limit, eliminating
  process-dependent candidate subsets.

Member 3's update was cherry-picked from `origin/vigo-branch` as local commit
`a7dc2aa` (source commit `88fcb79`). The checked-out `src/exact.py` now provides
`query()` and natively intersects a list of constraints. A direct smoke check
confirmed that two constraints produce their intersection and an unmatched
active constraint produces no candidates. The Agent adapter and end-to-end
coverage were completed in `cd91365`; deterministic exact limiting was added
in `dc7b5fd`.

The intended Member 1 adapter is:

```python
def _route_exact(
    self, session: dict, user_message: str, turn: int
) -> list[ScoredCandidate]:
    active_constraints = session.get("active_constraints")
    if isinstance(active_constraints, list) and active_constraints:
        return self._query_route(self._exact_route, active_constraints)
    return self._query_route(self._exact_route, user_message)
```

Member 3's `query()` accepts a constraint list, and `_query_route` now uses the
matching `str | list[str]` type contract. The implemented semantics are one
call over all accumulated active constraints, producing an AND intersection.

Before evaluating, verify all of the following:

- A real `ExactRoute` instance returns a non-empty result for a known matching
  constraint.
- Two compatible constraints produce their intersection, not their union.
- The exact IDs reach `_fuse_bm25_pool()` in an end-to-end Agent test.
- Invalid IDs are removed, duplicates are removed, and the route result is
  capped at 500.
- An exact-route failure still leaves the response valid and does not disable
  BM25.

The former silent-skip issue is also fixed and covered by tests. Day 3 did not
add a special color index for `color: black`; its reported coverage was only
3/40 and did not justify that scope.

## Member 1 deliverables

Member 1 owns the end-to-end pipeline and the ablation. The work should be done
in this order.

### 1. Add freshness and override reset

Add a per-session `shown` set during `reset()`.

On every normal turn:

1. Rank the candidates.
2. Exclude IDs already in `shown` before selecting the final ten.
3. Ensure deterministic fallback filling also avoids IDs in `shown`.
4. After the final ten valid IDs are chosen, add those IDs to `shown`.

On a message containing the intent-override signal (currently
`"ignore my earlier preference"`, case-insensitive), clear `shown` before final
selection. This matters because recommendations made during the evaluator's
pre-override blackout cannot count, and the target product itself does not
change.

Do **not** wipe all accumulated retrieval context on override. The dialog state
should demote the superseded preference while preserving the category and
unrelated constraints. Exact matching should receive the resulting active
constraint list in one call.

Required freshness tests:

- Consecutive normal turns do not repeat recommendations when enough catalog
  items exist.
- An override clears the freshness history and allows earlier products to be
  shown again.
- The superseded preference is inactive after override, while category and
  unrelated active constraints survive.
- Fallback filling respects freshness and still returns exactly ten valid,
  unique IDs.
- Session reset creates an independent empty `shown` set.

### 2. Integrate Member 3's exact intersection

Replace the current per-constraint loop in `_route_exact()` with one call over
the full active constraint list. Add the real integration tests described
above. A temporary diagnostic such as the exact-set size may be used locally,
but remove ad-hoc printing before committing.

Record exact candidate-set size as part of experiment diagnostics. Large sets
such as approximately 6,521 candidates carry little ranking information;
reported intersections near 24 candidates are the useful signal.

### 3. Establish a post-integration baseline

Run the full unit suite and then run the fixed tune split before comparing
weights:

```bash
python -m unittest discover -s tests -v
python scripts/eval.py --split tune --label day3-current-with-exact
```

Do not edit `evaluator/local_evaluator.py`. Use `data/eval_split.json` unchanged.
The dense cache already exists and should not be regenerated.

### 4. Run the six-way ablation

Use the same code, catalog, split, freshness behavior, and deterministic seed in
every run. Change only the stated fusion term. The configurations from the Day
3 handoff are:

| # | Configuration |
|---:|---|
| 1 | Current fusion baseline |
| 2 | Current fusion + freshness |
| 3 | Freshness, dense weight set to `0` |
| 4 | Freshness, bucket weight set to `0` |
| 5 | Freshness, exact weight set to `0` |
| 6 | Freshness + normalized BM25 rank only; all auxiliary weights set to `0` |

The current fusion formula is:

```text
normalized_bm25_rank
  + 0.35 * exact_match
  + 0.10 * bucket_match
  + 0.20 * normalized_dense_score
```

Run all six configurations on the full 200-session public set as requested by
the Day 3 handoff, with a unique label per run. Also preserve the fixed 140/60
tune/holdout discipline: use tune results for iterative decisions and inspect
holdout only for meaningful finalists, rather than repeatedly optimizing
against it.

For each configuration, report:

- overall TechnicalScore, Hit@10, MRR, MTTC, and Efficiency;
- the same metrics for buying, browsing, intent override, and boundary;
- exact-set size diagnostics where relevant;
- runtime and whether dense was actually loaded;
- the code revision and configuration values.

Configuration 6 is the control. It should approach the independently measured
`0.8553` BM25-plus-freshness result. If it does not, stop weight tuning and find
the behavioral difference between the two evaluation loops.

### 5. Choose what ships

Use measured deltas, not architectural preference:

- Keep freshness unless the reproduction contradicts the large reported gain.
- Disable dense if its weight continues to reduce end-to-end score. The prior
  report estimates a `0.075` cost.
- Keep or remove the bucket boost based on configuration 4. The separate bucket
  filter reduced candidate pools but did not improve ranking, so do not wire it
  in without new evidence.
- Keep exact fusion only if configuration 5 shows a benefit after the AND
  integration is genuinely active.
- Consider Member 4's exact-set promotion only after their threshold sweep
  shows a reliable gain. Promotion should be gated to small non-empty sets and
  must not be conflated with the fusion ablation.
- Prefer the simpler configuration when scores are effectively tied.

The final candidate should be checked on holdout once, then reported on all 200
sessions. Preserve a known-working BM25-plus-freshness fallback throughout.

### 6. Harden and hand off

Before declaring Day 3 complete:

```bash
python -m unittest discover -s tests -v
git diff --check
```

Also verify manually that:

- every response has a string `message`;
- every response has exactly ten valid unique recommendations;
- `ask_attribute` is always non-null and remains `"other"` unless Member 5's
  experiment proves a better policy;
- route failures remain isolated;
- no generated result JSON, dense cache, or unrelated user-owned file is added
  to a commit.

Commit major milestones separately: freshness/tests, exact adapter/tests, and
the final measured configuration. Do not automatically commit the currently
untracked `data/SHA256SUMS` or `runs/` directory, and do not push unless the team
requests it.

## Coordination with other members

| Member | Day 3 responsibility | Dependency for Member 1 |
|---|---|---|
| Member 2 (`brytaniaav`) | Measure near-duplicate suppression after the bucket filter produced no ranking gain | Integrate only if it improves evaluator MRR without hurting Hit@10 |
| Member 3 (`vrospix`) | Finish exact `query()`/intersection behavior, unmatched-constraint semantics, and set-size/coverage measurements | Member 1 needs the actual updated file or commit before changing `_route_exact()` |
| Member 4 (`kr701`) | Sweep confidence thresholds `10/25/50/100/200` for small exact-set promotion | Compare the best promotion result with ordinary exact fusion after the ablation |
| Member 5 (`shengyanmoo`) | Compare always-`other` with a rotating attribute policy and verify override/freshness invariants | Keep always-`other` unless the measured alternative wins |

## Day 3 execution schedule

### Morning

- Member 1 implements freshness and its tests.
- Confirm or merge Member 3's exact update and repair the Agent adapter.
- Run the real exact integration test and a tune baseline.
- Keep the BM25-plus-freshness fallback untouched.

### Midday

- Run the six configurations and assemble the overall/per-scenario table.
- Check configuration 6 against `0.8553` before drawing conclusions.
- Share exact-set sizes so Member 4 can interpret promotion results.

### Afternoon

- Select the smallest configuration supported by the data.
- Integrate only independently measured wins from Members 2, 4, or 5.
- Run the holdout checkpoint and then the final all-session report.
- Complete tests, diff checks, documentation, and focused commits.

### Hard stop

Do not begin new capability work after the Day 3 selection. Day 4 is for
hardening, final ablation and latency tables, and the adversarial harness.

## Decision log

The authoritative 200-session ablation was run after deterministic exact
limiting was added. Full details are in `docs/day3-ablation-report.md`.

| Config | Exact | Bucket | Dense | Freshness | Score | Hit@10 | MRR | MTTC | Decision/reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.35 | 0.10 | 0.20 | No | 0.825627 | 0.940 | 0.633756 | 2.725 | Reject: freshness is essential |
| 2 | 0.35 | 0.10 | 0.20 | Yes | 0.869976 | 0.995 | 0.664254 | 2.340 | Reject: dense lowers MRR and adds startup cost |
| 3 | 0.35 | 0.10 | 0 | Yes | **0.877011** | **0.995** | **0.690702** | 2.385 | **Ship** |
| 4 | 0.35 | 0 | 0.20 | Yes | 0.863247 | 0.995 | 0.646490 | 2.410 | Reject: bucket evidence helps overall |
| 5 | 0 | 0.10 | 0.20 | Yes | 0.830052 | 0.995 | 0.535506 | 2.405 | Reject: exact is the strongest MRR signal |
| 6 | 0 | 0 | 0 | Yes | 0.862811 | 0.995 | 0.648038 | 2.455 | Keep only as a simple fallback/control |

Production defaults were changed to configuration 3 in `f9f1095`.
