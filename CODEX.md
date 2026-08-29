# CODEX Project Handoff

This file is the durable context for continuing work on the Kwekers TechJam
shopping-agent project. Read it before modifying retrieval, dialog behavior, or
evaluation tooling. It records the state verified on 2026-08-29 against
`main` commit `e5a94ec` after fetching `origin/main`.

> **Current-main integration warning:** the latest merge changed
> `ExactRoute` from `query(text, limit)` to `exact_matches(constraints)`, but
> `starter/agent.py` still calls the old common `query()` interface. Exact
> evidence is therefore caught as a route failure and contributes nothing on
> current `main`. The green unit suite does not detect this. Repair and measure
> this adapter before treating the pre-merge `0.820121` score as current.

## 1. Objective and deliverable

The competition deliverable is exactly one Python class:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict: ...
```

The evaluator selects a hidden target from a frozen 50,000-product catalog. It
allows at most 10 turns and stops at the first turn where the target appears in
the first 10 valid, unique recommendations. There are 200 public development
sessions and 800 private sessions.

Scenarios have this fixed mix:

- 40% buying
- 40% browsing
- 15% intent override
- 5% boundary

The goal is to retrieve the target early and rank it highly, while remaining
fully local, robust, and inexpensive.

## 2. Evaluator facts that must not be forgotten

`respond()` must return a dictionary shaped like:

```python
{
    "message": "customer-facing text",
    "ask_attribute": "other",
    "recommendations": [{"parent_asin": "B000..."}, ...],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

Hard requirements and free wins:

1. `message` must be a string. Otherwise the evaluator discards the entire
   response and replaces it with an empty response.
2. Always return exactly 10 recommendations, including turn 1. Wrong guesses
   have no direct penalty.
3. Never return `ask_attribute: None`. It reveals nothing and burns a turn.
4. Recommendation normalization silently removes invalid and duplicate ASINs
   and scores only the first 10 valid unique IDs.
5. Recommendation `score` fields are accepted but ignored by the evaluator.
6. `usage` is optional and not part of TechnicalScore.

Allowed question attributes are:

```text
category, material, color, size, style, brand, budget,
feature, use_case, other, or null
```

The current safe strategy always asks `other`. In the public evaluator,
`customer_reply(..., "other", ...)` reveals up to two undisclosed constraints
without filtering by type. An intent card normally contains at most four useful
constraints: up to two hard constraints and two soft preferences. After the
card is drained, asking must still remain non-null.

In intent-override sessions, a recommendation made before the scheduled
override cannot count as a hit. The target product itself does not change.

Do not edit `evaluator/local_evaluator.py` when measuring or reporting scores.

## 3. Metrics and optimization implications

```text
HitRate@10 = successful sessions / N
MRR = mean(1 / target rank), with misses equal to 0
MTTC = mean(first hit turn), with misses assigned turn 11
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

A session hit at turn `t` and rank `r` contributes, relative to a miss:

```text
0.50 + 0.30 / r + 0.02 * (11 - t)
```

Consequences:

- Any top-10 hit is valuable because HitRate carries half the score.
- Moving a target toward rank 1 materially improves MRR.
- Asking useful questions and hitting earlier improves Efficiency.
- Retrieve wide first. A reranker cannot rescue a target that never entered its
  candidate pool.

## 4. Current verified architecture

Primary implementation: `starter/agent.py`.

Current turn flow:

```text
new customer message
  -> update per-session SlotState
  -> retain category and all active constraints
  -> build cumulative retrieval query
  -> run bucket, BM25, and dense routes; attempt exact
  -> take BM25's top 500 as the exclusive candidate pool
  -> normalize BM25 rank
  -> add bucket and dense evidence; exact is currently empty due API mismatch
  -> sort by fused score; BM25 rank breaks ties
  -> return the first 10 valid unique IDs
  -> deterministic random fill only if fewer than 10 exist
  -> ask `other`
```

Routes are logically independent but currently execute sequentially, each
inside its own `try/except`. One route failure cannot zero the response.
That safety behavior masks the current exact-route incompatibility: evaluation
continues instead of crashing, but silently loses the exact signal.

### 4.1 Accumulated dialog context

`reset()` creates a `src.dialog.SlotState` for each session. On every turn,
`_update_retrieval_context()`:

- parses newly disclosed constraints;
- retains the initial category;
- keeps previous active constraints;
- demotes the old preference when an override message is detected;
- builds a compact query such as `Shoes leather color: black`.

Example:

```text
Turn 1: I'm looking for Shirts. Department: Womens
Turn 3: Actually, ignore my earlier preference. What I need is: wool.
```

The active retrieval query changes from:

```text
Shirts Department: Womens
```

to:

```text
Shirts wool
```

The category and unrelated active constraints survive the override.

`QuestionPolicy` exists in `src/dialog.py` but is not connected. The agent still
records and asks `other` every turn.

### 4.2 Bucket evidence

`src.buckets.BucketRoute` extracts the coarse category from the original
opening-message template and returns catalog-ordered category members with
score `1.0`. The stored opening message lets category evidence remain available
after turn 1.

Bucket matches add `0.10` during fusion. There is currently no explicit penalty
for bucket mismatch.

Latest main also adds `BucketRoute.filter_by_category(pool, message)`. It
preserves BM25 order, keeps only category members, and safely returns the
original pool if parsing/bucket lookup fails or filtering would empty a usable
pool. `Agent` does **not** call this new filter yet; it still uses the older
positive-boost path. Reconcile these two designs deliberately rather than
assuming the new filter is already end-to-end.

### 4.3 Exact evidence

Latest `src.exact.ExactRoute` indexes cleaned feature strings, `Key: Value`
details, bare materials, and numeric prices. Its public retrieval method is now:

```python
exact_matches(constraints: list[str]) -> list[str]
```

It handles material tokens, `budget around $X` with 15% tolerance, and an
intersection over the non-empty constraint match sets.

`starter.Agent._route_exact()` has not been adapted. It still invokes
`_query_route()`, which calls `route.query(text, limit=500)`. Since the merged
`ExactRoute` has no `query()`, `_route_candidates()` catches `AttributeError`
and records an empty exact result for that turn. The configured `+0.35` exact
boost is therefore inactive on current main.

The repair should pass `session["active_constraints"]` to `exact_matches()`,
validate/deduplicate returned ASINs, cap at 500, and map them to
`[(asin, 1.0), ...]`. Add a real integration test using `ExactRoute`, not only a
fake object implementing the obsolete `query()` contract.

### 4.4 BM25 candidate generation

`src.retrieval.BM25Route` uses in-memory SQLite FTS5. It searches title,
categories, features, details, store, and description, favoring title and
category fields. It uses a strictness cascade:

```text
exact constraint phrases -> all content terms -> any content term
```

Tier offsets keep strict matches above loose matches. The agent requests
`limit=500`. BM25 is the exclusive v1 candidate-admission route.

### 4.5 Dense evidence

`src.retrieval.DenseRoute` uses `BAAI/bge-small-en-v1.5` with 384-dimensional,
normalized embeddings. Query similarity is cosine similarity implemented as a
matrix-vector product.

The supplied cache is:

```text
data/dense_cache.npz
50,000 ASINs
embedding shape 50,000 x 384
approximately 40.4 MB on disk
```

The cache is intentionally gitignored. Use the supplied cache; do not regenerate
it during normal development or evaluation. `Agent` passes
`build_if_missing=False`, so a missing or catalog-mismatched cache disables
dense retrieval instead of starting an encode. Only
`scripts/build_dense_cache.py` permits generation as a recovery operation.

The model loader prefers local Hugging Face files first to avoid offline
metadata retries. `requirements-dense.txt` pins:

```text
sentence-transformers==6.0.0
```

On the pre-merge feature branch, a real smoke run measured about 21.25 seconds
for full Agent initialization and 0.256 seconds for one fused response. A
post-merge full evaluation attempt did not reach model progress after roughly
four minutes and was stopped; investigate startup separately from ranking.

### 4.6 Fusion mathematics

Constants in `starter/agent.py`:

```python
ROUTE_CANDIDATE_LIMIT = 500
EXACT_MATCH_BOOST = 0.35
BUCKET_MATCH_BOOST = 0.10
DENSE_SIMILARITY_WEIGHT = 0.20
```

The intended configured formula for a BM25 candidate `i` is:

```text
fused(i) = normalized_bm25_rank(i)
         + 0.35 * exact_match(i)
         + 0.10 * bucket_match(i)
         + 0.20 * normalized_dense_score(i)
```

BM25 rank is mapped linearly from rank 1 to `1.0` and the last pool rank to
`0.0`. Dense results are min-max normalized within the returned dense list.
Original BM25 rank is the deterministic tie-breaker.

On current main, `exact_match(i)` is always zero in end-to-end Agent execution
because of the adapter mismatch described above.

Do not sum raw route scores. BM25 uses tiered values around thousands, exact and
bucket use `1.0`, and dense cosine is roughly in `[-1, 1]`; raw addition would
make the largest numerical scale dominate regardless of signal quality.

This is deliberately not equal-weight RRF. The weights are reasonable v1
defaults, not systematically tuned values.

### 4.7 Final recommendation selection

Every turn recomputes retrieval and fusion from the cumulative active query and
returns the current fused top 10. New constraints often change the list, but
there is no cross-turn `shown` set. Products may repeat across turns.

Override constraint replacement is implemented, but `shown` clearing is not:
there is no `shown` state to clear. Adding a deliberate freshness/reset policy
is an outstanding Day-3 decision, not something already present.

`top_k` is ignored intentionally; competition output is fixed at 10.

If retrieval returns fewer than 10 valid unique catalog IDs, `_random_fill()`
uses a deterministic seed derived from the user profile and turn. If the
catalog itself is missing or too small, placeholders preserve response shape,
although placeholders are invalid for scoring.

## 5. Verified evaluation results

Starter reference from the repository:

```text
HitRate@10 0.125
MRR 0.068034
MTTC 9.81
TechnicalScore approximately 0.107
```

Earlier route-concatenation checkpoint:

| Split | TechnicalScore | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Tune, 140 | 0.658371 | 0.764286 | 0.432188 | 3.671429 |
| Holdout, 60 | 0.656248 | 0.783333 | 0.393049 | 3.666667 |

BM25-base fusion v1, measured on commit `b5637f9`/feature integration before the
latest main route-API merge:

| Split | TechnicalScore | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Tune, 140 | **0.818984** | 0.942857 | 0.597089 | 2.578571 | 0.842143 |
| Holdout, 60 | **0.822776** | 0.950000 | 0.598142 | 2.583333 | 0.841667 |
| All, 200 | **0.820121** | 0.945000 | 0.597405 | 2.580000 | 0.842000 |

Full 200-session scenario results:

| Scenario | N | TechnicalScore | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Buying | 80 | 0.819759 | 0.950000 | 0.552530 | 2.050000 | 0.895000 |
| Browsing | 80 | 0.828790 | 0.950000 | 0.607634 | 2.425000 | 0.857500 |
| Intent override | 30 | 0.799179 | 0.900000 | 0.717262 | 4.300000 | 0.670000 |
| Boundary | 10 | 0.816500 | 1.000000 | 0.515000 | 2.900000 | 0.810000 |

The observed pre-merge wall-clock time for one full 200-session evaluator
invocation was approximately 74 seconds with the supplied dense cache present.

These numbers are a historical working floor, **not a verified score for
`e5a94ec` current main**. A post-merge full run was attempted while preparing
this handoff but was stopped after approximately four minutes without progress,
so no replacement score was recorded. Re-run tune first after repairing the
exact adapter, then use holdout only at the next meaningful checkpoint.

Treat the public set as a training signal, not proof of private-set quality.
Tune on the 140 sessions and inspect the 60-session holdout only at meaningful
checkpoints.

## 6. Evaluation and development commands

Install optional dense dependencies:

```bash
python -m pip install -r requirements-dense.txt
```

Run all tests:

```bash
python -m unittest discover -s tests -v
```

The latest-main verified count is 27 passing tests. This suite currently has a
coverage hole: it does not instantiate the real merged `ExactRoute` through
`Agent`, so it passes despite exact evidence being disconnected.

Run tune, holdout, or all sessions:

```bash
python scripts/eval.py --split tune --label experiment-name
python scripts/eval.py --split holdout --label checkpoint-name
python scripts/eval.py --split all --label report-name
```

`scripts/eval.py` prints overall and per-scenario metrics, writes ignored
`results_<split>.json`, and appends a timestamped row to `runs/runs.csv` unless
`--runs` overrides the path.

Run the evaluator directly:

```bash
python -m evaluator.local_evaluator
```

Trace one released sample for learning/debugging:

```bash
python scripts/trace_evaluate.py --sample-id public_0001
```

Relevant onboarding documents:

- `docs/agent_onboarding_guide.md`
- `docs/local_evaluator_guide.md`
- `docs/public_0001_trace_walkthrough.md`
- `docs/competition_specification.md`

## 7. Fixed split and result discipline

`data/eval_split.json` is the committed, scenario-stratified split:

```text
tune: 140 = 56 buying + 56 browsing + 21 override + 7 boundary
holdout: 60 = 24 buying + 24 browsing + 9 override + 3 boundary
```

Do not change this manifest casually. Everyone must report the same split.

Generated `results.json`, `results_*.json`, `data/catalog.jsonl`, and
`data/dense_cache.npz` are gitignored. Evaluation summaries belong in the run
CSV or documentation, not as committed result blobs unless the team explicitly
changes that policy.

At the time this handoff was written, `data/SHA256SUMS` and `runs/` were
untracked. Do not automatically delete or commit them; determine ownership and
the team's intended run-log policy first.

## 8. Completed checklist and post-merge status

The original Member-1/Edward pipeline checklist was completed and measured on
the feature branch:

- end-to-end Agent assembled;
- bucket, exact, BM25, and dense routes connected at that checkpoint;
- exactly 10 recommendations every turn;
- string `message` guaranteed;
- non-null `ask_attribute` guaranteed;
- BM25 retrieves 500 candidates;
- route initialization and calls are failure-isolated;
- route identity and scores are preserved;
- BM25-base fusion replaces fixed route concatenation;
- `SlotState` accumulation and override demotion are connected;
- fixed 140/60 split exists;
- evaluator tooling prints scenario metrics and logs runs;
- tune, holdout, and full-set checkpoints are reported;
- 26 tests passed at that checkpoint.

After PR merge `e5a94ec`, main has 27 passing tests and new teammate bucket and
exact implementations, but exact is no longer wired through Agent. Therefore,
do not report “all four routes connected” for current main until the adapter is
fixed and an end-to-end test proves it.

Important commits:

```text
1b084a4 feat: widen retrieval and preserve route scores
b5637f9 feat: add BM25-base retrieval fusion
f3efba4 final feature-branch integration before PR merge
e5a94ec latest origin/main merge audited for this updated handoff
```

## 9. Known limitations and open decisions

The first two items are integration work introduced/exposed by the latest main
merge. The remaining items are optimization work:

1. **Exact adapter is broken on main.** `ExactRoute.exact_matches()` replaced
   `query()`, but Agent still calls `query()`. Failure isolation hides the error
   and exact contributes an empty list.
2. **New bucket filter is not wired.** `filter_by_category()` exists and has
   safety tests, but Agent still uses category membership only as a `+0.10`
   boost. Decide whether filtering precedes fusion or remains optional evidence.
3. **No candidate rescue.** Exact or dense results outside BM25's top 500 cannot
   enter the final pool. Measure BM25 recall@500 before changing this.
4. **Mostly positive fusion.** Exact, bucket, and dense promote matches. There is
   no explicit hard-constraint violation penalty; nonmatches only move down
   relatively when another product is promoted.
5. **Untuned weights.** `0.35`, `0.10`, and `0.20` are initial values. Tune only
   on the 140-session split and guard against overfitting.
6. **No shown/freshness policy.** Recommendations can repeat between turns.
   Decide whether to exclude shown IDs normally and whether to clear that set on
   override. This can materially affect Day-3 performance.
7. **QuestionPolicy disconnected.** Always asking `other` is a strong floor but
   does not use candidate-pool information gain after constraints are exhausted.
8. **Routes are sequential.** Concurrency could reduce latency after correctness
   and diagnostics are stable.
9. **User profile is unused for ranking.** It currently contributes only to the
   deterministic fallback seed.
10. **Dense normalization is simplistic.** Min-max normalization makes the worst
   returned dense candidate equal to zero and can be sensitive to score range.
11. **NgramRoute is unused.** `src.retrieval.NgramRoute` exists for fuzzy and
    truncation-robust matching but is not part of the current Agent.
12. **Limited observability.** Route constructor errors are stored in
    `Agent._route_errors`, but per-candidate route ranks and fused scores are not
    exposed by a trace tool.
13. **Historical remaining misses.** The pre-merge full-set HitRate@10 was 0.945,
    leaving 11 public misses. Re-establish the current-main score before using
    that miss list as the next tuning set.

## 10. Recommended next sequence

Keep changes measurable and commit every major milestone, per user request.

1. Adapt `Agent._route_exact()` to `ExactRoute.exact_matches()` and add a real
   integration test proving exact evidence reaches `_fuse_bm25_pool()`.
2. Decide how `BucketRoute.filter_by_category()` composes with the existing
   positive bucket boost; add an Agent integration test for the chosen behavior.
3. Run tune and record a post-merge baseline. Run holdout only after the
   integration repair is stable.
4. Add a read-only diagnostic/trace mode that reports, for one session and turn:
   cumulative slots, route ranks/scores, normalized features, fused score, and
   whether the target was absent from BM25 top 500 or merely reranked poorly.
5. Classify current misses into candidate-generation, parsing/override,
   fusion, and question-strategy failures.
6. Measure BM25 recall@500 and conditional reranking success. This decides
   whether candidate rescue is worth adding.
7. Add candidate-pool attribute statistics and connect `QuestionPolicy` after
   the current `other` strategy drains useful constraints.
8. Add explicit, conservative constraint-violation features. Prefer demotion to
   hard filtering until parser precision is proven.
9. Tune fusion weights on tune only; use holdout only for checkpoints.
10. Evaluate a bounded rescue lane such as BM25 top 500 plus high-confidence
   exact and a small number of top dense candidates.
11. Consider cross-turn shown/freshness behavior, including explicit clearing on
   override, and measure it independently.
12. Optimize latency or route concurrency only after ranking behavior is
   observable and stable.

Useful decomposition:

```text
P(Hit@10)
  = P(target in candidate pool)
  * P(target reranked into top 10 | target in candidate pool)
```

Do not attempt to fix a candidate-generation miss by tuning reranker weights.

## 11. Engineering and Git practices

- Preserve user-owned or unrelated working-tree changes.
- Use `rg`/`rg --files` for repository search.
- Use `apply_patch` for manual file edits.
- Never modify evaluator behavior to improve a reported score.
- Keep the dense cache and generated result JSON files out of Git.
- The repository currently ignores the entire `scripts/` directory. Existing
  tracked scripts remain tracked, but a new script may require an intentional
  `git add -f`; review it carefully before doing so.
- Run the 27-test suite and `git diff --check` before each major commit. Add the
  missing real ExactRoute-through-Agent integration test before relying on green
  tests as proof that every route is wired.
- Commit every major change with a focused message, as explicitly requested by
  the user. Do not automatically push unless asked.
- Avoid committing `data/SHA256SUMS` or `runs/` until their ownership/policy is
  confirmed.
- Never use destructive Git cleanup to remove unrelated work.

## 12. Definition of the intended milestone

The first pipeline milestone was implemented and measured before the latest
route-API merge:

```text
accumulated intent
  -> BM25 top 500
  -> exact/category/dense evidence
  -> normalized interpretable fusion
  -> current top 10
```

The next agent should preserve this working floor and improve one measured
failure mode at a time. On current main, first restore the exact adapter so the
effective flow again matches this diagram; until then it is BM25 plus category
and dense evidence only.
