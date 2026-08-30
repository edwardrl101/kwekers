# Train/evaluation split and cross-validation plan

> **Implementation update (2026-08-30):** Milestones 1 through 4 and a
> first-version CI workflow are implemented. The deterministic manifest is
> `data/cv_folds.json`, the runners are `scripts/build_cv_folds.py` and
> `scripts/eval_cv.py`, and integrity tests are in `tests/test_cv_folds.py` and
> `tests/test_eval_cv.py`. Production no-dense scored `0.877011` mean with
> `0.019642` population SD after a complete six-way comparison; see
> `docs/cv-baseline-report.md`. The fixed holdout remains unchanged and
> partially consumed.

## Purpose

This document is the handoff for improving evaluation discipline in the
Kwekers shopping-agent project. It records the current tune/holdout procedure,
where overfitting can occur, what has and has not been validated, and a concrete
plan for introducing grouped, scenario-stratified cross-validation.

The immediate goal is not to improve the public score. It is to obtain a more
reliable estimate of whether ranking changes will generalize to the private
800-session evaluator.

## Current repository state

At the time of this handoff:

- Active development branch: `feat/route-integration`.
- Production ranking configuration:
  - BM25 top 500 as the exclusive candidate pool;
  - exact AND evidence with weight `0.35`;
  - bucket/category evidence with weight `0.10`;
  - cross-turn freshness enabled;
  - shown-product history cleared on intent override;
  - dense retrieval disabled with weight `0.0`.
- Full public-set result:
  - TechnicalScore: `0.877011`;
  - HitRate@10: `0.995000`;
  - MRR: `0.690702`;
  - MTTC: `2.385000`;
  - Efficiency: `0.861500`.
- Combined feature-branch test suite: 54 passing tests after the CV work.
- Full 200-session no-dense evaluation time in the Day 3 environment:
  approximately `30.599` seconds.

Relevant files:

- `starter/agent.py`: production Agent and configurable fusion weights.
- `scripts/eval.py`: fixed-split evaluator and named Day 3 ablations.
- `data/eval_split.json`: committed tune/holdout manifest.
- `data/public_set.jsonl`: 200 public development sessions.
- `data/catalog.jsonl`: frozen 50,000-product catalog.
- `evaluator/local_evaluator.py`: authoritative local evaluator; do not edit to
  improve scores.
- `docs/day3-ablation-report.md`: full Day 3 ablation evidence.
- `CODEX.md`: durable project architecture and status handoff.

Generated result JSON and run logs are not the source of truth for committed
behavior. Evaluation summaries should be recorded in documentation or a
deliberately managed run log.

## Competition evaluation structure

The evaluator selects a hidden target from the frozen product catalog. The
Agent has at most ten turns and returns ten recommendations per turn. A session
ends when the target appears in the first ten valid unique recommendations.

The released development set contains 200 sessions. The final private set is
expected to contain 800 sessions.

Scenario proportions are:

| Scenario | Proportion | Public count |
|---|---:|---:|
| Buying | 40% | 80 |
| Browsing | 40% | 80 |
| Intent override | 15% | 30 |
| Boundary | 5% | 10 |

TechnicalScore is:

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore =
    0.50 * HitRate@10
  + 0.30 * MRR
  + 0.20 * Efficiency
```

HitRate therefore dominates the score, while target rank and first-hit turn
remain material.

## Current tune/holdout procedure

The repository uses one fixed, committed, scenario-stratified split:

| Split | Sessions | Composition | Intended use |
|---|---:|---|---|
| Tune | 140 | 56 buying, 56 browsing, 21 override, 7 boundary | Development and configuration selection |
| Holdout | 60 | 24 buying, 24 browsing, 9 override, 3 boundary | Infrequent finalist checkpoint |
| All public | 200 | 80 buying, 80 browsing, 30 override, 10 boundary | Final public reporting |
| Private | 800 | Unseen | Genuine final evaluation |

The split is defined in `data/eval_split.json`. It should not be casually
regenerated, because shared fixed IDs make teammate comparisons reproducible.

Current commands are:

```bash
python scripts/eval.py --split tune --label experiment-name
python scripts/eval.py --split holdout --label checkpoint-name
python scripts/eval.py --split all --label final-public-report
```

Day 3 followed this approximate sequence:

1. Develop and compare configurations on the 140-session tune split.
2. Take only the two finalists—full freshness and no-dense—to holdout.
3. Select no-dense after it tied on tune and won clearly on holdout.
4. Run the selected configuration and the six-way ablation on all 200 public
   sessions for final reporting.

Finalist results were:

| Split | Configuration | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---|---:|---:|---:|---:|---:|
| Tune, 140 | Full freshness with dense | 0.866845 | 0.992857 | 0.659484 | 2.371429 | 0.862857 |
| Tune, 140 | No dense | **0.867513** | 0.992857 | **0.662664** | 2.385714 | 0.861429 |
| Holdout, 60 | Full freshness with dense | 0.877282 | 1.000000 | 0.675384 | **2.266667** | **0.873333** |
| Holdout, 60 | No dense | **0.899171** | 1.000000 | **0.756124** | 2.383333 | 0.861667 |

The procedure was more disciplined than repeatedly selecting against all 200
sessions, but it remains a single fixed holdout design.

## Is K-fold cross-validation currently performed?

Yes, for fixed configuration stability evaluation. The implementation now has:

- a committed, reproducibly generated five-fold manifest;
- exact scenario stratification in every fold;
- target-ASIN and conservative title-family group constraints;
- a runner that evaluates named configurations on one or all folds;
- mean, population standard deviation, worst-fold, and out-of-fold reporting;
- tests for disjointness, completeness, group isolation, malformed IDs, and
  determinism.

Because the Agent does not fit on sessions, this is not conventional K-1-fold
training. It evaluates predeclared fixed configurations on every held-out fold.
Once fold results influence new design choices, they are development evidence,
not an untouched estimate.

## Why overfitting is possible without model training

The project does not train a conventional predictive model, but it can still
overfit through repeated heuristic and configuration development. Developers
act as the optimizer: every inspected failure influences parser rules, route
weights, freshness behavior, and candidate selection.

### 1. Repeated tuning on 140 sessions

Repeatedly inspecting the same tune failures can specialize the pipeline to
those exact messages, products, categories, or constraint combinations.

This includes changes to:

- parser regular expressions;
- route weights;
- candidate-pool rules;
- override behavior;
- fallback selection;
- question policy;
- exact-match normalization;
- product-specific or category-specific exceptions.

### 2. Holdout leakage through repeated inspection

The 60-session holdout remains independent only while it is used infrequently.
If the system is changed after every holdout result, the holdout becomes a
second tuning set.

The Day 3 selection used the holdout to choose no-dense over dense. This is a
legitimate checkpoint, but it consumes some of the holdout's independence. The
same holdout should not now be used repeatedly for small weight or parser
decisions.

### 3. Full-public-set leakage

The full 200-session ablation was useful for reporting, but once its results are
used to make further decisions, all public sessions have influenced the design.

Future work should clearly distinguish:

```text
reporting a full-set result
```

from:

```text
selecting a change because of the full-set result
```

The second action makes the public set part of the development signal.

### 4. Evaluator-specific assumptions

Current performance uses known evaluator behavior:

- the target does not change during intent override;
- recommendations before the scheduled override cannot count;
- asking `other` reveals up to two undisclosed constraints;
- public messages follow recognizable templates;
- exactly ten guesses carry no direct penalty;
- the public catalog and evaluator are frozen.

These are valid competition optimizations. They remain a generalization risk if
private messages use different paraphrases or the private evaluator differs in
undocumented ways.

### 5. Small scenario subsets

The current holdout contains only:

- nine override sessions;
- three boundary sessions.

A single success or rank movement can materially change these scenario metrics.
Perfect performance on three boundary sessions is weak evidence of robust
boundary behavior.

### 6. Product and category correlation

A session-level random split may place related product families, categories,
or near-duplicate titles on both sides of the split. That can make validation
look more stable than performance on genuinely different products.

Potential grouping keys include:

- target `parent_asin`;
- coarse catalog category;
- normalized title or near-duplicate title family;
- brand/store where reliable;
- a stable product-family fingerprint derived from title and category.

## What K-fold would and would not solve

K-fold cross-validation would provide:

- multiple validation estimates instead of one;
- mean and variance for each configuration;
- visibility into fold-sensitive improvements;
- a better basis for treating small score differences as ties;
- more evidence for choosing the simpler configuration.

It would not provide:

- a genuinely untouched final test set;
- protection against repeatedly designing around all fold results;
- proof that public evaluator templates match the private evaluator;
- independence if near-duplicate products leak across folds;
- reliable per-fold estimates for very small scenarios without careful design.

The private 800-session evaluator must remain the genuine final test.

## Recommended cross-validation design

### Default recommendation: five folds

Use five folds of approximately 40 sessions each.

Five folds are a practical compromise:

- each validation fold is large enough for an overall score;
- every public session is validated exactly once;
- the full run is inexpensive now that dense is disabled;
- five evaluations per configuration remain manageable;
- scenario proportions can be approximately preserved.

Expected per-fold scenario allocation is approximately:

| Scenario | Total | Approximate count per fold |
|---|---:|---:|
| Buying | 80 | 16 |
| Browsing | 80 | 16 |
| Intent override | 30 | 6 |
| Boundary | 10 | 2 |

Boundary will remain noisy at two sessions per fold. Report aggregate boundary
metrics across out-of-fold predictions as well as fold-level variance.

### Stratification

Every fold should preserve scenario proportions as closely as possible. At a
minimum, stratify on `scenario_type`.

Do not use an unstratified shuffle: it can produce folds with too few override
or boundary sessions to interpret.

### Grouping

Prefer stratified group assignment rather than plain stratification.

Groups should prevent closely related sessions from appearing in both the
development and validation portions of a fold. Candidate grouping hierarchy:

1. Exact target `parent_asin`.
2. Conservative normalized-title family.
3. Coarse category where group sizes remain feasible.

Grouping by entire coarse category may be too aggressive because there are few
large categories and it can destroy scenario balance. Start with exact target
and conservative near-duplicate title families, then measure whether meaningful
cross-fold similarity remains.

### Deterministic assignment

Fold assignment must be reproducible:

- use a documented fixed seed;
- sort stable IDs before randomized assignment;
- commit the resulting manifest or commit a deterministic generator plus hash;
- never depend on Python set iteration order;
- validate that rerunning the generator produces identical output.

Suggested manifest shape:

```json
{
  "version": 1,
  "seed": "kwekers-cv-v1",
  "grouping": "target-and-normalized-title-family",
  "folds": [
    {"name": "fold_0", "validation_sample_ids": ["public_..."]},
    {"name": "fold_1", "validation_sample_ids": ["public_..."]},
    {"name": "fold_2", "validation_sample_ids": ["public_..."]},
    {"name": "fold_3", "validation_sample_ids": ["public_..."]},
    {"name": "fold_4", "validation_sample_ids": ["public_..."]}
  ]
}
```

## Important semantic point: there is no fitted model

The current Agent does not fit parameters from a training fold. BM25 and exact
indexes are built from the product catalog, not from evaluator sessions.

Therefore, K-fold evaluation in this project is configuration validation:

```text
For each fold:
    use the other four folds as the development evidence
    freeze one configuration
    evaluate that configuration on the held-out fold
```

If the same already-selected configuration is simply evaluated on all five
folds, the result estimates its fold sensitivity but is not a complete nested
model-selection procedure.

For honest configuration comparison, one of these approaches is needed:

### Option A: predeclare configurations

Define a small finite set of configurations before viewing fold results. Run
all configurations across all folds, compare aggregate performance, select the
winner, and stop.

This is practical for the current heuristic pipeline.

### Option B: nested cross-validation

Use an outer fold for unbiased evaluation and inner folds for selecting weights
or policies. This is statistically stronger but expensive and probably
unnecessary for the small competition project unless systematic weight tuning
begins.

Recommended starting point: Option A with a preregistered configuration list.

## Configuration comparison rules

For every configuration, report:

- mean TechnicalScore across folds;
- standard deviation of TechnicalScore;
- minimum and maximum fold scores;
- mean HitRate@10;
- mean MRR;
- mean MTTC;
- out-of-fold aggregate metrics over all 200 sessions;
- the same metrics by scenario;
- wall-clock runtime;
- code commit and configuration values.

Suggested summary table:

| Config | Mean score | Score SD | Worst fold | OOF Hit@10 | OOF MRR | OOF MTTC | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production no-dense | | | | | | | |
| BM25 + freshness | | | | | | | |
| Candidate configuration | | | | | | | |

Decision rule:

1. Reject changes that reduce HitRate materially.
2. Prefer improvements that are positive on most folds and scenarios.
3. Treat a mean gain smaller than fold-to-fold variability as a tie.
4. In a tie, ship the simpler, faster, and less brittle configuration.
5. Do not tune against the worst fold after inspecting it without beginning a
   new declared experiment.

## Relationship to the existing fixed holdout

Do not delete or silently replace `data/eval_split.json`.

The existing split remains valuable for continuity with historical results.
Use it as follows:

- preserve historical tune/holdout comparisons;
- introduce a separate cross-validation manifest;
- use cross-validation for new configuration stability estimates;
- avoid repeatedly checking the fixed 60-session holdout;
- reserve one final fixed-holdout run for a meaningful frozen checkpoint, if
  the team decides the holdout still has enough independence.

Because Day 3 already used holdout to select no-dense, the holdout must be
described as partially consumed, not fully untouched.

## Proposed implementation plan

### Milestone 1: audit the data and grouping keys

Create a read-only diagnostic that reports:

- duplicate target ASIN counts;
- scenario counts;
- category distribution;
- normalized-title and near-duplicate family sizes;
- candidate grouping schemes and their largest groups;
- whether a proposed grouping makes five-fold stratification feasible.

Do not write the fold manifest until these diagnostics are reviewed.

### Milestone 2: implement deterministic fold generation

Add a script such as:

```text
scripts/build_cv_folds.py
```

Responsibilities:

- load the public sessions and catalog read-only;
- construct stable group IDs;
- allocate groups across five scenario-stratified folds;
- validate disjointness and complete coverage;
- write a versioned manifest;
- print fold and scenario summaries;
- print a manifest hash.

Add unit tests for:

- all 200 session IDs appear exactly once as validation IDs;
- validation folds are mutually disjoint;
- no group crosses validation folds;
- scenario counts are within declared tolerances;
- generation is deterministic;
- malformed or duplicate session IDs fail loudly.

### Milestone 3: implement the cross-validation runner

Extend `scripts/eval.py` or add:

```text
scripts/eval_cv.py
```

Prefer a separate runner if adding fold logic would make the simple fixed-split
CLI harder to understand.

The runner should:

- accept a fold manifest;
- accept named Agent configurations;
- run one fold or all folds;
- preserve per-session results for aggregation;
- compute mean, standard deviation, and worst-fold metrics;
- compute out-of-fold aggregate metrics;
- report per-scenario metrics;
- record runtime and configuration metadata;
- avoid committing generated result blobs by default.

Suggested CLI:

```bash
python scripts/eval_cv.py --config no-dense --folds all
python scripts/eval_cv.py --config bm25-only --folds all
python scripts/eval_cv.py --config candidate-name --folds fold_0
```

### Milestone 4: establish cross-validation baselines

Run at least:

1. Production no-dense:

```text
exact 0.35 + bucket 0.10 + freshness + dense 0
```

2. BM25-only freshness control:

```text
exact 0 + bucket 0 + freshness + dense 0
```

3. Optionally, full freshness with dense only if the environment and dependency
   are deliberately restored for comparison. Dense should not become a default
   requirement for cross-validation.

Do not introduce new ranking changes until these baseline variance estimates
are recorded.

### Milestone 5: integrate cross-validation into CI

Use two evaluation tiers.

#### Pull-request checks

Run on every PR:

- `python -m unittest discover -s tests -v`;
- `git diff --check`;
- fold-manifest validation;
- a small deterministic evaluator smoke subset;
- response-schema invariants;
- no-dense production-default assertion;
- exact AND and freshness/override tests.

Do not run all five folds for every small commit unless CI time is acceptable.

#### Scheduled or manual checks

Run nightly or through manual dispatch:

- all five folds for production and candidate configurations;
- per-scenario and aggregate reports;
- latency and memory measurements;
- regression comparison against the committed production floor;
- artifact upload for reports and per-session results.

Suggested quality gates:

- all unit and manifest tests pass;
- no validation group leakage;
- response-schema failure count is zero;
- candidate HitRate does not regress beyond a declared tolerance;
- score or MRR improvement is stable across folds;
- runtime remains below a declared budget.

## Typical ML CI/CD context

For a larger ML project, CI/CD normally covers four independently versioned
surfaces:

```text
code + data + configuration/model + runtime environment
```

### CI responsibilities

- linting, formatting, typing, and unit tests;
- data-schema and data-quality validation;
- deterministic feature/index generation;
- offline model or retrieval evaluation;
- fairness/safety checks where applicable;
- latency, memory, and artifact-size gates;
- dependency and security scanning;
- reproducibility metadata and artifact hashes.

### CD responsibilities

- build an immutable application/model artifact;
- register model, index, and data versions;
- deploy to staging;
- run integration and shadow tests;
- canary a small traffic percentage;
- monitor quality, drift, errors, latency, and cost;
- promote or automatically roll back.

This competition project is local rather than a hosted ML service. Its
deployment equivalent is a reproducible submission artifact containing the
correct Agent code and permitted local assets.

## Required run metadata

Every serious evaluation should record:

- Git commit;
- active branch;
- Agent configuration;
- catalog checksum;
- public-set checksum;
- fold/split manifest checksum;
- Python version;
- dependency versions;
- random seed;
- dense enabled/disabled state;
- runtime;
- overall metrics;
- per-scenario metrics;
- fold-level metrics where applicable.

This makes a metric traceable to the exact code, data, and configuration that
produced it.

## Actions to avoid

- Do not modify `evaluator/local_evaluator.py` to improve reported metrics.
- Do not regenerate the current fixed split without an explicit migration.
- Do not use the 60-session holdout after every small change.
- Do not select changes from the full 200-session score and then describe that
  score as unbiased validation.
- Do not create folds using unordered set iteration.
- Do not allow the same product or near-duplicate group to cross folds once a
  grouping policy is adopted.
- Do not report only the best fold.
- Do not hide failed or negative configuration results.
- Do not infer private-set performance directly from the public score.
- Do not add K-fold complexity without tests proving fold integrity.

## Open decisions for the next conversation

1. Should grouping use only target ASIN, or target plus conservative
   near-duplicate title families?
2. Is five-fold cross-validation the desired cost/variance tradeoff, or should
   the team use repeated stratified holdout?
3. Should the current fixed 60-session holdout be frozen completely after Day
   3, or used once for a final cross-validation-selected checkpoint?
4. Which configurations will be preregistered for the first CV comparison?
5. What score, HitRate, and runtime regression tolerances should CI enforce?
6. Should cross-validation reports be committed as Markdown summaries, stored
   only as CI artifacts, or both?
7. What ownership and retention policy should apply to `runs/`?
8. Should near-duplicate grouping reuse `src/dedup.py`, and if so, which
   similarity threshold is conservative enough to avoid grouping unrelated
   products?

## Recommended next-conversation starting point

Start with a read-only data/grouping audit. Do not immediately implement fold
assignment.

Suggested first steps:

1. Read `CODEX.md`, this file, `data/eval_split.json`, `scripts/eval.py`, and
   `src/dedup.py`.
2. Confirm the active branch and current production defaults.
3. Measure duplicate targets and candidate title-family group sizes.
4. Compare target-only grouping against conservative near-duplicate grouping.
5. Propose fold-balance tolerances before generating a manifest.
6. Implement the generator and its integrity tests as the first committed
   milestone.

## Definition of done

The split/cross-validation work is complete only when:

- a documented grouping policy has been selected;
- a deterministic five-fold manifest exists;
- all 200 sessions appear in exactly one validation fold;
- no declared group crosses folds;
- scenario balance is verified within declared tolerances;
- fold-generation tests pass;
- an all-fold evaluator reports mean, standard deviation, worst fold, and
  out-of-fold aggregate metrics;
- production no-dense and BM25-only baselines have been measured;
- CI has a fast integrity/smoke tier and a scheduled full-CV tier;
- results identify code, data, manifest, configuration, and runtime versions;
- the fixed holdout's partially consumed status is documented;
- no remaining claim incorrectly describes the current procedure as K-fold
  cross-validation.
