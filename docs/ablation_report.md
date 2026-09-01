# Day 3 ablation report

## Outcome

Ship the Member 1 pipeline with cross-turn freshness, exact AND evidence at
`0.35`, bucket evidence at `0.10`, and dense disabled. This configuration
scores **0.877011** on all 200 public sessions, with Hit@10 **0.995**, MRR
**0.690702**, and MTTC **2.385**.

The production configuration improves on both reported Day 3 comparison points:

- historical Member 1 full stack: `0.820121`;
- Member 4 BM25-plus-freshness reference: `0.8553`.

All authoritative rows below were run after exact results were made
deterministic. Generated result JSON remains gitignored; run summaries are kept
locally in `runs/runs.csv`.

## Implementation checkpoints

| Commit | Milestone |
|---|---|
| `a7dc2aa` | Cherry-pick Member 3's ExactRoute `query()` and unmatched-constraint fix |
| `cd91365` | Pass all active constraints through Agent in one exact query; add real integration coverage |
| `534dbba` | Add cross-turn freshness and override reset |
| `071cf76` | Add named, reproducible Day 3 ablation configurations |
| `dc7b5fd` | Sort exact intersections before limiting to eliminate process-dependent subsets |
| `f9f1095` | Promote the measured no-dense configuration to production defaults |

The Day 3 implementation introduced and verified a 39-test suite. After the
commit stack was moved onto refreshed `origin/main`, the combined feature branch
contains **48 passing tests**, including nine inherited near-duplicate utility
tests.

A timed production run over all 200 sessions took **30.599 seconds** in the
Day 3 development environment. This is less than half the historical
approximately 74-second dense-backed run, although wall time remains
machine-dependent.

## Exact-set diagnostics

The checked-in `src/exact.py` accumulation benchmark was run against the 80
buying-session targets. It intersects the first one through four indexed target
feature constraints, so this is a route precision/coverage diagnostic rather
than a reconstruction of the evaluator's disclosure order.

| Accumulated constraints | Target coverage | Median candidate set |
|---:|---:|---:|
| 1 | 80/80 (100%) | 54 |
| 2 | 80/80 (100%) | 4 |
| 3 | 80/80 (100%) | 1 |
| 4 | 80/80 (100%) | 1 |

The result supports the AND integration: additional compatible constraints
rapidly turn broad exact evidence into a high-precision set without losing the
target in this benchmark.

## Split checkpoints

The two finalists were full freshness and freshness without dense.

| Split | Configuration | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---|---:|---:|---:|---:|---:|
| Tune, 140 | Full freshness | 0.866845 | 0.992857 | 0.659484 | 2.371429 | 0.862857 |
| Tune, 140 | No dense | **0.867513** | 0.992857 | **0.662664** | 2.385714 | 0.861429 |
| Holdout, 60 | Full freshness | 0.877282 | 1.000000 | 0.675384 | **2.266667** | **0.873333** |
| Holdout, 60 | No dense | **0.899171** | 1.000000 | **0.756124** | 2.383333 | 0.861667 |

No-dense is effectively tied on tune and wins holdout decisively through MRR.
It also removes dense-model loading, the optional dependency, and approximately
40 MB of runtime cache dependency from the production path.

## Full 200-session ablation

| Config | Exact | Bucket | Dense | Freshness | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current baseline | 0.35 | 0.10 | 0.20 | No | 0.825627 | 0.940 | 0.633756 | 2.725 | 0.827500 |
| Full freshness | 0.35 | 0.10 | 0.20 | Yes | 0.869976 | 0.995 | 0.664254 | **2.340** | **0.866000** |
| **No dense (ship)** | **0.35** | **0.10** | **0** | **Yes** | **0.877011** | **0.995** | **0.690702** | 2.385 | 0.861500 |
| No bucket | 0.35 | 0 | 0.20 | Yes | 0.863247 | 0.995 | 0.646490 | 2.410 | 0.859000 |
| No exact | 0 | 0.10 | 0.20 | Yes | 0.830052 | 0.995 | 0.535506 | 2.405 | 0.859500 |
| BM25 only | 0 | 0 | 0 | Yes | 0.862811 | 0.995 | 0.648038 | 2.455 | 0.854500 |

## Per-scenario results

### Buying (80)

| Config | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Current baseline | 0.809257 | 0.925000 | 0.583358 | 2.412500 | 0.858750 |
| Full freshness | 0.860674 | 0.987500 | 0.618080 | 1.925000 | 0.907500 |
| No dense | 0.860433 | 0.987500 | 0.618110 | 1.937500 | 0.906250 |
| No bucket | **0.864692** | 0.987500 | **0.639807** | 2.050000 | 0.895000 |
| No exact | 0.826707 | 0.987500 | 0.509856 | 2.000000 | 0.900000 |
| BM25 only | 0.856559 | 0.987500 | 0.610198 | 2.012500 | 0.898750 |

### Browsing (80)

| Config | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Current baseline | 0.842680 | 0.962500 | 0.629767 | 2.375000 | 0.862500 |
| Full freshness | 0.876414 | 1.000000 | 0.662212 | **2.112500** | **0.888750** |
| **No dense** | **0.891164** | **1.000000** | **0.718046** | 2.212500 | 0.878750 |
| No bucket | 0.860071 | 1.000000 | 0.609405 | 2.137500 | 0.886250 |
| No exact | 0.832193 | 1.000000 | 0.518978 | 2.175000 | 0.882500 |
| BM25 only | 0.867372 | 1.000000 | 0.641240 | 2.250000 | 0.875000 |

### Intent override (30)

| Config | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Current baseline | 0.817750 | 0.900000 | 0.785833 | 4.400000 | 0.660000 |
| **Full freshness** | **0.894250** | **1.000000** | **0.838611** | 3.866667 | 0.713333 |
| No dense | 0.893274 | 1.000000 | 0.833135 | **3.833333** | **0.716667** |
| No bucket | 0.875500 | 1.000000 | 0.778333 | 3.900000 | 0.710000 |
| No exact | 0.827611 | 1.000000 | 0.618704 | 3.900000 | 0.710000 |
| BM25 only | 0.863052 | 1.000000 | 0.741283 | 3.966667 | 0.703333 |

### Boundary (10)

| Config | Score | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Current baseline | 0.843786 | 1.000000 | 0.612619 | 3.000000 | 0.800000 |
| Full freshness | 0.820071 | 1.000000 | 0.526905 | **2.900000** | **0.810000** |
| No dense | 0.847619 | 1.000000 | 0.625397 | 3.000000 | 0.800000 |
| No bucket | 0.840333 | 1.000000 | 0.601111 | 3.000000 | 0.800000 |
| No exact | 0.847000 | 1.000000 | 0.623333 | 3.000000 | 0.800000 |
| **BM25 only** | **0.875619** | **1.000000** | **0.725397** | 3.100000 | 0.790000 |

The boundary split has only ten sessions, so its isolated winner should not
override the aggregate result.

## Decisions

- **Ship freshness.** It raises the full score by `0.044349` and Hit@10 by
  `0.055` relative to the no-freshness current baseline.
- **Ship exact AND evidence.** Removing exact lowers full MRR by `0.128748`
  relative to the dense-backed freshness configuration and drops score by
  `0.039924`.
- **Ship the bucket boost.** Removing it lowers full score by `0.006729`
  relative to the comparable dense-backed freshness configuration.
- **Disable dense.** Removing it raises full score by `0.007035` and MRR by
  `0.026448`, while also reducing startup and dependency risk.
- **Retain BM25-only as a control/fallback, not the primary path.** The selected
  system improves its score by `0.014200` and MRR by `0.042664`.

No bucket hard filter, exact-set promotion, near-duplicate suppression, or
rotating question policy was added to the production path without a separate
measured win.
