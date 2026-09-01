# Kwekers — Multi-Turn Conversational Shopping Agent

A conversational shopping agent for TikTok TechJam Track 4. It finds one hidden target product out of a 50,000-item clothing catalog by asking clarifying questions across up to ten turns, then ranks it into the top 10.

**Shipped TechnicalScore: `0.891084`** on the 200-session public development set (starter baseline: `0.10671`). HitRate@10 `1.000` (cumulative across up to 10 turns, not single-turn accuracy), MRR `0.707615`, MTTC `2.060` turns. The scored path is fully deterministic and offline: zero network calls, zero model cost.

## What it does

Every turn, the agent returns 10 ranked `parent_asin` values and one clarification question. It accumulates what the customer has revealed across turns, demotes (never deletes) a preference the customer explicitly overrides, and excludes already-shown products from future turns unless an override clears that exclusion. The conversation ends the moment the target appears in the returned top 10, or at turn 10.

Sessions come in four types, mixed 40/40/15/5: **buying** (states a hard requirement up front), **browsing** (starts vague), **intent override** (changes their mind on turn 3 or 4), **boundary** (has no preference on some attribute).

```
TechnicalScore = 0.50 x HitRate@10 + 0.30 x MRR + 0.20 x Efficiency
Efficiency      = clip((11 - MTTC) / 10, 0, 1)
```

## Architecture

```
customer message
      |
regex-first parsing + override detection        (src/dialog.py)
      |
optional LLM fallback, regex-miss only, OFF by default
   -> proposal must clean through clean_constraint()
   -> and match the ExactRoute catalog index, or it is discarded
      |
accumulated SlotState: constraints persist across turns,
overrides demote the old preference, never delete it
      |
BM25 top-500 pool                                (src/retrieval.py)
   + exact AND-intersection evidence, weight 0.35 (src/exact.py)
   + category bucket-membership evidence, 0.10    (src/buckets.py)
   + dense cosine evidence, weight 0.00, disabled  (src/retrieval.py)
      |
normalized fusion: BM25 rank breaks ties          (starter/agent.py)
      |
fresh top 10, excluding already-shown, override clears the exclusion
      |
deterministic customer explanation, optional LLM rewrite (fails closed)
```

BM25 is the only route that can add a candidate. Exact, bucket, and dense evidence can only reorder what BM25 already retrieved. The LLM layer never touches product selection: it can only normalize incoming phrasing or rewrite outgoing prose, and every proposed constraint must independently match something already in the catalog before it is accepted.

## Results

| Metric | Shipped | Starter baseline |
|---|---:|---:|
| TechnicalScore | **0.891084** | 0.10671 |
| HitRate@10 | 1.000 | 0.125 |
| MRR | 0.707615 | 0.068034 |
| MTTC | 2.060 | 9.81 |

Per scenario (public set): buying MRR 0.6045 / MTTC 1.325, browsing MRR 0.7649 / MTTC 2.0875, intent override MRR 0.8481 / MTTC 3.667, boundary MRR 0.6528 / MTTC 2.9.

Full ablations, statistical methodology, and the adversarial robustness matrix are in `docs/ablation_report.md`, `results/bootstrap.json`, and `results/adversarial_robustness_report.md`.

## Key findings

- **Ranking, not retrieval, is the bottleneck.** BM25 alone reaches Recall@10 of 0.221 on turn 1 but Recall@500 of 0.950. Almost everything the agent needs is already inside a pool of 500 candidates; the loss happens between rank 10 and rank 500.
- **Conversation handling outweighs adding retrieval routes.** Freshness (excluding shown products, clearing that on override) raised the full score by 0.044 and HitRate@10 by 0.055. Adding a second retrieval route (BM25 + character n-grams) made things worse, 0.585 versus 0.630 for BM25 alone.
- **The system is template dependent, measured with our own adversarial harness.** Scores hold at 0.891 with no perturbation and fall to 0.420 under semantic paraphrasing (`scripts/adversarial.py`, four levels). The exact-match route, worth 0.155 MRR, depends on the simulator quoting catalog metadata close to verbatim, and that assumption is what breaks first.

## Getting started

Python 3.10+.

```bash
# 1. Get the frozen catalog (not stored in Git; GitHub Release asset)
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
# verify against the published SHA256SUMS file

# 2. Install dependencies for the scored path
python3 -m pip install -r requirements.txt

# 3. Run the official evaluator
python3 -m evaluator.local_evaluator
# -> recommended_technical_score: 0.891084

# 4. Confirm zero LLM calls and byte-exact reproduction
python3 -m unittest tests.test_frozen_baseline -v

# 5. Full test suite
python3 -m unittest discover -v

# 6. Interactive demo — calls the real production Agent
python3 demo_server.py
# open http://127.0.0.1:8000
```

No API key or network access is required for any of the above.

### Team evaluation workflow

```bash
python scripts/eval.py --split tune       # scenario-stratified 140-session dev split
python scripts/eval.py --split holdout    # sealed 60-session checkpoint split, use sparingly
python scripts/eval.py --split all        # all 200 public sessions, for reporting a score
python scripts/eval_cv.py --config no-dense --folds all   # 5-fold stability check
```

`scripts/eval.py` prints overall and per-scenario metrics and appends a timestamped row to `runs/runs.csv`. Never tune decisions on the holdout split or on all 200 sessions.

### Optional LLM layer

The scored ranking path is offline by default. An optional OpenRouter layer (regex-miss constraint normalization, paraphrased override detection, LLM-written explanations) is behind four flags that default OFF: `ENABLE_LLM_NORMALIZE`, `ENABLE_LLM_OVERRIDE`, `ENABLE_LLM_MESSAGE`, `ENABLE_CONFIDENCE`. Copy `.env.example` to an untracked `.env`, add a key, then:

```bash
python scripts/llm_smoke.py
```

See `docs/llm_integration_guide.md` for the client's failure model, security boundaries, and reproducibility commands. Never commit `.env` or an API key.

## Repository layout

```text
starter/agent.py                  Agent entry point — the only file the evaluator calls
src/dialog.py                     scenario routing, slot state, question policy
src/retrieval.py                  BM25 (only route that can add a candidate), dense, n-gram
src/exact.py, src/normalize.py    exact-match evidence, catalog-constrained LLM guard
src/buckets.py                    category-bucket evidence
src/explain.py, src/confidence.py customer-facing explanation, read-only confidence
src/llm.py                        fail-closed OpenRouter client, all flags default off
evaluator/local_evaluator.py      frozen customer simulator and scorer — do not edit
data/                             catalog, public sessions, eval/CV split manifests
scripts/                          eval, bootstrap, adversarial, CV, and demo tooling
tests/                            143 tests, including the frozen offline-score gate
docs/                             competition rules, ablation reports, integration guides
demo/, demo_server.py             interactive replay dashboard over the production Agent
```

## Limitations

- The exact-match evidence route, worth 0.155 MRR, depends on the simulator's fixed message templates. On paraphrased or real customer input it would rarely fire; the adversarial harness in `results/adversarial_robustness_report.md` quantifies exactly how much that costs.
- Candidate rotation (freshness) optimizes HitRate@10 by deliberately withholding a good match so it can resurface later. That is a metric-appropriate choice for this evaluator, not a production-appropriate one.
- The agent always returns 10 recommendations and never abstains. A read-only confidence layer exists as groundwork (`src/confidence.py`) but currently measures flat across every turn; see `docs/ablation_report.md` for the diagnosed mechanism.
- The public set is 200 sessions; final judging draws from 800 private sessions this repository has never seen.

## Data source

Catalog and sessions are derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry` category, sampled deterministically from the official Clothing 5-core leave-last-out split. See `DATA_ATTRIBUTION.md` before reusing or redistributing the data.

## Competition rules

This repository implements the required `Agent` interface (`reset`, `respond`) against TikTok TechJam Track 4's frozen evaluator. Full rules, the Agent API contract, and submission requirements: `docs/competition_specification.md`, `docs/agent_api_contract.json`, `docs/submission_rules.md`.
