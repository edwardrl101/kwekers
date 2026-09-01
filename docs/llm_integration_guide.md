# Day 4 Member 1 guide: optional OpenRouter integration

## Goal

Day 4 adds robustness and presentation features without changing the frozen
offline recommendation path. The current post-merge production configuration
scores **0.891111** over all 200 public sessions. The older **0.877011** value
in the original Day 4 brief is historical and predates the three-state exact
constraint correction now merged into `main`.

Member 1 owns the integration boundary: `starter/agent.py`, `src/llm.py`, the
offline freeze, feature flags, failure isolation, and final reproducibility.
The release is acceptable only when a clean environment with no API key makes
zero LLM calls and still scores exactly `0.891111`.

## Current architecture

```text
customer message
  |
  +-> deterministic regex parsing -------------------------------+
  |                                                              |
  +-> optional LLM parsing, only after a regex miss               |
  |       -> clean_constraint()                                   |
  |       -> ExactRoute catalog lookup                            |
  |       -> discard every candidate with no catalog match        |
  |                                                              v
  +-> accumulated SlotState -> bucket/exact/BM25 -> fusion -> top 10
  |                                                              |
  +-> optional confidence (read-only; never changes ranking) -----+
  |                                                              |
  +-> optional LLM customer message                               |
          -> deterministic explanation on any failure <-----------+
```

The shared `src.llm.call()` boundary is synchronous, cached, and fail-closed.
It accepts only a concrete model ID ending in `:free`; `openrouter/free`,
`openrouter/auto`, and paid model IDs are rejected before a network attempt.
It has a fixed three-second timeout, disables model reasoning for short tasks,
uses temperature zero, and returns `None` on every error. Prompts, responses,
and credentials are never written to telemetry.

## Local OpenRouter setup

1. Create a dedicated OpenRouter API key.
2. Copy `.env.example` to `.env` locally.
3. Put the key only in `.env`:

   ```dotenv
   OPENROUTER_API_KEY=
   OPENROUTER_MODEL=cohere/north-mini-code:free
   ENABLE_LLM_NORMALIZE=false
   ENABLE_LLM_OVERRIDE=false
   ENABLE_LLM_MESSAGE=false
   ENABLE_CONFIDENCE=false
   ```

4. Never commit `.env`. The repository ignores it; `.env.example` contains
   names and safe defaults only.
5. Run one smoke request:

   ```bash
   python scripts/llm_smoke.py
   ```

The verified direct-API model is `cohere/north-mini-code:free`. A smoke call on
2026-08-30 completed in 1.023 seconds with 11 prompt tokens, one completion
token, and reported cost `0.0`. `thinkingmachines/inkling-small:free` is not a
drop-in option for this project: OpenRouter returns HTTP 403 because its free
endpoint is restricted to recognized agentic harnesses.

Process environment variables override `.env`. This is useful in CI or an
authorized demo environment. The client reads `.env` only as a local fallback
and does not require `python-dotenv`.

## Feature flags

All flags default to `false` and can also be passed explicitly to `Agent()`.

| Flag | Effect when enabled | Deterministic fallback |
|---|---|---|
| `ENABLE_LLM_NORMALIZE` | Proposes constraints only after regex misses | Keep regex-derived state; discard invalid proposals |
| `ENABLE_LLM_OVERRIDE` | Classifies plausible paraphrased replacement language | Existing override regex and normal freshness behavior |
| `ENABLE_LLM_MESSAGE` | Writes a concise explanation from constraints and top-product metadata | Template explanation |
| `ENABLE_CONFIDENCE` | Computes normalized softmax confidence from fused candidates | `0.0`; recommendation order is unchanged |

Recommended development sequence:

```dotenv
# First validate confidence without networking.
ENABLE_CONFIDENCE=true

# Then test the two robustness features separately.
ENABLE_LLM_NORMALIZE=true
ENABLE_LLM_OVERRIDE=false
ENABLE_LLM_MESSAGE=false
```

Do not enable the LLM message flag for a full public evaluation: it deliberately
calls the model every turn and free endpoints have low daily request limits.
Regex-first normalization and override detection are designed to avoid calls on
known public evaluator templates.

## Failure and security behavior

- Missing key or model: return `None` without incrementing `CALL_COUNT`.
- Non-`:free` or OpenRouter router ID: reject before networking.
- Timeout, HTTP error, rate limit, malformed JSON, empty response, unexpected
  served model, or nonzero reported cost: record a safe outcome and return
  `None`.
- Successful identical request: return the process-local cached value without
  another API call.
- The Agent never sends the user profile. The explanation layer sends only
  active constraints, up to three product titles/prices, and confidence.
- Free endpoints may log prompts. Do not send personal or confidential data.

The submission must not contain a team API key. Official judging may disable
network access, so the scored path is intentionally offline. If organizers
want to exercise the LLM route, they must inject their own authorized
`OPENROUTER_API_KEY` into the process environment.

## Verification commands

Fast unit and integration checks:

```bash
python -m unittest tests.test_llm tests.test_normalize tests.test_explain tests.test_agent -v
```

Mandatory frozen offline check (about 45 seconds with the full catalog):

```bash
python -m unittest tests.test_frozen_baseline -v
```

Full suite:

```bash
python -m unittest discover -s tests -v
```

Before committing, confirm secret hygiene:

```bash
git check-ignore -v .env
git diff --check
git grep -n -i -e 'sk''-or-'
```

The expected final gate is:

```text
flags off + no key -> score 0.891111 -> CALL_COUNT 0
```

## Member 1 release checklist

- Keep `starter/agent.py` integration changes centralized.
- Require deterministic fallbacks from every optional contributor.
- Run the frozen test before every integration commit.
- Record actual serving model, p50/p95 latency, prompt/completion tokens, and
  reported cost for online measurements.
- Preserve the full per-scenario ablation evidence, including rejected work.
- Reproduce from a clean clone without `.env`.
- Verify the submission contains no secret and does not require network access.
- Tag `v1.0-final` only after the release commit and README are verified.
