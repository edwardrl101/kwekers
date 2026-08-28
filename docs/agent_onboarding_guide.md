# Onboarding Guide to `starter/agent.py`

## 1. What `Agent` is responsible for

[`starter/agent.py`](../starter/agent.py) is the only competition deliverable
the evaluator directly calls. It owns catalog loading, retrieval-route startup,
per-session state, candidate collection, failure isolation, final recommendation
formatting, and the clarification attribute returned to the customer simulator.

The current implementation is a **working route shell** rather than the final
hybrid search architecture. It calls bucket, exact, BM25, and dense adapters and
preserves each route's scores, but still concatenates IDs in route order instead
of using those scores for fusion.

The two required public methods are:

```python
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
```

## 2. Current completion status

### Requirement audit

| Requirement | Status | Evidence or gap |
|---|---|---|
| Crude working agent before optimization | Complete | Clean evaluator runs finish on tune and holdout |
| Return exactly 10 every turn | Complete | `RECOMMENDATION_COUNT = 10`, random fill, unit tests |
| Never return null `ask_attribute` | Complete | `respond()` always returns `"other"` |
| Always return a string `message` | Complete | Constant string in every response |
| Call bucket route | Complete | `BucketRoute` is initialized and queried |
| Call exact route | Complete | `ExactRoute` is initialized and queried |
| Call BM25 route | Complete | `BM25Route` is initialized and queried |
| Call dense route | Partial | Dependency and offline-first loading work; the machine-local 50k embedding cache still needs its one-time build |
| Integrate dialog teammate module | Not done | `SlotState` and `QuestionPolicy` exist but are not used by `Agent` |
| Retrieve 500 BM25 candidates | Complete | `ROUTE_CANDIDATE_LIMIT = 500` and the route-adapter test verifies it |
| Preserve route scores and identity | Complete | `_route_candidates()` returns a scored list for each named route |
| Narrow and rescore a wide BM25 pool | Not done | Scores are retained but not yet consumed; route lists are still concatenated |
| Isolate route failures | Complete | Initialization and every route call have separate exception handling |
| Evaluation script and CSV logging | Complete | `scripts/eval.py` and `runs.csv` exist |
| Fixed 140/60 split | Complete | `data/eval_split.json` is committed |
| Avoid equal-weight RRF | Complete | No RRF is implemented |
| BM25 as base ordering | Not done | Bucket candidates currently come before BM25 candidates |
| First integrated score reported | Newly measured | Clean in-memory measurement is documented below |

As a practical summary:

- The **reliability and measurement shell is mostly complete**.
- The **four retrieval adapters are connected**.
- The **500-candidate collection and score-preservation plumbing is complete**.
- The **actual promote/demote fusion and dialog-state design remains to be
  implemented**.

## 3. First integrated measurement

The temporary debug prints and unconditional first-session `break` have been
removed from `evaluator/local_evaluator.py`, so `scripts/eval.py` once again
runs complete splits. The following measurement predates the current
500-candidate/score-preservation change and should be treated as its comparison
baseline:

| Split | Samples | Technical score | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|---:|
| Tune | 140 | **0.658371** | 0.764286 | 0.432188 | 3.671429 |
| Holdout | 60 | **0.656248** | 0.783333 | 0.393049 | 3.666667 |

### Tune scenarios

| Scenario | N | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 56 | 0.821429 | 0.470855 | 3.000000 |
| Browsing | 56 | 0.875000 | 0.491461 | 2.428571 |
| Intent override | 21 | 0.285714 | 0.140476 | 8.904762 |
| Boundary | 7 | 0.857143 | 0.523810 | 3.285714 |

### Holdout scenarios

| Scenario | N | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 24 | 0.916667 | 0.533681 | 2.208333 |
| Browsing | 24 | 0.750000 | 0.253803 | 3.750000 |
| Intent override | 9 | 0.444444 | 0.370370 | 7.888889 |
| Boundary | 3 | 1.000000 | 0.450000 | 2.000000 |

These numbers show that the modules already contain useful retrieval signal.
Intent override is the clearest weakness, which is consistent with dialog state
being implemented in `src/dialog.py` but disconnected from `Agent`.

## 4. Architecture at a glance

### Current architecture

```mermaid
flowchart TD
    Catalog[Catalog JSONL] --> Load[Load catalog dictionary]
    Load --> BucketInit[Initialize BucketRoute]
    Load --> ExactInit[Initialize ExactRoute]
    Load --> BM25Init[Initialize BM25Route]
    Load --> DenseInit[Try to initialize DenseRoute]

    Message[Current user message] --> BucketQ[Bucket query limit 500]
    Message --> ExactQ[Exact query limit 500]
    Message --> BM25Q[BM25 query limit 500]
    Message --> DenseQ[Dense query limit 500]

    BucketInit --> BucketQ
    ExactInit --> ExactQ
    BM25Init --> BM25Q
    DenseInit --> DenseQ

    BucketQ --> Merge[Concatenate and deduplicate]
    ExactQ --> Merge
    BM25Q --> Merge
    DenseQ --> Merge
    Merge --> FirstTen[Keep first 10 valid IDs]
    FirstTen --> Fill[Random fill if fewer than 10]
    Fill --> Response[Return message, other, and 10 recommendations]
```

### Important consequence

The merge order is:

```python
bucket -> exact -> BM25 -> dense
```

If the bucket route returns 500 IDs, the first ten bucket IDs become the final
recommendations. Exact, BM25, and dense still run, but their candidates cannot
reach the output on that turn.

This is not equal-weight RRF, but it is also not the requested BM25-base fusion.
It is route-priority concatenation.

## 5. Constants

```python
RECOMMENDATION_COUNT = 10
ROUTE_CANDIDATE_LIMIT = 500
RANDOM_FILL_SEED = "kwekers-day1-random-fill-v1"
```

### `RECOMMENDATION_COUNT`

Hard-codes the output length required by the Day 1 strategy. The evaluator
scores at most ten valid unique recommendations, so returning fewer gives away
free guesses.

### `ROUTE_CANDIDATE_LIMIT`

Every route currently receives `limit=500`. This satisfies the wide-retrieval
requirement. A later tuning pass may use separate constants such as:

```python
BM25_CANDIDATE_LIMIT = 500
SUPPORTING_ROUTE_LIMIT = 500
```

This makes it possible to tune route sizes independently.

### `RANDOM_FILL_SEED`

Makes fallback recommendations reproducible. It is not intended to improve
retrieval quality; it only ensures a complete response when routes return fewer
than ten usable products.

## 6. `Agent.__init__()`

```python
agent = Agent("data/catalog.jsonl")
```

Initialization performs four steps:

1. Normalize the catalog path.
2. Load the full catalog into memory.
3. Create catalog ID collections and an empty session dictionary.
4. Initialize each retrieval route independently.

```python
self._catalog = self._load_catalog()
self._catalog_ids = list(self._catalog)
self._catalog_id_set = set(self._catalog_ids)
self._sessions = {}
self._initialize_routes()
```

The dictionary preserves catalog order in modern Python. That order later
affects bucket ordering and deterministic fallback behavior.

### Current runtime availability

On the present machine:

```text
BucketRoute = available
ExactRoute  = available
BM25Route   = available
DenseRoute  = model/dependency available; full-catalog cache pending
```

The dense route passes a small smoke test and now prefers locally cached model
files without network metadata checks. Its one-time 50k CPU cache build is not
complete on this machine; the measured throughput makes that a separate,
long-running preparation step. Dense remains failure-isolated because the
dependency, model, or cache may not exist in an offline judging environment.

## 7. `_load_catalog()`

This helper reads the catalog into:

```python
dict[parent_asin, product_dictionary]
```

### Example input JSONL

```json
{"parent_asin":"A1","title":"Blue Cotton Shirt"}
{"parent_asin":"A2","title":"Black Leather Boot"}
```

### Result

```python
{
    "A1": {"parent_asin": "A1", "title": "Blue Cotton Shirt"},
    "A2": {"parent_asin": "A2", "title": "Black Leather Boot"},
}
```

Malformed JSON rows, non-dictionaries, and missing IDs are skipped. Duplicate
IDs keep their first product. A missing or unreadable file produces `{}` rather
than crashing agent construction.

This is deliberately defensive, although silently skipping malformed catalog
data can make diagnostics harder. Production-quality logging could report the
number of skipped rows without breaking the response contract.

## 8. `_initialize_routes()`

This function imports and constructs every retrieval route independently:

```python
try:
    from src.buckets import BucketRoute
    self._bucket_route = BucketRoute(self._catalog)
except Exception:
    self._bucket_route = None
```

The same structure is repeated for exact, BM25, and dense.

### Why import inside the function?

Optional dependencies are isolated. Importing `starter.agent` does not fail
merely because dense-model dependencies are absent.

### Why one `try` per route?

If DenseRoute fails, BM25 and exact search remain usable. A single broad `try`
around every constructor would incorrectly disable all later routes after one
failure.

### Limitation

All exceptions are silent. This protects evaluation, but developers cannot see
which route failed without inspecting private attributes. A later improvement
could store health information:

```python
self._route_health = {
    "dense": "SentenceTransformer dependency unavailable",
}
```

## 9. `reset()`

The evaluator calls:

```python
agent.reset(session_id, user_profile)
```

Current state stored per session:

```python
{
    "seed_key": "sha256-of-profile",
    "user_profile": safe_profile,
}
```

### Example

```python
agent.reset(
    "session-1",
    {"summary": "Prefers durable, comfortable products"},
)
```

Conceptual result:

```python
agent._sessions["session-1"] = {
    "seed_key": "<stable SHA-256 hex string>",
    "user_profile": {
        "summary": "Prefers durable, comfortable products",
    },
}
```

The profile hash ensures random fill is stable even though evaluator session
IDs contain random UUIDs.

### Major missing integration

`src.dialog.SlotState` is not created here. Consequently, the agent does not
currently retain:

- Accumulated constraints from earlier user turns
- Asked attributes
- Boundary replies
- Intent exhaustion
- Override history
- Active versus demoted preferences

The final version should store a `SlotState` and perhaps a `QuestionPolicy` in
each session.

## 10. `_query_route()`

All teammate retrieval routes follow this interface:

```python
route.query(text, limit) -> list[tuple[parent_asin, score]]
```

`_query_route()` adapts that into an ID-only list.

### Example route output

```python
[
    ("A1", 12.7),
    ("A2", 9.4),
]
```

### Current adapter output

```python
[
    ("A1", 12.7),
    ("A2", 9.4),
]
```

It validates the outer list, ASIN, and finite numeric score before preserving
the pair:

```python
candidates.append((parent_asin, float(score)))
```

### Why preserving scores matters

The next fusion layer can now distinguish:

```text
Exact match with score 1.0
Weak dense similarity with score 0.21
Strong BM25 result with score 3020
```

`_route_candidates()` also preserves the route name. The remaining work is to
normalize and apply these incomparable score scales in a BM25-base fusion.

## 11. Route adapters

These four functions currently have the same shape:

```python
def _route_bm25(self, session, user_message, turn):
    return self._query_route(self._bm25_route, user_message)
```

The available arguments anticipate future behavior:

- `session` can provide accumulated dialog state.
- `user_message` is the newest customer message.
- `turn` can support turn-aware strategy.

At present, only `user_message` is used.

### Bucket route

`BucketRoute` extracts a coarse category from evaluator opening-message
templates and returns catalog-ordered products in that category.

Useful for candidate filtering, but its order is not relevance ranking. It
should normally constrain or boost BM25 candidates rather than become the base
Top 10.

### Exact route

`ExactRoute` indexes cleaned feature strings and `key: value` detail pairs. It
can be extremely strong because the simulator derives customer constraints
from the same product metadata.

Its extraction currently captures the whole matters clause. A message with two
semicolon-separated constraints may not match separately, which is a future
improvement opportunity.

### BM25 route

`BM25Route` uses SQLite FTS5 with weighted fields and a strictness cascade:

```text
exact phrases -> all terms -> any term
```

This is the best current candidate for the requested base ordering.

### Dense route

`DenseRoute` uses `BAAI/bge-small-en-v1.5` embeddings. Its cached 50,000 by 384
float16 matrix is about 38 MB on disk. It is feasible in memory, but requires
model dependencies and a prepared cache.

```bash
python -m pip install -r requirements-dense.txt
python scripts/build_dense_cache.py --device cpu
```

The builder validates row counts and runs a smoke query after writing
`data/dense_cache.npz`. That generated machine-local artifact is intentionally
gitignored. On this CPU, the original full-document encoding was projected to
take several hours, so run the one-time build on faster CPU or GPU hardware when
possible.

### Existing but unused N-gram route

`src.retrieval.NgramRoute` provides character 3-to-5-gram TF-IDF for fuzzy and
truncated constraints. `Agent` does not initialize or call it. If this module
belongs to the agreed teammate surface, it remains an integration gap.

## 12. `_route_candidates()`

This function calls every named adapter inside an independent `try` block:

```python
for name, route in routes:
    try:
        candidates = route(session, user_message, turn)
    except Exception:
        candidates = []
    route_results[name] = candidates
```

Its output preserves route identity, ranking, and scores. The separate
`_concatenate_route_ids()` compatibility step still concatenates and
deduplicates IDs while retaining first occurrence; this is the outstanding
fusion gap.

### Example

Suppose routes return:

```python
bucket = [("A", 1.0), ("B", 0.8), ("C", 0.7)]
exact  = [("C", 1.0), ("TARGET", 1.0)]
bm25   = [("TARGET", 3020.0), ("D", 3010.0), ("A", 3001.0)]
dense  = [("E", 0.82), ("TARGET", 0.80)]
```

Current merged output:

```python
["A", "B", "C", "TARGET", "D", "E"]
```

`TARGET` keeps its first position from the exact route. Scores are available in
`route_results`, but the compatibility concatenation does not use them yet.

### Route failure example

If exact raises an exception:

```python
bucket = [("A", 1.0), ("B", 0.8), ("C", 0.7)]
exact  = exception
bm25   = [("TARGET", 3020.0), ("D", 3010.0)]
```

The merged output remains:

```python
["A", "B", "C", "TARGET", "D"]
```

This satisfies the requirement that one teammate's crash must not zero the
whole evaluation turn.

## 13. `_random_fill()`

Despite its name, this function has two responsibilities:

1. Validate and truncate routed candidates to ten.
2. Randomly fill any missing output positions.

It first keeps catalog-valid unique route candidates:

```python
for parent_asin in candidates:
    if parent_asin in catalog and parent_asin not in seen:
        result.append(parent_asin)
    if len(result) == 10:
        return result
```

If only seven survive, it selects three deterministic random catalog IDs. If
the catalog is unavailable, unique placeholder IDs preserve the response
schema and exact list length, although the evaluator will discard them as
invalid.

### Example

```python
candidates = ["A", "A", "MISSING", "B"]
```

With `A` and `B` in the catalog, the result begins:

```python
["A", "B", ...eight deterministic random valid IDs...]
```

This helper is the final safety net enforcing the ten-recommendation rule.

## 14. `respond()`

`respond()` is the public turn entry point.

```mermaid
sequenceDiagram
    participant E as Evaluator
    participant A as Agent.respond
    participant R as Retrieval routes
    E->>A: session ID, user message, turn, top K
    A->>A: Recover session or safe fallback state
    A->>R: Query bucket, exact, BM25, and dense
    R-->>A: Candidate ID lists or isolated failures
    A->>A: Concatenate, deduplicate, validate, and fill to 10
    A-->>E: String message, ask other, and 10 recommendations
```

### Example input

```python
response = agent.respond(
    session_id="session-1",
    user_message=(
        "I'm looking for Jewelry Necklaces. "
        "A key requirement is: Material:alloy."
    ),
    turn=1,
    top_k=10,
)
```

### Output shape

```python
{
    "message": "I am refining the shortlist. What else should I consider?",
    "ask_attribute": "other",
    "recommendations": [
        {"parent_asin": "A1"},
        # exactly nine more
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    },
}
```

The entire body has a final exception fallback, while route-level isolation
already handles expected teammate failures.

The supplied `top_k` argument is currently ignored in favor of the hard-coded
ten-item competition rule.

## 15. Teammate modules and their integration state

| Module | Main runtime pieces | Current Agent use |
|---|---|---|
| `src/buckets.py` | `BucketRoute` | Initialized and queried |
| `src/exact.py` | `ExactRoute` | Initialized and queried |
| `src/retrieval.py` | `BM25Route` | Initialized and queried |
| `src/retrieval.py` | `DenseRoute` | Connected; local model works, 50k cache build pending |
| `src/retrieval.py` | `NgramRoute` | Not used |
| `src/dialog.py` | `SlotState` | Not used |
| `src/dialog.py` | `ScenarioRouter` | Not used directly |
| `src/dialog.py` | `QuestionPolicy` | Not used |

### What the disconnected dialog module already provides

`SlotState.update()` can:

- Parse key requirements and multi-constraint replies
- Accumulate constraints across turns
- Detect boundary replies
- Detect exhausted `other` responses
- Recognize intent overrides
- Demote the overridden preference
- Produce a compact structured context

`QuestionPolicy.next_attribute()` can:

- Ask `other` while the public intent card is still being drained
- Avoid null attributes
- Use candidate information-gain scores later
- Fall back through useful specific attributes

Connecting these classes is the most direct path to improving intent-override
performance.

## 16. What the end goal should look like

The final system should not treat route outputs as equal lists. It should use
BM25 to define a broad candidate pool, then let other signals move products
within that pool.

```mermaid
flowchart TD
    User[New user message] --> State[Update SlotState]
    Profile[User profile] --> Query[Build compact cumulative query]
    State --> Query

    Query --> BM25[BM25 retrieve 500]
    Query --> Exact[Exact matching signals]
    Query --> Bucket[Category bucket membership]
    Query --> Dense[Dense semantic scores]

    BM25 --> Pool[BM25 base pool and ordering]
    Exact --> Adjust[Promote exact matches]
    Bucket --> Adjust
    Dense --> Adjust
    Pool --> Adjust[Promote or demote within base pool]
    Adjust --> Rank[Final ranked Top 10]

    State --> Policy[QuestionPolicy]
    Pool --> Policy
    Policy --> Ask[Non-null ask attribute]
    Rank --> Response[Agent response]
    Ask --> Response
```

### Suggested non-RRF fusion shape

Preserve route scores instead of converting immediately to IDs:

```python
bm25_results = bm25.query(query, limit=500)
exact_ids = {asin for asin, _ in exact.query(query, limit=500)}
bucket_ids = set(bucket.bucket_for_message(opening_message))
dense_scores = dict(dense.query(query, limit=500))
```

Initialize candidates from BM25 rank:

```python
for bm25_rank, (asin, bm25_score) in enumerate(bm25_results, start=1):
    fused[asin] = {
        "base_rank": bm25_rank,
        "score": normalized_bm25_score,
    }
```

Then apply interpretable adjustments:

```python
if asin in exact_ids:
    score += exact_boost
if bucket_is_known and asin not in bucket_ids:
    score -= category_penalty
score += dense_weight * normalized_dense_score
```

This obeys the instruction to start with BM25 ordering and let teammate signals
promote or demote. It is not equal-weight reciprocal-rank fusion.

## 17. Dialog-state integration target

`reset()` should eventually create state resembling:

```python
from src.dialog import QuestionPolicy, SlotState

self._sessions[session_id] = {
    "user_profile": user_profile,
    "slot_state": SlotState(session_id=session_id),
    "question_policy": QuestionPolicy(),
}
```

Every `respond()` call should:

```python
state.update(user_message, turn)
query = build_query(state.to_context(), user_profile)
candidates = retrieve_and_fuse(query)
ask_attribute = policy.next_attribute(state, information_gain)
state.record_ask(ask_attribute)
```

This solves a major current defect: routes now see only the newest message.
After the customer reveals a new constraint, earlier category and constraint
information can disappear from the query. A cumulative structured query keeps
active constraints while correctly demoting overridden ones.

## 18. Why intent override is currently weak

The tune intent-override metrics are:

```text
Hit@10 = 0.285714
MRR    = 0.140476
MTTC   = 8.904762
```

Likely contributing mechanisms are visible in the code:

1. No `SlotState` is connected.
2. Routes query only the current message.
3. Old and new preferences are not explicitly distinguished by `Agent`.
4. Target hits before the override are intentionally ineligible.
5. Preserved route scores are not yet used for ranking.
6. Route concatenation can hide strong BM25 or exact results behind bucket IDs.

The dialog teammate code was designed specifically to address accumulation and
override behavior, so integration should precede more exotic fusion work.

## 19. Evaluation workflow

The temporary debug statements and unconditional one-session `break` have been
removed from `evaluator/local_evaluator.py`. The standard commands are:

```bash
python scripts/eval.py --split tune --label your-change-name
python scripts/eval.py --split holdout --label your-change-name
```

The script:

1. Loads the committed split manifest.
2. Runs the evaluator.
3. Prints overall and per-scenario metrics.
4. Writes a detailed ignored result JSON.
5. Appends one timestamped row to `runs.csv`.

Use tune for ordinary iteration. Run holdout only at meaningful checkpoints so
the team does not gradually optimize against it.

## 20. Tests and safety guarantees

The current repository has 21 passing tests. Agent-specific tests verify:

- Exactly ten unique recommendations
- A non-null `"other"` question
- Missing-catalog survival
- Calling `respond()` without `reset()` does not crash
- One broken route does not break the response
- Stable random fill across evaluator UUIDs
- Every route adapter delegates with the 500-candidate limit
- Route identity and finite scores are preserved

Tests establish the shell contract, but do not yet verify:

- BM25 base ordering
- Score-preserving fusion
- Dialog-state accumulation
- Override demotion through the full `Agent`
- Dense behavior when dependencies are available

## 21. Recommended next implementation order

### Midday correctness pass

1. Build the machine-local dense cache on suitable hardware.
2. Make BM25 the base pool and apply simple exact, bucket, and dense boosts.
3. Connect `SlotState` and `QuestionPolicy` in `reset()` and `respond()`.
4. Run and log tune plus holdout with a new label.

### Later improvement pass

1. Split multi-constraint exact queries on semicolons.
2. Add the N-gram route if agreed by the team.
3. Compute candidate-pool attribute information gain.
4. Tune signal weights on tune only.
5. Benchmark and validate dense recall after the cache is built.
6. Add route latency and health diagnostics.

## 22. Key takeaways for a new contributor

1. `Agent` already produces valid responses and calls four route adapters.
2. Ten recommendations, a string message, and non-null asking are guaranteed.
3. The current high score comes from useful teammate route implementations,
   especially exact and BM25 retrieval.
4. Route concatenation, unused scores, and disconnected dialog state are the
   largest remaining gaps relative to the stated architecture.
5. `src/dialog.py` is not speculative work; it directly addresses the weakest
   measured scenario.
6. The immediate end goal is a reliable cumulative-query agent with a 500-item
   BM25 base pool and simple interpretable promotion and demotion signals.
