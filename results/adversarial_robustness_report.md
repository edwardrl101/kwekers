# Adversarial robustness report

## Frozen baseline verification

Level 0 reproduced TechnicalScore **0.891111** on 200 sessions. The shipped response reported 0 tokens.
The assignment's `0.877011` gate is historical; current main's frozen offline test is `0.891111` after the three-state exact-constraint correction.

## Adversarial levels

| Level | Score | Hit@10 | MRR | MTTC | Abs. delta | Rel. delta | Improved | Unchanged | Worsened | Disappeared |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.891111 | 1.000000 | 0.707704 | 2.060000 | +0.000000 | +0.000% | 0 | 200 | 0 | 0 |
| 1 | 0.816230 | 0.915000 | 0.695099 | 3.490000 | -0.074881 | -8.403% | 53 | 55 | 92 | 17 |
| 2 | 0.508444 | 0.575000 | 0.399812 | 5.950000 | -0.382667 | -42.943% | 35 | 37 | 128 | 85 |
| 3 | 0.460816 | 0.520000 | 0.362054 | 6.390000 | -0.430295 | -48.287% | 30 | 34 | 136 | 96 |
| 4 | 0.419598 | 0.465000 | 0.352325 | 6.930000 | -0.471513 | -52.913% | 27 | 30 | 143 | 107 |

## Per-scenario robustness

| Level | Scenario | N | Hit@10 | MRR | MTTC |
|---:|---|---:|---:|---:|---:|
| 0 | buying | 80 | 1.000000 | 0.604509 | 1.325000 |
| 0 | browsing | 80 | 1.000000 | 0.764876 | 2.087500 |
| 0 | intent_override | 30 | 1.000000 | 0.848743 | 3.666667 |
| 0 | boundary | 10 | 1.000000 | 0.652778 | 2.900000 |
| 1 | buying | 80 | 0.925000 | 0.674816 | 3.037500 |
| 1 | browsing | 80 | 0.950000 | 0.732307 | 3.037500 |
| 1 | intent_override | 30 | 0.766667 | 0.576111 | 5.900000 |
| 1 | boundary | 10 | 1.000000 | 0.916667 | 3.500000 |
| 2 | buying | 80 | 0.600000 | 0.377966 | 5.425000 |
| 2 | browsing | 80 | 0.625000 | 0.423646 | 5.487500 |
| 2 | intent_override | 30 | 0.266667 | 0.244444 | 9.133333 |
| 2 | boundary | 10 | 0.900000 | 0.850000 | 4.300000 |
| 3 | buying | 80 | 0.550000 | 0.365258 | 5.900000 |
| 3 | browsing | 80 | 0.625000 | 0.421126 | 5.500000 |
| 3 | intent_override | 30 | 0.033333 | 0.033333 | 10.766667 |
| 3 | boundary | 10 | 0.900000 | 0.850000 | 4.300000 |
| 4 | buying | 80 | 0.487500 | 0.362515 | 6.500000 |
| 4 | browsing | 80 | 0.550000 | 0.418299 | 6.250000 |
| 4 | intent_override | 30 | 0.033333 | 0.033333 | 10.766667 |
| 4 | boundary | 10 | 0.900000 | 0.700000 | 4.300000 |

## Representative before/after examples

### Level 1 — public_0001 turn 1

- Before: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
- After: Please help me find Jewelry Necklaces. A key requirement is: Material:alloy!
- Constraint changed: NO

### Level 1 — public_0002 turn 1

- Before: I'm looking for Accessories Belts. Buckle closure
- After: Could you show me Accessories Belts. Buckle closure
- Constraint changed: NO

### Level 1 — public_0002 turn 2

- Before: For that, what matters is: leather; 100% Leather.
- After: For that, what matters is: leather; 100% Leather...
- Constraint changed: NO

### Level 1 — public_0002 turn 3

- Before: Actually, ignore my earlier preference. What I need is: leather.
- After: Actually, ignore my earlier preference. What I need is: leather!
- Constraint changed: NO

### Level 1 — public_0003 turn 1

- Before: I'm looking for Watches Wrist Watches. Stainless Steel Band
- After: Could you show me Watches Wrist Watches. Stainless Steel Band
- Constraint changed: NO

### Level 2 — public_0001 turn 1

- Before: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
- After: Please help me find Jewelry Necklaces. The important part is: Material:alloy!
- Constraint changed: NO

### Level 2 — public_0001 turn 2

- Before: For that, what matters is: Triple Moon Pentagram Symbol; The Triple Moon represents the Phases of the Moon which are linked to the three aspects of the Goddess and the phases of the Life of Women.The Pentagram representing the holistic r.
- After: The important part is: Triple Moon Pentagram Symbol; The Triple Moon represents the Phases of the Moon which are linked to the three aspects of the Goddess and the phases of the Life of Women.The Pentagram representing the holistic r...
- Constraint changed: NO

### Level 2 — public_0002 turn 1

- Before: I'm looking for Accessories Belts. Buckle closure
- After: Please help me find Accessories Belts. Buckle closure
- Constraint changed: NO

### Level 2 — public_0002 turn 2

- Before: For that, what matters is: leather; 100% Leather.
- After: My main requirement is: leather; 100% Leather...
- Constraint changed: NO

### Level 2 — public_0002 turn 3

- Before: Actually, ignore my earlier preference. What I need is: leather.
- After: Actually, ignore my earlier preference. The important part is: leather...
- Constraint changed: NO

### Level 3 — public_0001 turn 1

- Before: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
- After: Please help me find Jewelry Necklaces. What would suit me is Material:alloy.
- Constraint changed: NO

### Level 3 — public_0001 turn 2

- Before: For that, what matters is: Triple Moon Pentagram Symbol; The Triple Moon represents the Phases of the Moon which are linked to the three aspects of the Goddess and the phases of the Life of Women.The Pentagram representing the holistic r.
- After: What would suit me is Triple Moon Pentagram Symbol; The Triple Moon represents the Phases of the Moon which are linked to the three aspects of the Goddess and the phases of the Life of Women.The Pentagram representing the holistic r...
- Constraint changed: NO

### Level 3 — public_0002 turn 1

- Before: I'm looking for Accessories Belts. Buckle closure
- After: I'm hoping to get Accessories Belts. Buckle closure
- Constraint changed: NO

### Level 3 — public_0002 turn 2

- Before: For that, what matters is: leather; 100% Leather.
- After: What would suit me is leather; 100% Leather.
- Constraint changed: NO

### Level 3 — public_0002 turn 3

- Before: Actually, ignore my earlier preference. What I need is: leather.
- After: Please replace what I said before; What would suit me is leather.
- Constraint changed: NO

### Level 4 — public_0002 turn 2

- Before: For that, what matters is: leather; 100% Leather.
- After: What would suit me is made from genuine hide; made entirely from genuine hide!
- Constraint changed: YES

### Level 4 — public_0002 turn 3

- Before: Actually, ignore my earlier preference. What I need is: leather.
- After: Please replace what I said before; What would suit me is made from genuine hide!
- Constraint changed: YES

### Level 4 — public_0002 turn 4

- Before: For that, what matters is: Imported; Buckle closure.
- After: What would suit me is produced outside the domestic market; Buckle closure...
- Constraint changed: YES

### Level 4 — public_0003 turn 2

- Before: For that, what matters is: Water Resistant; 3 Year Battery.
- After: What would suit me is able to handle light rain; 3 Year Battery!
- Constraint changed: YES

### Level 4 — public_0003 turn 3

- Before: Actually, ignore my earlier preference. What I need is: Water Resistant.
- After: Please replace what I said before; What would suit me is able to handle light rain...
- Constraint changed: YES

## Level 4 failure examples

### public_0002

- Original message: For that, what matters is: leather; 100% Leather.
- Perturbed message: What would suit me is made from genuine hide; made entirely from genuine hide!
- Original target rank: 1
- Perturbed target rank: None
- Likely cause: constraint wording changed, weakening exact-match and BM25 lexical overlap

### public_0003

- Original message: For that, what matters is: Water Resistant; 3 Year Battery.
- Perturbed message: What would suit me is able to handle light rain; 3 Year Battery!
- Original target rank: 1
- Perturbed target rank: 1
- Likely cause: constraint wording changed, weakening exact-match and BM25 lexical overlap

### public_0004

- Original message: Actually, ignore my earlier preference. What I need is: polyester.
- Perturbed message: Please replace what I said before; What would suit me is a durable synthetic fibre!
- Original target rank: 1
- Perturbed target rank: None
- Likely cause: override phrase changed, so the agent's freshness reset may not be recognized

### public_0005

- Original message: I'm looking for Outdoor & Work Snow & Cold Weather. A key requirement is: leather.
- Perturbed message: Please help me find Outdoor & Work Snow & Cold Weather. What would suit me is made from genuine hide!
- Original target rank: 2
- Perturbed target rank: 4
- Likely cause: constraint wording changed, weakening exact-match and BM25 lexical overlap

### public_0007

- Original message: I'm looking for Tees & Blouses Tunics, but I'm still exploring.
- Perturbed message: Could you show me Tees & Blouses Tunics, but I'm still exploring!
- Original target rank: 1
- Perturbed target rank: 10
- Likely cause: surface/template wording changed, likely affecting category or constraint parsing

## Latency

Only `Agent.respond()` is included in per-turn latency; Agent/catalog construction is excluded. Total wall time covers the evaluator loop.

| Level | Turns | Mean ms | p50 ms | p95 ms | Wall seconds |
|---:|---:|---:|---:|---:|---:|
| 0 | 412 | 35.024 | 32.790 | 74.007 | 14.505 |
| 1 | 681 | 35.784 | 35.335 | 63.609 | 24.472 |
| 2 | 1105 | 21.396 | 17.658 | 42.051 | 23.733 |
| 3 | 1182 | 21.753 | 17.690 | 42.502 | 25.859 |
| 4 | 1279 | 22.005 | 18.645 | 44.227 | 28.301 |

## LLM on/off cost

| Mode | Available | Score | Model | Calls | Input tokens | Output tokens | p50 ms | p95 ms | Cost/session | Cost/1M sessions |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llm_off | yes | 0.891111 | none | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| llm_on_level_4 | yes | 0.423564 | cohere/north-mini-code:free | 484 | 3410 | 295 | 64.03016650438076 | 444.6235587973202 | not computed | not computed |

LLM-off is always measured with explicit false Agent flags and a reset telemetry state. LLM-on is only run when `--measure-llm-on` is supplied; this prevents an ordinary robustness run from making network calls. Token-price projections remain `not computed` unless explicit prices are supplied.

## Previous rejected experiments

- Day 2 category filtering: browsing R@10 remained 0.0375 before and after. Rejected.
- Day 3 near-duplicate suppression: only 7/200 sessions had a pair at 0.80; best score delta was about +0.00005. Rejected / do not ship.

## Limitations

- Levels 1–4 are deterministic synthetic perturbations, not an LLM paraphrase distribution.
- Level 4 uses a bounded semantic rule set; unrecognised long constraints retain their literal body inside a natural-language wrapper.
- Session comparison uses the evaluator's first successful target rank, not every per-turn rank after conversion.
