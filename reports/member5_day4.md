# Member 5 (Sheng Yan) — Day 4 completion report

## Current baseline after rebasing to latest main

The earlier Member-5 result (`0.867675`, MRR `0.66225`) is obsolete. The latest
team measurements in the Day-4 instructions report the corrected frozen
offline baseline as:

| Metric | Current main |
|---|---:|
| TechnicalScore | **0.891111** |
| HitRate@10 | **1.000000** |
| MRR | **0.707704** |
| MTTC | **2.060000** |
| LLM calls on the public set | **0** |

The previous `0.877011` freeze mentioned earlier in the plan is also historical;
the current baseline changed after the three-state exact-constraint correction.

## Question-policy conclusion

Both policies were measured previously on the full evaluator:

- always `other`: `0.867675`
- `other`, `other`, then rotate `feature -> material -> color -> style`: `0.866142`
- absolute difference: `0.001533`
- team bootstrap noise floor: approximately `±0.019`

The difference is far inside the noise floor, so it is **not** evidence that one
policy is meaningfully better. We ship the simpler constant `other` policy and
retain the rotation policy as a tested fallback in `src/dialog.py` because
`other` relies on a simulator behavior that could be less robust to changed
message generation.

The dialog-state diagnostic still records a mean **2.11 question turns** to
drain the released four-item intent card with the constant policy.

## Day-4 Member-5 work

### `src/dialog.py`

- Keeps SlotState accumulation and override demotion behavior.
- Restores explicit named policies for constant-`other` and the measured
  rotation fallback.
- Defaults to constant `other`, matching the frozen production behavior.
- Enforces the safety invariant that a policy never emits null, `brand`, or
  `category`.
- Keeps public diagnostics for scenario-router accuracy and card-drain speed.
- Documents the policy comparison honestly as statistically equivalent.

### `src/explain.py`

- Deterministic-first explanation layer.
- Uses disclosed constraints in the message without inventing product facts.
- Confidence changes wording only; it never changes ranking or state.
- High confidence: states that the shortlist looks strong.
- Low confidence: says the system is still narrowing among several directions.
- Visible question stays open-ended so it is semantically aligned with
  `ask_attribute="other"`.
- Optional LLM rewrite is bounded and always falls back deterministically.

### `scripts/demo.py`

Interactive CLI for the screen recording:

```bash
python scripts/demo.py
```

Each turn shows:

- customer-facing explanation
- `ask_attribute`
- confidence
- ranked recommendation titles, prices, and ASINs

It can also reproduce the three required saved public-set transcripts:

```bash
python scripts/demo.py --save-examples
```

Generated files:

- `demo_transcripts/buying.md`
- `demo_transcripts/browsing.md`
- `demo_transcripts/intent_override.md`

The intent-override transcript deliberately continues through the evaluator's
blackout period and demonstrates the mind-change handling before hitting the
target.

## Ownership / integration safety

No changes are made to `starter/agent.py`. Day 4 assigns that file to Member 1,
and current main already contains the Member-5 integration: SlotState is updated
inside the turn loop, shown products are cleared on an override, accumulated
constraints are preserved, and the scored response asks `other`.
