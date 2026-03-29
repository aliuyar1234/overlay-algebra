# CLAIMS_MATRIX.md

Status values:
- **intended**: claim we hope to validate
- **partially supported**: some evidence exists, but not enough for full wording
- **supported**: required evidence exists
- **unsupported**: evidence failed or is absent
- **weakened**: must be stated in reduced form

Experiments are defined in `docs/EVALUATION_PLAN.md`.

| Claim ID | Exact claim text | Type | Status | Required evidence | Experiments / analyses needed | Implementation dependencies | Risk of overclaiming | If evidence is insufficient, weaken to... |
|---|---|---|---|---|---|---|---|---|
| C1 | In this semantics-fixed QA setting, at least two single-overlay adapters can be decomposed into a shared semantic scaffold plus an operational residual such that `S + R_k` recovers most of the target overlay behavior with only small semantic degradation. | empirical / methods | supported | single-overlay recovery table; semantic-retention metrics; direct comparison to `F_k` and prompt-only | E1, E1b | R1-R10, R12-R13 | high | "Some overlay-specific signal survives residualization, but recovery is weak or overlay-dependent." |
| C2 | For at least one unseen overlay pair, SOAR composition (`S + R_i + R_j`) outperforms both prompt-only control and tuned whole-adapter merge on the active overlay metrics under a semantic-retention guardrail. | empirical / ablation | intended | pair-composition table; baseline table; tuning logs; semantic guardrail report | E2 | R1-R13 | high | "SOAR exposes composition trade-offs but does not clearly beat tuned merging." |
| C3 | JSON is more linearly composable and causes less collateral damage than citation and quote in this setting. | empirical / analysis | intended | compatibility heatmap; collateral-damage analysis; pair/triple results | E2, E3 | R1-R13 | medium-high | "Composability differs across overlays, but no stable JSON > citation/quote ordering was established." |
| C4 | Dense-delta residualization is the correct technical object for the method because raw LoRA factor arithmetic is representation-dependent; using dense deltas avoids this ambiguity. | theoretical / methods | partially supported | factor non-uniqueness argument; gauge-invariance toy test; optional empirical ablation vs factor subtraction | P1, E4 | R8-R9 | medium | "Dense-delta residualization is a principled design choice; empirical superiority over factor arithmetic was not established." |
| C5 | The evaluation pipeline deterministically measures semantic retention, JSON validity/schema compliance, citation accuracy, and quote accuracy on a leak-controlled dataset. | systems / reproducibility | supported | parser golden tests; round-trip compiler tests; canonical JSON + embedded-quote goldens; split audit; metric unit tests | E0, E5 | R2-R4, R11-R13 | low | "The core evaluation is mostly deterministic, but some measurements still need manual inspection." |
| C6 | Direct pair/triple training remains a useful upper bound; SOAR aims for partial composability, not full equivalence to monolithic multi-behavior training. | empirical / framing | intended | upper-bound comparison tables; composition gap analysis | E2, E6 | R7-R13 | low | "SOAR occasionally matches the upper bound, but we do not claim it in general." |
| C7 | The project contributes a semantics-fixed experimental protocol for studying operational overlay composition that is meaningfully distinct from generic adapter merging. | methods / positioning | intended | explicit protocol, baselines, compiler spec, novelty boundary in related work map | design review + final paper framing | R1-R13, related work review | medium | "This is a controlled adapter-composition protocol within one narrow QA substrate." |
| C8 | The strongest SOAR overlays show at least limited transfer to a small real-document slice beyond the SQuAD-derived core dataset. | empirical / external validity | intended (late-stage) | transfer-slice metrics and examples | E7 | R14 plus core method | high | "Operational relevance beyond the core syntheticized QA setup remains unvalidated." |

## Claim gating notes

### Strong support thresholds (default; may be revised only with explicit doc updates)
- **C1 strong support:** for at least 2 overlays, `primary_overlay_score(S+R_k) >= 0.80 * primary_overlay_score(F_k)` and `semantic_F1(S+R_k) >= semantic_F1(S) - 3.0`.
- **C2 strong support:** at least 1 unseen pair where SOAR beats **both** baselines by >= 3 absolute points on mean active overlay primary score while `semantic_F1 >= semantic_F1(S) - 3.0`.
- **C3 strong support:** JSON has the smallest median composition gap and lowest median collateral-damage score across pair/triple conditions.
- **C4 strong support:** factor-gauge toy test passes and dense-delta method outperforms or is clearly more stable/interpretable than factor subtraction in E4.
- **C5 strong support:** all compiler/parser/metric golden tests pass, including canonical JSON and embedded-quote cases; split audit shows no title leakage.
- **C8 strong support:** at least one composed pair retains its ranking advantage over prompt-only on the transfer slice.

### Important honesty rule
Do not promote C1/C2/C3/C8 to supported with cherry-picked examples only.
Tables and saved metrics must exist in the repo artifacts.

## Claim-to-evidence bookkeeping
Every milestone evidence note should explicitly say:
- which C-IDs it affects,
- whether support increased, stayed flat, or weakened,
- what remains unresolved.
