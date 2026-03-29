# M4 JQ Writing Note

This note is a paper-facing summary for the corrected reduced-scope `M4` bundle.
Use it together with the copied source reports in this bundle, not as a replacement for them.

## Recommended framing

Recommended main result:
- In the semantics-fixed QA setting, `SOAR` cleanly recovers two operational overlays at full scale: `J` and `Q`.
- The strongest safe claim is a single-overlay residualization claim, not a composition claim.
- The current bundle supports `C1`, keeps `C4` partially supported, and supports `C5`.

Recommended one-line paper framing:
- Dense-delta residualization can isolate reusable operational behavior for at least two overlays in a semantics-fixed frozen-LM QA setting, but broader composition claims remain deferred.

## Core Table

Primary source files:
- `full_single_overlay/metrics.json`
- `single_overlay_recovery_j/metrics.json`
- `single_overlay_recovery_q/metrics.json`
- `claim_checks_c1_c4_c5/metrics.json`

| Overlay | Primary Metric | Prompt-only Primary | Direct Primary | SOAR Primary | Direct Answer F1 | SOAR Answer F1 | Recovery Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| J | `json_strict_schema_valid` | 0.0000 | 0.9986 | 0.9986 | 0.8346 | 0.8344 | 1.0000 |
| Q | `quote_exact` | 0.0000 | 0.9413 | 0.9414 | 0.8121 | 0.8122 | 1.0001 |

Takeaway:
- `J`: `SOAR` matches direct performance on the primary metric and is effectively identical on semantic retention.
- `Q`: corrected `SOAR` slightly exceeds direct on both the primary metric and answer F1, with prompt-only near zero.

## Claim-Safe Language

Recommended contribution bullets:
- We introduce a semantics-fixed protocol for studying operational overlays in frozen language models using deterministic parsing and saved artifact traces.
- We show that dense-delta residualization recovers two full-scale single overlays, `J` and `Q`, with negligible degradation relative to their direct single-overlay adapters.
- We provide an auditable artifact chain for the reduced-scope `M4` study, including corrected metric handling, archived tainted runs, and regenerated canonical reports.

Recommended results sentence:
- On the corrected full-scale `M4` bundle, `S + R_J` matches direct `J` on strict JSON validity, and `S + R_Q` matches or slightly exceeds direct `Q` on quote exactness and answer F1.

Recommended limits sentence:
- These results support strong single-overlay recoverability for `J` and `Q`, but they do not yet establish unseen pair or triple compositionality.

## Suggested Abstract Language

Candidate abstract core:

> We study whether operational output behaviors can be separated from shared task semantics in frozen language models. In a semantics-fixed evidence-grounded QA setting, we train a shared semantic scaffold and behavior-specific adapters for JSON and quote outputs, then form Semantic-Operational Adapter Residuals (SOAR) by subtracting the scaffold in effective dense-delta space. On the corrected full-scale single-overlay study, the resulting residualized systems recover both behaviors cleanly: `S + R_J` matches direct JSON performance, and `S + R_Q` matches or slightly exceeds direct quote performance while preserving answer quality. These results support the view that some operational behaviors are recoverable as reusable residual overlays in a controlled setting, while broader composition claims remain future work.

## Do Not Claim

- Do not claim pairwise or triple composition from this bundle.
- Do not claim that `SOAR` generally replaces direct multi-behavior training.
- Do not claim universal modularity across overlays or model families.
- Do not claim a `C` success case at full scale from the current reduced-scope path.
- Do not present `C4` as fully supported; the factor-subtraction ablation is still missing.

## Appendix Placement For `C`

Recommended appendix-safe wording:
- direct full-scale `C` training exists at `runs/oa_m4_full_c_17/`
- the clean downstream full-scale `C` evidence chain does not exist
- therefore `C` should appear, if at all, as scope/operations context in an appendix note rather than as a main-result row

Recommended appendix assets:
- `appendix_c_scope_note.md`
- `tables/tableA1_c_artifact_status.md`

Do not write:
- that `C` was never run
- that full-scale `C` succeeded
- that full-scale `C` failed

## If More Compute Is Approved

- The only default next compute candidate should be a single `JQ` pair in `M5`.
- Do not auto-expand to `JC`, `CQ`, or any triple until the paper framing decision is made explicitly.
