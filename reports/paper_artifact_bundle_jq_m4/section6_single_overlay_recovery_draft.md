# Section 6 Draft: Single-Overlay Recovery

Our primary single-overlay question is whether a residualized system of the form `S + R_k` can recover the behavior of the corresponding direct single-overlay adapter `F_k` without substantial semantic degradation. Table 1 and Figure 1 summarize the corrected full-scale results for the reduced-scope `J/Q` path. In both overlays, the prompt-only scaffold under the same output contract is near zero on the active overlay metric, while the residualized system closely matches the direct adapter. This is the core empirical pattern required for `C1`: the residual must contribute genuine operational behavior beyond prompting alone while preserving the underlying QA task.

For both overlays, validation selected `beta = 1.0` under the standard semantic guardrail, so the full-scale test results below correspond to the most direct form of the residualization rule rather than a heavily tuned subtraction coefficient. This matters for interpretation: the positive results are not being driven by a fragile or exotic coefficient setting.

For `J`, the result is especially clean. `S + R_J` matches the direct adapter exactly on the primary metric, reaching `0.9986` strict JSON-schema validity, while remaining effectively identical on semantic quality (`0.8344` vs. `0.8346` answer F1). The prompt-only scaffold under the same JSON contract is not merely weaker but functionally noncompetitive on the primary metric (`0.0000`). This makes `J` the clearest full-scale case that an operational formatting requirement can be separated from shared QA semantics with no meaningful loss on the main outcome measures.

For `Q`, the corrected full-scale result is also positive. After fixing the condition-metric aggregation bug and rerunning the affected path, `S + R_Q` reaches `0.9414` quote exactness versus `0.9413` for the direct `Q` adapter, while slightly exceeding the direct adapter on answer F1 (`0.8122` vs. `0.8121`). The earlier apparent `Q` weakness was therefore not a failure of residualization but an artifact of the old metric aggregation bug, whose tainted outputs were archived and replaced with regenerated canonical reports.

Taken together, these results support a strong single-overlay residualization claim in this semantics-fixed setting. Both `J` and `Q` show recovery ratios at or above `1.0`, and both remain far above the prompt-only control on the active overlay metric. The corrected bundle therefore supports the interpretation that at least some operational behaviors can be isolated as reusable residual overlays on top of a shared semantic scaffold.

We deliberately stop short of stronger claims here. First, the current reduced-scope bundle does not include a clean full-scale `C` recovery result, so citation is not presented as a third full-scale positive case. Second, these experiments do not establish pairwise or triple compositionality. Third, they do not fully settle `C4`, because the factor-subtraction ablation remains absent. The correct reading of this section is narrower and more precise: dense-delta residualization recovers two operational overlays cleanly at full scale, which is sufficient for the paper's single-overlay contribution but not for a broader composition or universality claim.

## Table 1

| Overlay | Primary Metric | Prompt-only | Direct | SOAR | Prompt-only Answer F1 | Direct Answer F1 | SOAR Answer F1 | Recovery Ratio | Chosen Beta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J | `json_strict_schema_valid` | 0.0000 | 0.9986 | 0.9986 | 0.0000 | 0.8346 | 0.8344 | 1.0000 | 1.0 |
| Q | `quote_exact` | 0.0000 | 0.9413 | 0.9414 | 0.0000 | 0.8121 | 0.8122 | 1.0001 | 1.0 |

Suggested caption:

> Corrected full-scale single-overlay recovery results for the reduced-scope `J/Q` path. In both overlays, `SOAR` matches or slightly exceeds the direct single-overlay adapter while remaining far above the prompt-only scaffold baseline on the active operational metric. Validation selected `beta = 1.0` for both overlays under the standard semantic guardrail.

## Figure 1

Use `figures/figure1_single_overlay_recovery.svg` with the caption in `figures/figure1_caption.md`.
