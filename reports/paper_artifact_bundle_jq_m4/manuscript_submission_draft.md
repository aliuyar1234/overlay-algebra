# SOAR on a Semantics-Fixed QA Substrate: Submission Draft

This draft is intentionally claim-safe relative to the current reduced-scope artifact bundle.
It treats the corrected full-scale `J/Q` single-overlay results as the main paper and leaves pair/triple composition out of the main result path.

## Title

Semantic-Operational Adapter Residuals for Single-Overlay Recovery in Frozen Language Models

## Abstract

Post-training adapters often entangle task semantics with operational output behavior, which makes behavior reuse difficult to study cleanly. We ask whether some output requirements can be isolated as reusable residual overlays on top of a shared semantic scaffold in a frozen language model. Our setting is deliberately controlled: filtered SQuAD 1.1 question answering with one gold support sentence per example, deterministic target compilation, and parser-based evaluation without an LLM judge in the critical path. We train a semantic scaffold adapter and direct single-overlay adapters for JSON (`J`) and exact quote (`Q`) outputs, then form Semantic-Operational Adapter Residuals (SOAR) by subtracting the scaffold in effective dense-delta space and recompressing the result.

On the corrected full-scale single-overlay study, both overlays recover cleanly. `S + R_J` matches the direct JSON adapter on strict schema validity (`0.9986` vs. `0.9986`) while preserving answer quality (`0.8344` vs. `0.8346` answer F1). `S + R_Q` matches or slightly exceeds the direct quote adapter on both quote exactness (`0.9414` vs. `0.9413`) and answer F1 (`0.8122` vs. `0.8121`). In both cases, the prompt-only scaffold under the same output contract is near zero on the active overlay metric, indicating that the recovered behavior is not explained by prompting alone.

These results support a strong single-overlay residualization claim in a semantics-fixed frozen-LM QA setting. The current paper path is intentionally narrower than a full composition story: `C1` and `C5` are supported, `C4` remains partial, and pair/triple composition claims are deferred. We therefore present SOAR as evidence for partial operational recoverability, not as a claim of universal modularity or full equivalence to direct multi-behavior training.

## 1. Introduction

Post-training often bundles together at least two things we would like to study separately: the semantic content of a task and the operational contract that constrains how an answer must be rendered. In practice, this means that a model tuned for a narrow output behavior such as strict JSON or exact supporting quotes is also free to absorb the underlying QA semantics into the same update. Once those ingredients are mixed together, it becomes hard to ask whether the behavior itself is reusable.

This paper studies a deliberately narrow version of that question. We fix the semantic substrate to evidence-grounded question answering with one gold support sentence per example and vary only the output contract. Our target behaviors are JSON formatting (`J`), citation (`C`), and exact quote output (`Q`), but the corrected default paper path focuses on the two full-scale overlays that currently have clean evidence: `J` and `Q`.

Our method constructs a shared semantic scaffold adapter `S`, trains direct single-overlay adapters `F_k`, and then forms Semantic-Operational Adapter Residuals (SOAR) by subtracting the scaffold in effective dense-delta space and recompressing the result. The key empirical question is not whether these residuals are interesting in the abstract, but whether a residualized system `S + R_k` can recover the corresponding operational behavior without substantial semantic degradation.

The current contribution is intentionally scoped. We do not claim universal behavior atoms, general adapter composition guarantees, or full equivalence to direct multi-behavior training. Instead, we make three narrower claims. First, under a semantics-fixed protocol, at least two operational overlays can be recovered cleanly at full scale from a shared scaffold (`C1`). Second, the evaluation pipeline is deterministic, parser-based, and auditable enough to support that claim (`C5`). Third, dense deltas remain the principled technical object for residualization, although the missing factor-subtraction ablation means that claim remains partial rather than complete (`C4`).

## 2. Related Work Positioning

The related-work section should preserve the novelty boundary in `docs/RELATED_WORK_MAP.md` rather than overclaiming novelty. The paper is adjacent to four lines of work: behavior reading from trained updates, generic adapter composition and merging, structured-output or citation-control papers, and token- or activation-space steering methods. The responsible positioning is that this project contributes a semantics-fixed protocol plus a dense-delta residualization method for operational overlays. It should not claim to be the first decomposition of behavior from adapters, the first adapter composition method, or a broadly superior alternative to generic merge/routing systems.

Before submission, replace any placeholder references here with verified citations from the related-work map. The novelty claim should remain narrow even after citations are added.

## 3. Problem Setup

We use a filtered SQuAD 1.1 derivative in which every retained example has exactly one unambiguous support sentence containing the gold answer span. Contexts are rendered with sentence labels `[S1]`, `[S2]`, and so on. The output contract changes by overlay condition, but the question, labeled context, and underlying answer semantics do not.

The processed dataset version in this bundle is `squad_jcq_v1`. It contains `78,098` training examples, `9,353` validation examples, and `10,554` test examples, with `164` raw examples dropped because the answer span did not map to exactly one sentence. The split audit reports no title overlap across train, validation, and test, and the dataset manifest reports `0` round-trip compilation failures over `98,005` checked processed examples.

The current default paper path uses only the `J` and `Q` overlays in the main result section. `C` remains part of the project scope, and direct full-scale `C` training did complete, but the full downstream `C` residualization/evaluation/report chain was not completed cleanly enough to support a full-scale results row. We therefore treat `C` as appendix-style scope context rather than as main-result evidence.

## 4. Method

Let `S` denote a semantic scaffold adapter trained on plain-answer targets, and let `F_k` denote a direct single-overlay adapter for overlay `k`. For each adapted module `m`, we work with effective dense deltas rather than raw LoRA factors:

`Delta_m = (alpha_m / r_m) * B_m @ A_m`

This choice matters because raw factor representations are not unique, while the effective dense update is the stable object that actually modifies the frozen backbone.

For each overlay `k`, we compute a raw residual by subtracting a scaled scaffold delta from the direct overlay delta:

`Gamma_k,m = Delta F_k,m - beta_k * Delta S_m`

We then recompress the residual with truncated SVD to obtain a deployable low-rank residual adapter `R_k`. In the full-scale `J/Q` path, validation selected `beta = 1.0` for both overlays under the standard semantic guardrail.

The deployment-time residualized system is `S + R_k`. In the broader project design, pair or triple systems would compose multiple residuals on top of the scaffold. However, that composition step is intentionally outside the default paper claim path for the current reduced-scope bundle.

## 5. Experimental Setup

All main-path experiments use the locked backbone `Qwen/Qwen2.5-7B-Instruct`. Adapters target `o_proj` and `down_proj`. Training uses the same semantics-fixed prompt protocol across conditions, while only the output format specification and target string change. Evaluation is parser-based and deterministic, with greedy decoding and no judge model in the critical claim path.

We report two kinds of outcomes. The first is the active overlay metric: `json_strict_schema_valid` for `J` and `quote_exact` for `Q`. The second is semantic retention, measured by answer F1 after parser-based answer extraction. Prompt-only means the scaffold adapter `S` is asked to satisfy the overlay contract via prompting alone. Direct means the single-overlay adapter `F_k`. SOAR means the residualized system `S + R_k`.

This paper reports one seed (`17`) and therefore should present its evidence honestly as a single-seed full-scale result bundle. The contribution lives in the controlled protocol and the saved artifact chain, not in a claim of broad statistical coverage.

## 6. Results: Single-Overlay Recovery

Table 1 and Figure 1 summarize the main result. In both overlays, prompt-only is near zero on the active overlay metric, while SOAR closely matches the direct adapter. That is exactly the recovery signature the paper needs to justify calling `R_k` an operational residual rather than a redundant restatement of prompt conditioning.

For `J`, the result is effectively exact: `S + R_J` matches the direct adapter on strict JSON-schema validity (`0.9986` vs. `0.9986`) and stays nearly identical on answer F1 (`0.8344` vs. `0.8346`). For `Q`, the corrected full-scale result is similarly strong: `S + R_Q` reaches `0.9414` quote exactness versus `0.9413` for the direct adapter, while slightly exceeding it on answer F1 (`0.8122` vs. `0.8121`).

The `Q` result is especially important because the project previously surfaced a real metric aggregation bug. The canonical tainted `Q` outputs were archived, the metric path was repaired, and the corrected reports in this bundle now show that the apparent `Q` weakness was an evaluation bug rather than a failure of residualization. That artifact trail strengthens the credibility of the current bundle: it shows that the pipeline does not hide failures and that corrected numbers can be traced to regenerated outputs.

Taken together, the `J` and `Q` rows are enough to support `C1` under the repo's own thresholding logic. They are not enough to support pair or triple composition, and the paper should say so directly.

## 7. Reliability and Claim Scope

The project's reproducibility contribution is narrower than a full multi-milestone result claim, but it is real. `C5` is supported because the processed dataset manifest reports zero round-trip failures, the split audit passes, configs and predictions are saved, and the evaluation path remains parser-based rather than judge-based. Table 2 translates that into a manuscript-safe claim boundary.

`C4` should remain partial in the main text. The paper can say that dense deltas are the principled technical object because factor-space subtraction is representation-dependent, but it should not claim empirical superiority over factor subtraction without the missing ablation.

## 8. Discussion and Limits

This is not a universal modularity paper. It is a controlled single-overlay recovery paper. The strongest justified interpretation is that some operational behaviors can be isolated from shared semantics in a frozen-LM QA setting and reintroduced through dense-delta residualization. The current default path does not establish unseen pairwise composition, triple composition, or transfer beyond the syntheticized QA substrate. It also does not establish a clean full-scale citation success case, even though the direct full-scale `C` training artifact exists.

Those limitations are not side notes; they are part of the paper's framing. The contribution is strongest when stated narrowly: a semantics-fixed protocol, a dense-delta residualization method, and full-scale positive evidence for two operational overlays.

## Appendix A. Citation (`C`) Scope Note

The citation overlay should be handled as appendix-style context rather than as a main paper result. The repo contains a completed direct full-scale `C` training artifact at `runs/oa_m4_full_c_17/`, so it would be misleading to imply that `C` was never run at all. However, the clean downstream full-scale evidence chain for `C` does not exist: the only full-scale `C` residualization attempt in the current repo state is the archived off-path run `runs/oa_m4_full_residualize_c_17_archived_manual_stop_off_path_2026-03-26/`, which stopped in preparation before any useful evaluation units completed. There is therefore no clean full-scale `C` residualization, no full-scale `C` prompt/direct/SOAR evaluation bundle, and no full-scale `C` recovery report to support a claim-safe main-table row.

The right manuscript use of `C` is therefore narrow. It is acceptable to say that `C` remained in scope, that direct full-scale `C` training was completed, and that the default paper path deprioritized downstream `C` compute in favor of the cleaner `J/Q` evidence. It is not acceptable to imply a positive or negative full-scale `C` recovery result from the current artifact set. If desired, Appendix Table A1 can be used to make that status explicit.

## Table 1. Corrected Full-Scale Single-Overlay Recovery

| Overlay | Primary Metric | Prompt-only | Direct | SOAR | Prompt-only Answer F1 | Direct Answer F1 | SOAR Answer F1 | Recovery Ratio | Chosen Beta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J | `json_strict_schema_valid` | 0.0000 | 0.9986 | 0.9986 | 0.0000 | 0.8346 | 0.8344 | 1.0000 | 1.0 |
| Q | `quote_exact` | 0.0000 | 0.9413 | 0.9414 | 0.0000 | 0.8121 | 0.8122 | 1.0001 | 1.0 |

## Table 2. Claim Scope Used In This Draft

| Claim | Status | Role in manuscript |
|---|---|---|
| `C1` | supported | main empirical claim |
| `C5` | supported | reproducibility / measurement claim |
| `C4` | partially supported | methods rationale only; not a full empirical claim |
| `C2`, `C3`, `C6`, `C7`, `C8` | not on the main claim path | defer or frame as future work only |

## Figure 1

Use `figures/figure1_single_overlay_recovery.svg` with the caption in `figures/figure1_caption.md`.

## Appendix Table A1

Use `tables/tableA1_c_artifact_status.md` if the paper needs an explicit appendix-facing statement of what exists and what does not for `C`.

## Submission Notes

- Keep the paper centered on the corrected `J/Q` single-overlay evidence.
- Do not mention `M5` as an unfinished near-result in the main body; if needed, note only that broader composition claims are deferred.
- Before submission, replace the Related Work placeholders with verified citations from `docs/RELATED_WORK_MAP.md`.
