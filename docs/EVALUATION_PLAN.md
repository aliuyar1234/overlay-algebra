# EVALUATION_PLAN.md

## 1. Evaluation goals
The evaluation is designed to answer four questions:

1. **Single-overlay recovery:** does `S + R_k` behave like a usable overlay?
2. **Unseen composition:** do composed residuals beat obvious baselines?
3. **Behavior differences:** which overlays compose more cleanly than others?
4. **Method validity:** do dense deltas matter, and is the evaluation deterministic?

## 2. Core dataset plan

### 2.1 Core data source
- raw source: SQuAD 1.1
- use only answerable examples
- filter to one-support-sentence items
- process once and version the result

### 2.2 Splits
- start from official train/dev
- within official train, create train/val by article/title
- use official dev (after filtering) as test
- no title may appear in both train and val/test

### 2.3 Leakage risks
- same article title across splits
- re-running sentence splitting differently and rebuilding targets
- accidental use of dev/test during coefficient tuning

Required audits:
- title overlap audit
- exact dataset hash / manifest
- parser round-trip audit on the processed dataset
- canonical JSON / quote-literal golden tests

## 3. Optional late-stage transfer slice
Purpose:
- modest external-validity check, not a new benchmark

Plan:
- ~300 manually curated QA items from public documents
- each item annotated with answer text, support sentence ID, support sentence text
- compile with the same overlay grammar

If not implemented, C8 remains unsupported and operational-relevance language must stay narrow.

## 4. Models compared
Core compared systems:

1. `S`
2. `S + prompt-only overlay instructions`
3. direct `F_J`, `F_C`, `F_Q`
4. direct upper bounds `F_JC`, `F_JQ`, `F_CQ`, `F_JCQ`
5. tuned whole-adapter merge for each unseen pair/triple
6. SOAR single overlays `S + R_J`, `S + R_C`, `S + R_Q`
7. SOAR pair/triple compositions

Optional sanity baseline:
- base model `M0` with prompt-only instructions (informative but not a required main-table baseline)

## 5. Primary metrics

### 5.1 Semantic metrics
- parsed answer EM
- parsed answer token F1

### 5.2 Overlay primary metrics
- `J`: strict-schema valid rate
- `C`: citation exact sentence-ID accuracy
- `Q`: normalized exact quote match

### 5.3 Overlay secondary metrics
- `J`: JSON valid rate
- `Q`: quote token F1
- all conditions: semantic retention gap vs `S`

### 5.4 Derived metrics
- **composition gap:** difference between a SOAR composition and the direct upper bound for the same overlay set
- **collateral damage:** average drop on inactive overlay metrics or semantic F1 when activating an overlay
- **recovery ratio:** overlay_primary(`S+R_k`) / overlay_primary(`F_k`)

## 6. Experiments and claim mapping

### E0. Dataset / parser validation
Goal:
- establish deterministic, leak-controlled evaluation

Outputs:
- split audit
- round-trip parser report
- metric unit tests

Claims:
- supports C5
- prerequisite for all others

### E1. Single-overlay recovery
Compare:
- `S`
- prompt-only on `S`
- `F_k`
- `S + R_k`

For each `k in {J, C, Q}` report:
- semantic EM/F1
- primary overlay metric
- secondary overlay metrics
- recovery ratio

Claims:
- primary support for C1
- informative for C3
- informative for C4 if factor ablation is included

Support criteria:
- strong C1 if at least two overlays meet the threshold from `CLAIMS_MATRIX.md`

### E1b. Beta sensitivity
Goal:
- show whether `β_k` selection matters

Compare:
- fixed `β=1`
- chosen `β_k`

Claims:
- informs C1 and C4

### E2. Unseen pair composition
Pairs:
- `JC`
- `JQ`
- `CQ`

Compare:
- prompt-only on `S`
- tuned whole-adapter merge
- SOAR composition
- direct upper bound

Report:
- semantic EM/F1
- active overlay primary metrics
- mean active overlay score
- composition gap to upper bound
- chosen alpha tuple / merge weights

Claims:
- primary support for C2
- informs C3 and C6

### E3. Triple composition
Triple:
- `JCQ`

Compare the same systems as E2.

Claims:
- informs C3 and C6
- strengthens the paper if partially positive
- not required for the minimum success bar

### E4. Dense-delta vs factor subtraction ablation
Goal:
- test whether the dense-delta formulation matters in practice

Compare:
- SOAR (dense delta)
- a clearly documented raw-factor subtraction baseline (only as ablation, never as main method)

Claims:
- supports/weakens C4

### E5. Deterministic evaluation audit
Goal:
- prove the evaluation itself is well-defined

Artifacts:
- parser goldens
- round-trip compiler tests
- canonical JSON and embedded-quote golden cases
- metric fixtures
- claim-check fixture

Claims:
- supports C5

### E6. Analysis and failure mapping
Possible analyses:
- module subset ablation (`o_proj + down_proj` vs all linear)
- residual rank sweep (16 vs 32)
- alpha sweep heatmaps
- overlap / energy analysis by module
- qualitative failure examples

Claims:
- supports C3 and C6
- can rescue the paper if C2 is weak but the failure story is strong

### E7. Optional transfer slice
Goal:
- test limited external validity

Claims:
- supports or leaves open C8

## 7. What counts as support, weak support, or failure

### C1
- strong: >= 2 overlays recover well under the formal threshold
- weak: only 1 overlay recovers, or recovery is lower but still clearly above prompt-only
- failure: no overlay recovers meaningfully

### C2
- strong: at least one pair beats both baselines under the guardrail
- weak: beats one baseline or wins only without the guardrail
- failure: never beats tuned merge and prompt-only

### C3
- strong: JSON consistently has the lowest composition gap / collateral damage
- weak: trend appears but is noisy
- failure: no stable ordering or reverse ordering

### C4
- strong: theory + ablation both favor dense deltas
- weak: theory only, empirical ablation inconclusive
- failure: factor subtraction performs equivalently and no practical argument remains

### C5
- strong: all deterministic tests pass
- weak: small parser exceptions remain but main results are still traceable
- failure: evaluation cannot be replayed exactly

## 8. Error analysis plan
For failures, classify at least:
- valid answer but invalid JSON
- right answer, wrong support ID
- right answer, quote from wrong sentence
- quote truncated or paraphrased
- composition collapse (one overlay dominates the other)
- semantic degradation without overlay gain

Save at least 20 manually inspected examples across success and failure buckets.

## 9. Expected artifacts
Tables:
- T1: dataset stats
- T2: single-overlay recovery
- T3: pair composition
- T4: triple composition
- T5: ablations / analysis

Figures:
- F1: method diagram
- F2: single-example overlay/composition demo
- F3: pairwise compatibility heatmap
- F4: alpha/beta sweep plot
- F5: residual energy / overlap plot

Files:
- metrics JSON / CSV
- predictions JSONL
- tuning manifests
- qualitative examples markdown or JSONL

## 10. Compute and statistical treatment
Target hardware:
- one workstation-class GPU with ~96 GB VRAM

Discipline:
- development: 1 seed is acceptable
- final evidence: if budget allows, run 2-3 confirmation seeds on the core C1/C2 comparisons
- always report example-level bootstrap confidence intervals if multiple seeds are not available
- do not pretend single-seed differences are definitive

Expected full-core budget:
- roughly 18-27 GPU hours for the scoped v1 path

## 11. Reproducibility notes
- every table row must map to saved predictions and a config
- no manual spreadsheet-only metric computation
- tuning logs must be saved
- if a result comes from a changed compiler or changed sentence splitter, it is a new dataset version
