# Overlay Algebra

**Paper title:** *Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models*

This repository is the public research/code surface for the project. It keeps the core method, public configs, code, tests, and paper-facing artifacts, while internal operational notes and private workflow documents stay local.

The primary paper source is now the LaTeX TMLR project in `paper/tmlr/`. Earlier Markdown drafting assets remain useful as supporting artifacts, but the canonical manuscript source has moved to LaTeX.

## Project summary
The project asks a narrow, operational question:

> In a semantics-fixed QA setting, can deployment-style output requirements be isolated as reusable low-rank overlays on top of shared semantics?

The v1 overlays are:
- **J**: valid JSON output
- **C**: support sentence citation
- **Q**: exact support quote

The core method is **SOAR**: **S**emantic-**O**perational **A**dapter **R**esiduals.  
Train a semantic scaffold adapter, train single-overlay adapters on the same examples, convert each adapter to effective dense deltas, subtract the shared scaffold delta, compress the remainder, and test whether the resulting residuals can be added back together to produce unseen behavior combinations.

## Central thesis
Current post-training often entangles:
1. what the model knows about a task, and
2. how the model is operationally required to answer.

If those can be separated even partially, teams may be able to reuse narrow behavior overlays instead of retraining a new monolithic adapter for every combination.

## Target contribution
This repo should produce:
1. a **semantics-fixed experimental protocol** for operational overlay composition,
2. a **fully specified residualization method** in dense-delta space,
3. **evidence** about what is composable, what is not, and what fails,
4. a **reproducible implementation** with honest claim tracking.

## What kind of paper this is
- empirical ML methods paper
- causal / ablation-heavy study
- reproducibility-aware experimental systems work
- not a benchmark-first paper
- not a generic adapter-merging paper

## High-level stack
- base model: frozen instruct LLM (planned default: Qwen2.5-7B-Instruct)
- adaptation: LoRA / PEFT
- data: filtered SQuAD 1.1 answerable-only QA with one support sentence
- evaluation: deterministic parsers; no LLM judges in core claims
- reference math/code: dense-delta residualization + TSVD recompression
- reference compiler/parsers: canonical JSON serialization + strict quote parsing

## Core components
- dataset compiler with sentence labels and deterministic targets
- semantic scaffold trainer
- single-overlay adapter trainer
- direct pair/triple upper-bound trainer
- dense-delta extractor
- SOAR residualizer
- composition / tuning engine
- evaluation + claim-check pipeline
- reference implementations for SOAR math and compiler/parser goldens

## Expected artifacts from this repo
- code in `src/overlay_algebra/`
- configs in `configs/`
- tests in `tests/`
- processed dataset manifests
- trained adapters and residual overlays
- metrics JSON / CSV
- prediction JSONL files
- paper tables / plots / qualitative examples
- claim-check outputs

## Current snapshot
As of `2026-03-29`:
- the reduced-scope full-scale `J/Q` single-overlay bundle is complete
- the strongest supported paper path is a single-overlay recovery paper, not a composition paper
- `C1` and `C5` are supported on the corrected full-scale `J/Q` path
- `C4` remains partially supported
- canonical `M5` composition is intentionally deferred on the default paper path
- direct full-scale `C` training exists, but `C` is treated only as appendix / scope context in the current paper framing

Public paper-facing artifacts live under:
- `paper/tmlr/`
- `reports/full_single_overlay_recovery_j/`
- `reports/full_single_overlay_recovery_q/`
- `reports/full_single_overlay_study/`
- `reports/claim_checks_c1_c4_c5/`
- `reports/paper_artifact_bundle_jq_m4/`

Internal working notes, evidence logs, and ops-oriented status documents are intentionally kept out of the public repo surface.

## What Is Implemented
- deterministic dataset compiler, split audit, strict parsers, and metric fixtures
- smoke train/eval path with saved checkpoints, manifests, and predictions
- dense-delta SOAR residualization with beta tuning and compact delta artifacts
- pilot-scale single-overlay recovery path for `S`, `J`, `C`, and `Q`
- full-study `M4` config/report scaffolding for train, residualization, eval, and claim checks
- long-run training controls: checkpointing, pause, resume, and heartbeat
- long-run residualization controls: checkpointing, pause, resume, heartbeat, cache-aware eval reuse, and length-aware batched validation
- runtime efficiency slice for full runs: larger microbatch at constant effective batch, dataset-versioned train/eval feature caches, length-aware batched greedy eval, load-once bundled eval entrypoints, and decode-budget audit tooling
- review-driven runtime/protocol guardrails: temporary sweep adapters, bundle-resume-safe aggregate predictions, config-driven composition report margin, explicit compose cache/sort controls, matched-budget compose search, and review-bundle-safe pilot fixture handling
- `M5` pair/triple code surfaces: direct upper-bound train configs, resumable composition tuning, tuned merge baseline, pair/triple eval configs, and composition-study reporting
- `M6` code surfaces: resumable ablation runner, rank/factor/energy/manifest aggregation tasks, and expanded claim checks for `C2/C3/C6/C8`
- `M7` code surfaces: transfer-eval config template and paper artifact export bundle script

## Paper Source
- LaTeX manuscript source: `paper/tmlr/main.tex`
- Build notes: `paper/tmlr/BUILD_INSTRUCTIONS.md`
- Shareable manuscript PDF: `paper/tmlr/overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf`
- Build output PDF: `paper/tmlr/main.pdf`
- Current paper mode: TMLR preprint with author block `Ali Uyar, Independent Researcher`
- Supporting drafting bundle: `reports/paper_artifact_bundle_jq_m4/`

## How To Run
Environment:
- `uv sync`
- `.venv\Scripts\python -m pytest -q`

Build the manuscript:
- `cd paper/tmlr`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

Regenerate the reduced-scope paper-facing reports:
- `python -m overlay_algebra.reports.full_single_overlay --config configs/reports/full_single_overlay_JQ.yaml --json`
- `python -m overlay_algebra.reports.claim_checks --config configs/reports/c1_c4_c5.yaml --json`
- `python -m overlay_algebra.reports.export_paper_artifacts --config configs/reports/paper_jq_m4.yaml --json`

Optional artifact reproduction commands:
- `python -m overlay_algebra.data.build_dataset --config configs/data/squad_jcq_core.yaml --json`
- `python -m overlay_algebra.eval.bundle --config configs/eval/full_bundle_J.yaml --json`
- `python -m overlay_algebra.eval.bundle --config configs/eval/full_bundle_Q.yaml --json`
- `python -m overlay_algebra.analysis.residualize --config configs/analysis/full_residuals_J.yaml --json`
- `python -m overlay_algebra.analysis.residualize --config configs/analysis/full_residuals_Q.yaml --json`

Long-run controls:
- pause: create `runs/<run_id>/control/pause.request`
- resume live paused run: delete `runs/<run_id>/control/pause.request`
- resume interrupted run: `python -m overlay_algebra.train.fit --config <config> --resume-latest --json`
- resume explicit checkpoint: `python -m overlay_algebra.train.fit --config <config> --resume-checkpoint <path> --json`
- residualize resume latest: `python -m overlay_algebra.analysis.residualize --config <config> --resume-latest --json`
- residualize resume explicit checkpoint: `python -m overlay_algebra.analysis.residualize --config <config> --resume-checkpoint <path> --json`
- compose resume latest: `python -m overlay_algebra.analysis.compose --config <config> --resume-latest --json`
- compose resume explicit checkpoint: `python -m overlay_algebra.analysis.compose --config <config> --resume-checkpoint <path> --json`
- ablate resume latest: `python -m overlay_algebra.analysis.ablate --config <config> --resume-latest --json`
- ablate resume explicit checkpoint: `python -m overlay_algebra.analysis.ablate --config <config> --resume-checkpoint <path> --json`
- bundled eval resume latest: `python -m overlay_algebra.eval.bundle --config <config> --resume-latest --json`
- bundled eval resume explicit checkpoint: `python -m overlay_algebra.eval.bundle --config <config> --resume-checkpoint <path> --json`
- decode audit resume latest: `python -m overlay_algebra.analysis.decode_audit --config <config> --resume-latest --json`
- decode audit resume explicit checkpoint: `python -m overlay_algebra.analysis.decode_audit --config <config> --resume-checkpoint <path> --json`

## What's Next
1. Polish the TMLR manuscript and venue-facing metadata around `paper/tmlr/`.
2. Keep the public repo surface centered on the validated reduced-scope `J/Q` paper path.
3. Treat `C` as appendix / scope context unless the clean downstream `C` evidence chain is completed later.
4. Reopen `M5` composition only if paper scope and compute budget are explicitly expanded in a later pass.

## Public docs
- project framing: `docs/RESEARCH_BRIEF.md`
- exact method: `docs/METHOD_SPEC.md`
- claim tracking: `docs/CLAIMS_MATRIX.md`
- evaluation plan: `docs/EVALUATION_PLAN.md`
- formalism / pseudocode: `docs/FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md`
- related-work map: `docs/RELATED_WORK_MAP.md`
- reproducibility: `docs/REPRODUCIBILITY_AND_ENV.md`
