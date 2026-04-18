# Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/overlay-algebra/raw/main/paper/tmlr/overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19313240.svg)](https://doi.org/10.5281/zenodo.19313240)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-TMLR%20source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/tmlr/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0F766E?style=flat-square)](LICENSE)
[![Scope](https://img.shields.io/badge/Scope-J%2FQ%20single--overlay-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models*

This repository accompanies a methods paper on whether deployment-style output requirements can be isolated as reusable low-rank overlays on top of shared question-answering semantics in a frozen language model. The current public release centers the validated reduced-scope `J/Q` paper path and the TMLR LaTeX manuscript.

## Abstract

Post-training adapters often mix task semantics with operational output behavior, which makes behavior reuse hard to study cleanly. We consider a deliberately controlled version of that problem. Starting from filtered SQuAD 1.1 question answering, we retain only examples with exactly one gold support sentence, compile targets deterministically, and evaluate with parsers rather than an LLM judge. On this fixed QA substrate, we train a semantic scaffold adapter for plain answers and direct single-overlay adapters for two deployment-style output contracts: strict JSON and exact support quotation. We then construct *Semantic-Operational Adapter Residuals* (SOAR) by subtracting the scaffold in effective dense-delta space and recompressing the result.

On the corrected full-scale study, both overlays recover cleanly. For JSON, S + R_J matches the direct adapter on strict schema validity (0.9986 vs. 0.9986) with essentially unchanged answer F1 (0.8344 vs. 0.8346). For exact quotation, S + R_Q matches or slightly exceeds the direct adapter on quote exactness (0.9414 vs. 0.9413) and answer F1 (0.8122 vs. 0.8121). Under the same output contracts, prompt-only control on the scaffold is near zero on the active overlay metric in both cases, so the recovered behavior is not explained by prompting alone.

We present these results as evidence for *partial operational recoverability*, not universal modularity. The validated claim surface is intentionally narrow: the main contribution of this paper is a semantics-fixed protocol, a dense-delta residualization method, and full-scale single-overlay recovery for two operational behaviors with a deterministic, auditable evaluation pipeline.

## Main Result

On the corrected full-scale study:

| Overlay | Metric | Direct adapter | S + R (recovered) | Prompt-only control |
| ------- | ------ | -------------- | ----------------- | ------------------- |
| J (JSON)  | Strict schema validity | 0.9986 | **0.9986** | near zero |
| J (JSON)  | Answer F1              | 0.8346 | 0.8344 | — |
| Q (Quote) | Quote exactness        | 0.9413 | **0.9414** | near zero |
| Q (Quote) | Answer F1              | 0.8121 | 0.8122 | — |

- `S + R_J` matches the direct JSON adapter on strict schema validity while remaining effectively identical on answer F1.
- `S + R_Q` matches or slightly exceeds the direct quote adapter on quote exactness and answer F1.
- Prompt-only control under the same output contract collapses on the active overlay metric in both cases.

The contribution is not breadth. It is a tightly controlled semantics-fixed protocol, a dense-delta residualization method, and an auditable artifact-backed recovery result for two operational behaviors.

## Contributions

1. A semantics-fixed QA protocol built from filtered SQuAD 1.1 single-support examples with deterministic target compilation and parser-based evaluation.
2. **SOAR** (**S**emantic-**O**perational **A**dapter **R**esiduals): a dense-delta residualization method that subtracts a shared semantic scaffold and recompresses the result into a deployable low-rank overlay.
3. Full-scale single-overlay recovery for two operational behaviors (strict JSON, exact support quotation) matching direct single-overlay adapters on their active metrics.
4. An honest scope statement: the paper supports partial operational recoverability for two overlays, not universal modularity or general composition.

## Scope

This public release is intentionally narrow and claim-safe.

- Validated overlays in the main paper: `J` (JSON) and `Q` (exact quote)
- `C` (citation): appendix / scope context only
- Main supported claims: `C1` and `C5`
- Partially supported claim: `C4`
- Out of main validated claim path: `C2`, `C3`, `C6`, `C7`, `C8`
- Canonical `M5` composition: intentionally skipped on the default paper path

The strongest current paper is therefore a **single-overlay recovery paper**, not a composition paper.

## Paper

- Download the manuscript PDF: [`overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf`](paper/tmlr/overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf)
- LaTeX source: [`paper/tmlr/main.tex`](paper/tmlr/main.tex)
- Build notes: [`paper/tmlr/BUILD_INSTRUCTIONS.md`](paper/tmlr/BUILD_INSTRUCTIONS.md)
- Compiled build output: [`paper/tmlr/main.pdf`](paper/tmlr/main.pdf)

## Repository Layout

- [`paper/tmlr/`](paper/tmlr/) — TMLR LaTeX manuscript, figures, appendix, and build instructions
- [`src/overlay_algebra/`](src/overlay_algebra/) — main Python implementation
- [`configs/`](configs/) — train, eval, analysis, and reporting configs
- [`tests/`](tests/) — parser, metrics, runtime, and report checks
- [`reports/full_single_overlay_recovery_j/`](reports/full_single_overlay_recovery_j/) — paper-facing JSON recovery report
- [`reports/full_single_overlay_recovery_q/`](reports/full_single_overlay_recovery_q/) — paper-facing quote recovery report
- [`reports/full_single_overlay_study/`](reports/full_single_overlay_study/) — aggregate reduced-scope single-overlay report
- [`reports/claim_checks_c1_c4_c5/`](reports/claim_checks_c1_c4_c5/) — saved claim-check outputs for the public paper path
- [`reports/paper_artifact_bundle_jq_m4/`](reports/paper_artifact_bundle_jq_m4/) — supporting paper-writing assets and tables/figures
- [`docs/`](docs/) — research brief, method spec, claims matrix, and evaluation plan

## Reproducibility

Environment:

```powershell
uv sync
.\.venv\Scripts\python -m pytest -q
```

Build the manuscript:

```powershell
cd paper/tmlr
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Regenerate the reduced-scope paper-facing reports:

```powershell
python -m overlay_algebra.reports.full_single_overlay --config configs/reports/full_single_overlay_JQ.yaml --json
python -m overlay_algebra.reports.claim_checks --config configs/reports/c1_c4_c5.yaml --json
python -m overlay_algebra.reports.export_paper_artifacts --config configs/reports/paper_jq_m4.yaml --json
```

Optional artifact reproduction:

```powershell
python -m overlay_algebra.data.build_dataset --config configs/data/squad_jcq_core.yaml --json
python -m overlay_algebra.eval.bundle --config configs/eval/full_bundle_J.yaml --json
python -m overlay_algebra.eval.bundle --config configs/eval/full_bundle_Q.yaml --json
python -m overlay_algebra.analysis.residualize --config configs/analysis/full_residuals_J.yaml --json
python -m overlay_algebra.analysis.residualize --config configs/analysis/full_residuals_Q.yaml --json
```

Public docs:

- [`docs/RESEARCH_BRIEF.md`](docs/RESEARCH_BRIEF.md)
- [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md)
- [`docs/CLAIMS_MATRIX.md`](docs/CLAIMS_MATRIX.md)
- [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md)
- [`docs/FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md`](docs/FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md)
- [`docs/RELATED_WORK_MAP.md`](docs/RELATED_WORK_MAP.md)
- [`docs/REPRODUCIBILITY_AND_ENV.md`](docs/REPRODUCIBILITY_AND_ENV.md)

## License

This repository is released under the [Apache-2.0 License](LICENSE).

## Citation

```bibtex
@misc{uyar2026overlayalgebra,
  author       = {Uyar, Ali},
  title        = {Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models},
  year         = {2026},
  doi          = {10.5281/zenodo.19313240},
  url          = {https://doi.org/10.5281/zenodo.19313240},
  note         = {TMLR preprint},
  howpublished = {\url{https://github.com/aliuyar1234/overlay-algebra}}
}
```

Machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff).
