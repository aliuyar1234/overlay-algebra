# Overlay Algebra

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square)](https://github.com/aliuyar1234/overlay-algebra/raw/main/paper/tmlr/overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf)
[![DOI](https://zenodo.org/badge/1195229853.svg)](https://doi.org/10.5281/zenodo.19313240)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-TMLR%20source-1D4ED8?style=flat-square)](paper/tmlr/main.tex)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0F766E?style=flat-square)](LICENSE)
[![Scope](https://img.shields.io/badge/Scope-J%2FQ%20single--overlay-374151?style=flat-square)](#current-scope)

**Paper title:** *Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models*

This repository accompanies a methods paper on whether deployment-style output requirements can be isolated as reusable low-rank overlays on top of shared question-answering semantics in a frozen language model. The current public release centers the validated reduced-scope `J/Q` paper path and the TMLR LaTeX manuscript.

## Overview

Modern post-training typically learns task semantics and output behavior together. This project studies a narrower question:

> In a semantics-fixed QA setting, can operational output contracts be separated from shared semantics strongly enough to recover them as reusable residual overlays?

The method is **SOAR**: **S**emantic-**O**perational **A**dapter **R**esiduals.

At a high level, the pipeline:
- trains a semantic scaffold adapter on plain QA targets
- trains direct single-overlay adapters under the same examples with different output contracts
- converts adapters to effective dense deltas
- subtracts the scaffold in dense-delta space
- recompresses the residual into a deployable low-rank overlay

## Current Scope

This public release is intentionally narrow and claim-safe.

- Validated overlays in the main paper: `J` (JSON) and `Q` (exact quote)
- `C` (citation): appendix / scope context only
- Main supported claims: `C1` and `C5`
- Partially supported claim: `C4`
- Out of main validated claim path: `C2`, `C3`, `C6`, `C7`, `C8`
- Canonical `M5` composition: intentionally skipped on the default paper path

The strongest current paper is therefore a **single-overlay recovery paper**, not a composition paper.

## Main Result

On the corrected full-scale study:
- `S + R_J` matches the direct JSON adapter on strict schema validity while remaining effectively identical on answer F1
- `S + R_Q` matches or slightly exceeds the direct quote adapter on quote exactness and answer F1
- prompt-only control under the same output contract collapses in both validated overlays

The contribution is not breadth. It is a tightly controlled semantics-fixed protocol, a dense-delta residualization method, and an auditable artifact-backed recovery result for two operational behaviors.

## Paper

- Download the manuscript PDF: [overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf](paper/tmlr/overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf)
- LaTeX source: [paper/tmlr/main.tex](paper/tmlr/main.tex)
- Build notes: [paper/tmlr/BUILD_INSTRUCTIONS.md](paper/tmlr/BUILD_INSTRUCTIONS.md)
- Compiled build output: [paper/tmlr/main.pdf](paper/tmlr/main.pdf)

## Repository Layout

- [paper/tmlr/](paper/tmlr/) — TMLR LaTeX manuscript, figures, appendix, and build instructions
- [src/overlay_algebra/](src/overlay_algebra/) — main Python implementation
- [configs/](configs/) — train, eval, analysis, and reporting configs
- [tests/](tests/) — parser, metrics, runtime, and report checks
- [reports/full_single_overlay_recovery_j/](reports/full_single_overlay_recovery_j/) — paper-facing JSON recovery report
- [reports/full_single_overlay_recovery_q/](reports/full_single_overlay_recovery_q/) — paper-facing quote recovery report
- [reports/full_single_overlay_study/](reports/full_single_overlay_study/) — aggregate reduced-scope single-overlay report
- [reports/claim_checks_c1_c4_c5/](reports/claim_checks_c1_c4_c5/) — saved claim-check outputs for the public paper path
- [reports/paper_artifact_bundle_jq_m4/](reports/paper_artifact_bundle_jq_m4/) — supporting paper-writing assets and tables/figures

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

## Public Docs

- [docs/RESEARCH_BRIEF.md](docs/RESEARCH_BRIEF.md)
- [docs/METHOD_SPEC.md](docs/METHOD_SPEC.md)
- [docs/CLAIMS_MATRIX.md](docs/CLAIMS_MATRIX.md)
- [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md)
- [docs/FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md](docs/FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md)
- [docs/RELATED_WORK_MAP.md](docs/RELATED_WORK_MAP.md)
- [docs/REPRODUCIBILITY_AND_ENV.md](docs/REPRODUCIBILITY_AND_ENV.md)

## Citation

If you use this repository or build on the paper, please cite it as:

```bibtex
@misc{uyar2026overlayalgebra,
  title        = {Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models},
  author       = {Ali Uyar},
  year         = {2026},
  note         = {TMLR preprint},
  howpublished = {\url{https://github.com/aliuyar1234/overlay-algebra}}
}
```

Machine-readable citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

This repository is released under the [Apache-2.0 License](LICENSE).
