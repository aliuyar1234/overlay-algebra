# Build instructions

This project uses the official TMLR style files (`tmlr.sty`, `tmlr.bst`, `math_commands.tex`).
The bibliography is included directly in `sections/references.tex`, so no BibTeX step is required.

## Build
Preferred:
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Portable:
```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Notes
- The manuscript is configured in de-anonymized TMLR preprint mode by default.
- Current author block: `Ali Uyar` / `Independent Researcher`.
- The checked-in `main.pdf` has been rebuilt successfully with MiKTeX + `latexmk` from this directory.
- The shareable named PDF artifact is `overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf`.
- After rebuilding, refresh the shareable copy with:
```powershell
Copy-Item main.pdf overlay_algebra_causal_composition_of_operational_behaviors_ali_uyar.pdf -Force
```
- To switch back to anonymous submission mode, replace `\usepackage[preprint]{tmlr}` with `\usepackage{tmlr}` in `main.tex` and restore an anonymous author block.
- Figures are generated from `scripts/generate_figures.py`.
