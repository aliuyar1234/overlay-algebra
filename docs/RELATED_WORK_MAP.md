# RELATED_WORK_MAP.md

This is a responsible positioning map, not a final bibliography.
Do not invent citation details from memory.
Before submission, replace placeholders with verified references.

## 1. Why this file exists
The novelty claim for this project depends on carefully separating it from nearby work.
This file tells Codex what categories matter and what not to overclaim.

## 2. Known nearby categories

### A. Adapter behavior reading / decomposition
Known targets to investigate and compare against:
- **W2T / weights-to-traits style work**
- **Gradient Atoms style work**

Why related:
- both try to read or decompose behavior-like structure from trained updates

Why we are different:
- this project does not only *read* behavior from weights
- it constructs **causal operational residuals** and evaluates add/remove/recompose behavior under a semantics-fixed protocol

Do not claim:
- first to study behavior structure in adapters
- first to use decomposition language around LoRA updates

### B. Adapter composition / merging / routing
Known targets:
- generic LoRA merging
- **LoDA**
- **MoLoRA**
- adaptive LoRA recycling / reuse papers
- unsafe composition / security papers such as **CoLoRA**

Why related:
- all are about combining or reusing adapter-like objects

Why we are different:
- we are not proposing a generic merge heuristic
- we isolate **operational** behaviors on one fixed semantic substrate
- direct pair/triple training is treated as an upper bound, not the main method

Do not claim:
- broad superiority over adapter merging in general
- safe composition guarantees
- production-grade routing system

### C. Operational output control / structured outputs / citation
Known targets:
- structured output / JSON generation work such as **JSONSchemaBench**
- citation / grounding quality papers
- instruction-following / format-control papers

Why related:
- these define the behaviors we instantiate as overlays

Why we are different:
- the point is not to win those tasks directly
- the point is to study whether such behaviors can be isolated and recomposed as residuals

Do not claim:
- new best structured-output benchmark result
- broad citation quality leadership

### D. Token-space or activation-space compositional control
Known targets:
- compositional steering tokens
- activation steering / feature steering
- other non-weight-space compositional control methods

Why related:
- they also address combining behaviors

Why we are different:
- our object is **weight-space residual overlays** on top of a task scaffold
- our evidence is adapter-space causal composition, not token-only steering

## 3. Novelty boundary to preserve
The project can responsibly claim:
- a semantics-fixed experimental protocol for operational overlay composition,
- a dense-delta residualization method for extracting operational overlays from single-behavior adapters,
- empirical evidence about partial composability and failure modes for J/C/Q overlays.

The project should **not** claim:
- universal behavior atoms,
- first decomposition of behavior from adapters,
- first adapter composition method,
- general safety guarantees,
- broad superiority over all merging/routing/steering methods.

## 4. Literature questions to resolve later
Before submission, verify:
1. exact bibliographic details for W2T-style work
2. exact bibliographic details for Gradient Atoms
3. closest work on compositional steering tokens
4. exact papers for LoDA / MoLoRA / CoLoRA
5. strongest structured-output/citation papers to cite for motivation
6. whether there is already a semantics-fixed composition protocol close enough to narrow the claim further

## 5. If literature review reveals a closer paper than expected
Do not hide it.
Update:
- `docs/RESEARCH_BRIEF.md`
- `docs/CLAIMS_MATRIX.md`
- `docs/PAPER_OUTLINE.md`

Then narrow the contribution claim honestly.
