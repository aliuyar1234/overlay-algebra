# METHOD_SPEC.md

This file is the exact method source of truth.
If code, configs, or experiments disagree with this file, the implementation must stop and record the contradiction.

## 1. Problem setup
We study a frozen base language model `M0` adapted with LoRA-style low-rank updates.
All experiments share:
- the same underlying QA semantics,
- the same sentence-labeled context format,
- the same evaluation parsers,
- different **operational output requirements**.

### Overlay vocabulary
- `J`: valid JSON with exact keys
- `C`: support sentence citation
- `Q`: exact support quote

Overlay sets are subsets of `{J, C, Q}`.

## 2. Requirement IDs
| ID | Requirement | Lock status |
|---|---|---|
| R1 | v1 scope is exactly J/C/Q overlays on answerable QA; no abstention. | locked |
| R2 | Every processed example must have one unambiguous support sentence containing the gold answer span. | locked |
| R3 | Contexts must be sentence-labeled `[S1] ... [S2] ...` and the labeled context string is shared across all overlay conditions. | locked |
| R4 | A deterministic target compiler must produce targets for plain, J, C, Q, and all pair/triple combinations. | locked |
| R5 | Train one semantic scaffold adapter `S` on plain-answer targets only. | locked |
| R6 | Train one direct single-overlay adapter each: `F_J`, `F_C`, `F_Q`. | locked |
| R7 | Train direct upper-bound adapters for `F_JC`, `F_JQ`, `F_CQ`, and `F_JCQ`. | locked |
| R8 | Convert adapters to effective dense deltas per module; do not use raw factor arithmetic as the primary method. | locked |
| R9 | Residualize each single-overlay adapter against `S`: `R_k = TSVD(ΔF_k - β_k ΔS)`. | locked |
| R10 | Compose unseen overlay sets with `ΔS + Σ α_k ΔR_k`, where `α_k` is tuned only on validation data. | locked |
| R11 | Baselines must include prompt-only control on `S` and tuned whole-adapter merge. | locked |
| R12 | Core evaluation must be deterministic and parser-based; no LLM judge in the critical claim path. | locked |
| R13 | Save configs, seeds, predictions, metrics, and tuning records for every reported run. | locked |
| R14 | Optional late-stage: add a small real-document transfer slice. | optional late-stage |

## 3. Data objects

### 3.1 Raw core data
Use SQuAD 1.1 train/dev as the raw source.
Only answerable items are used in v1.

### 3.2 Processed example schema
Each processed record must include:
```json
{
  "example_id": "string",
  "title": "string",
  "question": "string",
  "context": "raw paragraph text",
  "context_sentences": ["sentence 1", "sentence 2", "..."],
  "labeled_context": "[S1] sentence 1\n[S2] sentence 2\n...",
  "answer_text": "gold answer",
  "answer_start": 123,
  "support_sent_idx": 2,
  "support_sentence": "sentence containing the answer span",
  "targets": {
    "plain": "...",
    "J": "...",
    "C": "...",
    "Q": "...",
    "JC": "...",
    "JQ": "...",
    "CQ": "...",
    "JCQ": "..."
  }
}
```

### 3.3 Filtering rules
Keep an example only if all are true:
1. the answer span maps fully inside exactly one sentence boundary,
2. the support sentence text is non-empty after normalization,
3. sentence labels can be rendered without ambiguity,
4. target compilation succeeds for all overlay conditions.

Drop examples if:
- sentence splitting is unstable or empty,
- the answer span crosses a sentence boundary,
- JSON target serialization fails,
- parser round-trip tests fail.

## 4. Prompt protocol (locked logical form)
The **logical prompt content** must be constant except for the required output format.

### 4.1 Canonical user prompt
```text
Use only the provided context. Context sentences are labeled [S1], [S2], ...
Do not use outside knowledge.
Required output format: {FORMAT_SPEC}

Question:
{QUESTION}

Context:
{LABELED_CONTEXT}

Response:
```

The exact chat wrapper may depend on model family, but the semantic content above must remain unchanged.

### 4.2 Format specs (locked)
- `plain`: `plain answer text only`
- `J`: `valid JSON object with exactly one key "answer"`
- `C`: `two lines: ANSWER: <text> and SUPPORT: [S#]`
- `Q`: `two lines: ANSWER: <text> and QUOTE: <JSON string literal of exact support sentence text>`
- `JC`: `valid JSON object with keys "answer" and "support", where support is a one-item list like ["S2"]`
- `JQ`: `valid JSON object with keys "answer" and "quote"`
- `CQ`: `three lines: ANSWER: <text>, SUPPORT: [S#], QUOTE: <JSON string literal of exact support sentence text>`
- `JCQ`: `valid JSON object with keys "answer", "support", and "quote"`

## 5. Target compiler (locked grammar)

### 5.1 Intuition
The only thing that should change across overlay conditions is the output contract, not the underlying question or answer semantics.

### 5.2 Precise mechanism
Let:
- `a` = answer text
- `s` = one-based support sentence index
- `u` = exact support sentence text

Targets are:

```text
plain = a

C =
ANSWER: {a}
SUPPORT: [S{s}]

Q =
ANSWER: {a}
QUOTE: {quote_literal(u)}

CQ =
ANSWER: {a}
SUPPORT: [S{s}]
QUOTE: {quote_literal(u)}
```

Define two deterministic helper functions first:

```text
sid(s) = "S" + str(s)
quote_literal(u) = json.dumps(u, ensure_ascii=False)
canonical_json(obj) = json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
```

Notes:
- `quote_literal(u)` returns a JSON string literal, including outer double quotes and any required escapes.
- `canonical_json(obj)` fixes whitespace and key order for **compiled training targets**. Do **not** use the Python default spacing if you want deterministic training targets.
- Key order is locked by object construction order and `sort_keys=False` for compiler outputs.

JSON variants must be serialized with `canonical_json`:
```json
J   = canonical_json({"answer": a})
JC  = canonical_json({"answer": a, "support": [sid(s)]})
JQ  = canonical_json({"answer": a, "quote": u})
JCQ = canonical_json({"answer": a, "support": [sid(s)], "quote": u})
```

### 5.3 Implementation hooks
- Use 1-based sentence IDs everywhere.
- `support` is always a length-1 list in JSON variants.
- JSON keys are exact and no extras are allowed in strict schema checks.
- JSON output must use the locked canonical serializer with no spaces after commas/colons.
- For non-JSON quote outputs, render `QUOTE: {quote_literal(u)}` rather than raw string interpolation; this is how embedded quotes and backslashes stay parseable.
- The locked key order for **compiled targets** is:
  - `J`: `answer`
  - `JC`: `answer`, `support`
  - `JQ`: `answer`, `quote`
  - `JCQ`: `answer`, `support`, `quote`

### 5.4 Locked JSON schemas
Strict schema validity means the parsed JSON object must match exactly one of these schemas:

- `J`: `{"answer": <string>}`
- `JC`: `{"answer": <string>, "support": [<sid>]}` where `<sid>` matches `^S[1-9][0-9]*$`
- `JQ`: `{"answer": <string>, "quote": <string>}`
- `JCQ`: `{"answer": <string>, "support": [<sid>], "quote": <string>}`

Additional rules:
- no extra keys
- all required keys must be present
- key order does **not** affect schema validity at evaluation time
- `support` must be a list of length exactly 1
- `answer`, `quote`, and `support[0]` must all be strings

## 6. Adapter training

### 6.1 Intuition
`S` should capture task semantics. `F_k` should capture semantics plus one overlay behavior. Pair/triple models are upper bounds, not the main method.

### 6.2 Precise mechanism
Train the following adapters from the same frozen base `M0`:
- `S` on `plain`
- `F_J`, `F_C`, `F_Q`
- `F_JC`, `F_JQ`, `F_CQ`, `F_JCQ`

All runs use the same processed examples and the same prompt protocol; only `FORMAT_SPEC` and the target string change.

### 6.3 Implementation hooks
Initial defaults (not core science; may be tuned if needed, but changes must be logged):
- LoRA rank: 16
- LoRA alpha: 16
- dropout: 0.05
- target modules (initial): `o_proj` and `down_proj`
- optimizer: AdamW
- learning rate: 1e-4
- max length: 768
- epochs: 1 for full runs, shorter for smoke/pilot
- precision: bf16 if stable; else QLoRA fallback

Do not change target modules mid-study without updating `STATUS.md`.

## 7. Effective dense delta extraction

### 7.1 Intuition
LoRA factor matrices are not uniquely identifiable.
The stable algebra object is the **effective dense update** each adapter adds to each module.

### 7.2 Precise mechanism
For each adapted module `m` with LoRA factors `(A_m, B_m, alpha_m, r_m)`:
- `A_m` shape: `[r_m, d_in]`
- `B_m` shape: `[d_out, r_m]`

The effective dense update is:
- `Δ_m = (alpha_m / r_m) * B_m @ A_m`

Collect `Δ_m` for every adapted module into a delta map.

### 7.3 Implementation hooks
- Use adapter weights in eval mode.
- All adapters compared in subtraction/composition must target the same module names.
- If quantization is used, the delta map still comes from adapter parameters only.
- Save dense delta maps or a reproducible manifest that regenerates them exactly.

## 8. Residualization (SOAR core)

### 8.1 Intuition
A single-overlay adapter contains shared semantics plus overlay-specific behavior.
Subtract the shared scaffold; compress what remains.

### 8.2 Precise mechanism
For overlay `k in {J,C,Q}` and module `m`:
- semantic delta: `ΔS_m`
- full single-overlay delta: `ΔF_k,m`

Compute the raw residual:
- `Γ_k,m = ΔF_k,m - β_k * ΔS_m`

Project back to low rank with truncated SVD:
- `ΔR_k,m = TSVD_rank(Γ_k,m, r_res)`

Default:
- choose `β_k` from a small validation-only grid, per overlay
- default `r_res = 16`
- if C1 fails badly, ablate `r_res = 32`

### 8.3 Implementation hooks
- Beta grid (locked default): `{0.5, 0.75, 1.0, 1.25}`
- Choose `β_k` using single-overlay validation only
- Validation objective: maximize overlay primary metric subject to `semantic_F1 >= semantic_F1(S) - 3.0`
- If no beta meets the guardrail, choose the beta with highest semantic F1 and record failure
- After TSVD, re-factorize to LoRA factors for deployable adapters; reference impl uses `alpha = rank` so scale = 1

## 9. Composition

### 9.1 Intuition
If residuals isolate operational behavior, they should be addable on top of `S`.

### 9.2 Precise mechanism
For any active overlay set `O ⊆ {J,C,Q}`:
- `ΔSOAR_O,m = ΔS_m + Σ_{k in O} α_k * ΔR_k,m`

Evaluate the composed model using the prompt with the matching combined `FORMAT_SPEC`.

### 9.3 Implementation hooks
- Alpha grid (locked default): `{0.5, 0.75, 1.0, 1.25}`
- Tune `α_k` on validation only for each composition
- Validation objective: maximize the mean active overlay primary score subject to `semantic_F1 >= semantic_F1(S) - 3.0`
- If several candidates tie, pick the one closest to all-ones coefficients
- If none satisfy the guardrail, choose the best semantic-F1 candidate and mark the composition as failed-under-guardrail

## 10. Baselines and upper bounds

### 10.1 Prompt-only baseline
Model: `M0 + ΔS`
Inference prompt: same combined `FORMAT_SPEC` used for the target composition.

### 10.2 Tuned whole-adapter merge baseline
For active overlay set `O`, define:
- `ΔMERGE_O = Σ_{k in O} w_k * ΔF_k`
subject to:
- `w_k >= 0`
- `Σ w_k = 1`

Tune `w_k` on validation with the same objective and guardrail used for SOAR.

### 10.3 Direct upper bounds
Use the directly trained multi-overlay adapters:
- `F_JC`, `F_JQ`, `F_CQ`, `F_JCQ`

These are **upper bounds / ceilings**, not fairness baselines that SOAR must always beat.

## 11. Deterministic evaluation

### 11.1 Primary metrics
- semantic answer EM / F1 (after parsing)
- JSON valid rate
- JSON strict-schema valid rate
- citation exact sentence-ID accuracy
- quote normalized exact support-sentence match
- quote token F1 vs support sentence

### 11.2 Metric parsing rules
- invalid JSON => JSON metrics = 0 and answer extraction only if explicitly allowed by eval config; default is no salvage for strict metrics
- missing or malformed support field => citation accuracy = 0
- quote lines in non-JSON outputs must be parsed as JSON string literals after stripping the `QUOTE: ` prefix
- quote mismatch => quote exact = 0; quote F1 still computed if parsable
- semantic answer must be extracted via the parser, not by ad hoc regex outside the evaluator

## 12. Edge cases and failure modes
- sentence splitter changes => dataset version changes; old/new results not comparable
- repeated answer strings across sentences => safe if gold span maps to one sentence; citation still uses gold support sentence
- `F_k` much larger/smaller in norm than `S` => beta tuning becomes important
- residual rank too low => overlay recovery fails
- factorization mismatch => composed adapters do not reproduce intended dense deltas
- prompt-only unexpectedly strong => paper framing must acknowledge prompt contribution
- pair prompt semantics leak into comparison => acceptable only if identical across SOAR, merge, and prompt-only baselines

## 13. Incorrect shortcuts Codex must avoid
- subtracting raw `A` / `B` factor matrices directly
- using the official SQuAD split without a validation partition
- letting the sentence splitter run differently across sessions
- tuning alphas/betas on the test set
- evaluating formatted outputs with plain-answer scripts
- using direct pair/triple adapters during residual training
- changing the grammar to make parsing easier after seeing failures
