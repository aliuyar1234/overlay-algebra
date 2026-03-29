# FORMALISM_PSEUDOCODE_AND_PROOF_OBLIGATIONS.md

This file preserves the exact technical content of the method.
It includes:
- notation,
- equations,
- algorithm blocks,
- proof sketches where appropriate,
- explicit proof obligations,
- clear separation between proven facts and conjectures.

## 1. Notation

### Data
Let the processed dataset be
- `D = {(x_n, a_n, s_n, u_n)}_{n=1}^N`

where:
- `x_n = (q_n, c_n^L)` is the question and labeled context string,
- `a_n` is the gold answer text,
- `s_n ∈ {1, 2, ..., S_n}` is the 1-based support sentence index,
- `u_n` is the exact support sentence text.

Let `Ω = {J, C, Q}` be the set of operational overlays.

For any `O ⊆ Ω`, define a deterministic target compiler
- `T_O(x_n, a_n, s_n, u_n) -> t_{n,O}`.

### Model and adapters
Let the frozen base model be `M0` with base weights `W`.
A LoRA adapter `A` modifies a set of modules `m ∈ M` with low-rank updates.

For module `m`:
- base weight `W_m ∈ R^{d_out(m) x d_in(m)}`
- LoRA rank `r_m`
- factors `A_m ∈ R^{r_m x d_in(m)}`, `B_m ∈ R^{d_out(m) x r_m}`
- scalar `alpha_m`

The effective dense update for adapter `A` at module `m` is
- `Δ^A_m = (alpha_m / r_m) * B_m A_m`

This dense matrix is the object used in SOAR.

### Trained adapters
- `S`: semantic scaffold trained on `O = ∅`
- `F_J`, `F_C`, `F_Q`: direct single-overlay adapters
- `F_O` for `|O| >= 2`: direct multi-overlay upper bounds

## 2. Target compiler
The compiler `T_O` is deterministic and grammar-locked by `METHOD_SPEC.md`.

Examples:
- `T_∅ = a`
- define `sid(s) = "S" + str(s)`
- define `quote_literal(u) = json.dumps(u, ensure_ascii=False)`
- define `canonical_json(obj) = json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))`
- `T_J = canonical_json({"answer": a})`
- `T_C = "ANSWER: {a}\nSUPPORT: [" + sid(s) + "]"`
- `T_Q = "ANSWER: {a}\nQUOTE: " + quote_literal(u)`

For combined overlays, the grammar is the exact composition locked in `METHOD_SPEC.md`.

## 3. Training objective
For each overlay set `O`, train the corresponding adapter by minimizing the next-token negative log-likelihood over compiled targets:
- `L_O(θ_O) = - Σ_n log p_{M0 ⊕ A_O}( t_{n,O} | prompt_O(x_n) )`

where `⊕` denotes applying the adapter update to the frozen base model.

No special reward model or judge is used in core training.

## 4. SOAR residualization
For a single overlay `k ∈ Ω` and module `m`, define the raw residual:
- `Γ_k,m(β_k) = Δ^{F_k}_m - β_k Δ^S_m`

The compressed operational residual is:
- `Δ^{R_k}_m = P_{r_res}( Γ_k,m(β_k) )`

where `P_r(X)` is the best rank-`r` approximation to `X` under Frobenius norm, given by truncated SVD:
- if `X = U Σ V^T`, then `P_r(X) = U_r Σ_r V_r^T`.

### Composition
For any active overlay set `O ⊆ Ω`:
- `Δ^{SOAR(O)}_m = Δ^S_m + Σ_{k ∈ O} α_k Δ^{R_k}_m`

The composed model is:
- `M_SOAR(O) = M0 ⊕ {Δ^{SOAR(O)}_m}_{m ∈ M}`

## 5. Optimization / tuning rules
Beta selection:
- choose `β_k ∈ BETA_GRID`
- objective on validation:
  - maximize primary overlay metric for overlay `k`
  - subject to `semantic_F1 >= semantic_F1(S) - 3.0`

Alpha selection for composition:
- choose `α_k ∈ ALPHA_GRID`
- objective on validation:
  - maximize the mean active overlay primary score
  - subject to `semantic_F1 >= semantic_F1(S) - 3.0`

For tuned whole-adapter merge:
- `Δ^{MERGE(O)}_m = Σ_{k∈O} w_k Δ^{F_k}_m`
- `w_k >= 0`, `Σ w_k = 1`

## 6. Pseudocode

### Algorithm 1: Build processed dataset
```text
INPUT: raw SQuAD examples
OUTPUT: train/val/test processed JSONL files

for each raw example e:
    sentence_split context into sentences s_1, ..., s_T
    map gold answer span to sentence boundaries
    if answer span is not fully inside exactly one sentence:
        continue
    support_idx <- sentence containing answer span (1-based)
    support_sentence <- exact sentence text
    labeled_context <- render [S1] ... [ST]
    for each overlay set O in {plain, J, C, Q, JC, JQ, CQ, JCQ}:
        target_O <- compile_target(O, answer, support_idx, support_sentence)
        # compile_target must use canonical_json for JSON conditions
        # and quote_literal for non-JSON quote conditions
        if parse(target_O) fails its own strict parser:
            reject example
    save processed record
split by article/title: train, val, test
run split-audit and round-trip parser tests
```

### Algorithm 2: Train adapters
```text
INPUT: processed dataset, base model M0
OUTPUT: trained adapters S, F_J, F_C, F_Q, F_JC, F_JQ, F_CQ, F_JCQ

train S on prompt_plain -> target_plain
for k in {J, C, Q}:
    train F_k on prompt_k -> target_k
for O in {JC, JQ, CQ, JCQ}:
    train F_O on prompt_O -> target_O
save adapters, configs, and metrics
```

### Algorithm 3: Residualize single-overlay adapters
```text
INPUT: S, F_J, F_C, F_Q, validation set
OUTPUT: residual overlays R_J, R_C, R_Q and chosen betas

ΔS <- dense_delta_map(S)
for k in {J, C, Q}:
    ΔFk <- dense_delta_map(F_k)
    best <- None
    for beta in BETA_GRID:
        Γk <- ΔFk - beta * ΔS
        Rk(beta) <- TSVD_project(Γk, rank=r_res)
        eval model M0 ⊕ (ΔS + Rk(beta)) on validation using prompt_k
        score <- primary_overlay_metric
        retain <- semantic_F1 >= semantic_F1(S) - 3.0
        update best using lexicographic objective:
            1) prefer retain=True
            2) maximize score
            3) tie-break beta closest to 1.0
    save chosen beta and R_k
```

### Algorithm 4: Compose residuals for unseen combinations
```text
INPUT: ΔS, residuals R_J/R_C/R_Q, validation set, target overlay set O
OUTPUT: best composed delta and test metrics

best <- None
for alpha tuple in ALPHA_GRID^{|O|}:
    Δcand <- ΔS + Σ_{k in O} alpha_k * ΔR_k
    eval M0 ⊕ Δcand on validation using prompt_O
    overlay_score <- mean(primary_overlay_metric_k for k in O)
    retain <- semantic_F1 >= semantic_F1(S) - 3.0
    update best using lexicographic objective:
        1) prefer retain=True
        2) maximize overlay_score
        3) tie-break alpha tuple closest to all ones
freeze best alpha tuple
evaluate on test set and save outputs
```

## 7. Proven facts, standard facts, and conjectures

### Proposition P1 (factor gauge non-uniqueness)
For any invertible matrix `Q ∈ R^{r x r}`,
- `(B Q)(Q^{-1} A) = B A`.

Therefore many different factor pairs represent the same dense delta.
Arithmetic on raw factors is representation-dependent unless a canonical gauge is fixed.

**Proof sketch:** direct multiplication.  
This is enough to justify why the core method operates on `Δ = scale * B A`, not on raw factor coordinates.

### Proposition P2 (best rank-r projection)
Let `X = U Σ V^T` be the SVD of `X`.
Then `U_r Σ_r V_r^T` is the minimizer of `||X - Y||_F` over all matrices `Y` with `rank(Y) <= r`.

**Status:** standard linear algebra fact; cite a standard source in the final paper.

### Algebraic observation P3
If for some overlay `k`, `Δ^{F_k}_m = β_k Δ^S_m + O_{k,m}` and `rank(O_{k,m}) <= r_res` for every module `m`, then residualization with the matching `β_k` and `r_res` recovers `O_{k,m}` exactly.

**Status:** direct algebraic observation, not a theorem about real models.

### Conjecture G1 (empirical, unproven)
For some operational behaviors in the chosen setting, `O_k` is sufficiently low-rank and sufficiently separable from shared semantics that `R_k` is reusable across unseen combinations.

**Status:** unproven; this is the empirical question of the paper.

## 8. Proof obligations / method invariants
These are not all mathematical proofs. They are obligations that must hold for the science to remain coherent.

| ID | Obligation | How to check | Status at build-pack creation |
|---|---|---|---|
| O1 | All compared adapters target the same module names and shapes. | state-dict / config audit | open |
| O2 | Effective dense delta extraction reproduces the intended adapter update exactly up to numerical tolerance. | reference test + numerical check | open |
| O3 | TSVD refactorization reproduces the projected dense delta within tolerance. | golden test | open |
| O4 | The target compiler is deterministic and every gold target round-trips through the strict parser. | unit tests | open |
| O5 | Train/val/test splits are title-leak free. | split audit report | open |
| O6 | Beta and alpha tuning use validation only. | run manifest audit | open |
| O7 | Prompt-only, SOAR, merge, and direct upper bounds all use the same prompt for a given overlay set. | eval config audit | open |
| O8 | Claim-support scripts consume saved metrics, not hand-copied numbers. | claim-check pipeline | open |
| O9 | Target compiler uses the locked canonical JSON serializer and JSON-string-literal quote escaping. | golden tests | open |

## 9. What must stay aligned between math and code
- `Δ^A_m` in the math must map one-to-one to a dense update tensor in code.
- `support_sent_idx` is 1-based in both code and docs.
- `BETA_GRID`, `ALPHA_GRID`, and the semantic guardrail must match the configs used in runs.
- strict JSON schema validity must mean **exact required key set, exact types, no extras**; key order is fixed for compiled targets but not required for evaluation validity.
- non-JSON quote parsing must decode a JSON string literal after the `QUOTE: ` prefix.
- if the code chooses a different residual factorization convention, the effective dense delta must still match the formal object.
