# Table 2: Claim Scope For The Submission Draft

Source files:
- `claim_checks_c1_c4_c5/metrics.json`
- `status_md`

| Claim ID | Current status | Use in paper | Allowed wording |
|---|---|---|---|
| `C1` | supported | main empirical claim | At least two single overlays (`J`, `Q`) are recoverable through residualization with negligible semantic degradation. |
| `C5` | supported | main systems / reproducibility claim | The evaluation pipeline is deterministic, parser-based, and auditable on the saved dataset split. |
| `C4` | partially supported | methods rationale only | Dense deltas are the principled object for residualization, but factor-space empirical superiority is not established. |
| `C2` | intended | exclude from main claims | No unseen-pair composition success claim in this paper. |
| `C3` | intended | exclude from main claims | No stable cross-overlay composability ordering claim in this paper. |
| `C6` | intended | exclude from main claims | Direct pair/triple upper-bound framing should remain future work or motivation only. |
| `C7` | intended | optional framing only | If mentioned, describe the protocol as narrow and controlled rather than broadly novel. |
| `C8` | intended (late-stage) | exclude from main claims | No transfer claim beyond the SQuAD-derived core dataset. |

Caption:

> Claim-status table for manuscript drafting. The current reduced-scope paper uses `C1` and `C5` as supported claims, keeps `C4` partial, and defers all composition and transfer claims.
