# Appendix Note: Citation (`C`) Scope And Artifact Status

This note is intended for appendix or supplementary use only.
It should not be promoted into a main-result claim.

## Recommended appendix wording

The citation overlay (`C`) remained part of the overall project scope, and direct full-scale `C` training completed successfully at `runs/oa_m4_full_c_17/`. However, the reduced-scope one-GPU paper path did not carry `C` through the full downstream evidence chain. The only full-scale `C` residualization attempt currently present in the repo is the archived off-path run `runs/oa_m4_full_residualize_c_17_archived_manual_stop_off_path_2026-03-26/`, which stopped during preparation before any evaluation units completed. As a result, the current artifact set does not support a clean full-scale `C` recovery row, and the main paper therefore limits its supported empirical claims to the corrected `J/Q` single-overlay results.

## What can be said safely

- `C` was not dropped from scope because it failed outright; direct full-scale `C` training exists.
- The default paper path prioritized the cleaner `J/Q` evidence under the actual runtime budget.
- The current artifact set does not justify a full-scale positive or negative `C` recovery claim.

## What should not be said

- Do not imply that full-scale `C` recovery succeeded.
- Do not imply that full-scale `C` recovery failed.
- Do not imply that `C` was never run at full scale.

## Pointers

- direct full-scale `C` train artifact: `runs/oa_m4_full_c_17/metrics.json`
- archived downstream `C` residualization attempt: `runs/oa_m4_full_residualize_c_17_archived_manual_stop_off_path_2026-03-26/heartbeat.json`
- claim status source of truth: `docs/CLAIMS_MATRIX.md`
