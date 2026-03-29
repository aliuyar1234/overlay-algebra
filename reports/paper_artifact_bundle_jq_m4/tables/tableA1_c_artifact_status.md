# Appendix Table A1: Full-Scale Citation (`C`) Artifact Status

Source files:
- `runs/oa_m4_full_c_17/metrics.json`
- `runs/oa_m4_full_residualize_c_17_archived_manual_stop_off_path_2026-03-26/heartbeat.json`
- `runs/oa_m4_full_residualize_c_17_archived_manual_stop_off_path_2026-03-26/state.json`

| Stage | Status | Evidence | Safe manuscript interpretation |
|---|---|---|---|
| direct full-scale `C` training | completed | `runs/oa_m4_full_c_17/metrics.json` | full-scale `C` was run at the direct-training stage |
| full-scale `C` residualization | not completed cleanly | archived off-path run stopped at `0 / 5` evaluation units | no full-scale `C` residual result exists |
| full-scale `C` prompt/direct/SOAR eval bundle | absent | no clean `runs/oa_m4_full_{prompt_only,direct,soar}_c_eval_17/` artifacts | no full-scale `C` test metrics should be reported |
| full-scale `C` recovery report | absent | no `reports/full_single_overlay_recovery_c/` | no full-scale `C` recovery row belongs in the main table |

Caption:

> Appendix-only status table for the citation overlay. Direct full-scale `C` training exists, but the downstream full-scale evidence chain was not completed cleanly enough to support a claim-safe results row.
