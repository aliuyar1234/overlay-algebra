# Table 1: Corrected Full-Scale Single-Overlay Recovery

Source files:
- `full_single_overlay/metrics.json`
- `single_overlay_recovery_j/metrics.json`
- `single_overlay_recovery_q/metrics.json`

| Overlay | Primary Metric | Prompt-only | Direct | SOAR | Prompt-only Answer F1 | Direct Answer F1 | SOAR Answer F1 | Recovery Ratio | Chosen Beta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J | `json_strict_schema_valid` | 0.0000 | 0.9986 | 0.9986 | 0.0000 | 0.8346 | 0.8344 | 1.0000 | 1.0 |
| Q | `quote_exact` | 0.0000 | 0.9413 | 0.9414 | 0.0000 | 0.8121 | 0.8122 | 1.0001 | 1.0 |

Caption:

> Corrected full-scale single-overlay recovery for the reduced-scope `J/Q` path. In both overlays, prompt-only on the scaffold collapses under the operational contract, while `SOAR` matches or slightly exceeds the direct single-overlay adapter on the active metric and remains effectively identical on answer F1.
