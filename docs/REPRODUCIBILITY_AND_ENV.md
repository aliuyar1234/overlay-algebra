# REPRODUCIBILITY_AND_ENV.md

## 1. Expected environment
Target platform:
- Linux workstation
- single GPU with ~96 GB VRAM preferred
- enough local storage for raw data, processed data, checkpoints, and predictions

Initial software expectations:
- Python 3.11 preferred
- PyTorch + CUDA stack compatible with the target GPU
- Hugging Face Transformers / PEFT / Datasets family
- NumPy / SciPy
- `spaCy` with `spacy.blank("en")` + sentencizer; the current M1 lock is `spaCy 3.8.13`
- pytest for tests

Important:
- use `uv` for environment management
- `uv.lock` now exists for the current workspace state
- the sentence splitter version is pinned in `configs/data/squad_jcq.yaml` and recorded in generated dataset manifests

## 2. Hardware assumptions
Primary target:
- one workstation-class GPU (planned target: RTX PRO 6000 class, 96 GB VRAM)

Fallback:
- QLoRA / lower batch size / smaller max length are acceptable if recorded
- if fallback materially changes the run path, treat those as separate experiment families

## 2.1 Locked model and implementation defaults
- base model for v1: `Qwen2.5-7B-Instruct`
- use the same backbone from the first smoke path onward; do not introduce a smaller dev backbone for M0-M2
- for smoke tests, reduce dataset size, batch size, sequence length, and training steps instead of changing models
- initial LoRA target modules: `o_proj` and `down_proj` only
- preferred training stack: `transformers` + `peft` + a simple custom training loop
- evaluation decoding default: greedy with `temperature=0`
- config system: plain YAML configs with a small typed loader and `argparse` CLIs
- do not use Hydra
- `reference_impl/` is the locked behavior oracle; production code lives separately in `src/`
- do not introduce a second dataset before the main controlled path is working

## 3. Seed policy
Minimum:
- save one explicit seed per run

Preferred:
- set and log seeds for Python, NumPy, and Torch
- enable deterministic behavior where practical
- if exact determinism is not possible, record the nondeterministic components

Suggested seed set for final confirmation runs:
- `{17, 23, 42}`

If compute is too tight for multiple seeds:
- run 1 seed,
- add bootstrap confidence intervals over examples,
- state the limitation honestly.

## 4. Config policy
- all training, evaluation, tuning, and analysis runs must be config-driven
- use plain YAML configs plus explicit typed loading; avoid config metaprogramming
- no hidden hyperparameters in scripts
- every run saves the exact resolved config it used
- config changes after a run create a new run ID; never overwrite

## 5. Dataset location assumptions
Suggested local paths:
- raw: `data/raw/squad_v1_1/`
- processed: `data/processed/squad_jcq_v1/`
- transfer slice (optional): `data/processed/transfer_jcq_v1/`

The processed dataset must save:
- version string
- raw source identity
- sentence splitter identity/version
- split manifest
- record count summary
- content hash if feasible

Locked split policy:
- start from official SQuAD 1.1 train/dev
- create train/val as a `90/10` split within official train by article/title only
- use official dev (after filtering) as test
- freeze sentence IDs once filtered dataset artifacts are created; do not change the splitter mid-project without creating a new dataset version

Current M1 validation path:
- bundled SQuAD-format fixture raw files live under `tests/fixtures/raw/squad_v1_1/`
- the validated processed output path is `data/processed/squad_jcq_v1_fixture/`
- this is a reproducibility fixture shard, not yet the full raw SQuAD-derived core dataset

## 6. Artifact naming conventions
Suggested run ID pattern:
- `oa_{milestone}_{model}_{condition}_{seed}_{timestamp}`

Suggested artifact tree:
- `runs/<run_id>/resolved_config.yaml`
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/predictions.jsonl`
- `runs/<run_id>/stdout.log`
- `runs/<run_id>/environment.txt`
- `runs/<run_id>/tuning_manifest.json`
- `runs/<run_id>/adapter/` or `runs/<run_id>/delta_manifest/`

Current important run-state note:
- the first full scaffold artifact was archived as `runs/oa_m4_full_scaffold_17_archived_tainted_pre_guard_fix/`
- the canonical rerun target for the clean scaffold remains `runs/oa_m4_full_scaffold_17/`

## 6.1 Current runtime controls
Training runs now expose:
- `heartbeat.json` for progress and ETA
- `checkpoints/latest.json` plus checkpoint directories for resume state
- `control/pause.request` for cooperative pause

Supported commands:
- start clean run: `python -m overlay_algebra.train.fit --config <config> --json`
- resume latest checkpoint for that config: `python -m overlay_algebra.train.fit --config <config> --resume-latest --json`
- resume explicit checkpoint: `python -m overlay_algebra.train.fit --config <config> --resume-checkpoint <path> --json`

The same control family now exists for the long-running analysis paths:
- `python -m overlay_algebra.analysis.compose --config <config> --json`
- `python -m overlay_algebra.analysis.compose --config <config> --resume-latest --json`
- `python -m overlay_algebra.analysis.compose --config <config> --resume-checkpoint <path> --json`
- `python -m overlay_algebra.analysis.ablate --config <config> --json`
- `python -m overlay_algebra.analysis.ablate --config <config> --resume-latest --json`
- `python -m overlay_algebra.analysis.ablate --config <config> --resume-checkpoint <path> --json`
- `python -m overlay_algebra.eval.bundle --config <config> --json`
- `python -m overlay_algebra.eval.bundle --config <config> --resume-latest --json`
- `python -m overlay_algebra.eval.bundle --config <config> --resume-checkpoint <path> --json`
- `python -m overlay_algebra.analysis.decode_audit --config <config> --json`
- `python -m overlay_algebra.analysis.decode_audit --config <config> --resume-latest --json`
- `python -m overlay_algebra.analysis.decode_audit --config <config> --resume-checkpoint <path> --json`

Composition runs save stable chosen adapters under:
- `runs/<run_id>/compositions/<condition>/soar/adapter`
- `runs/<run_id>/compositions/<condition>/merge/adapter`

Ablation runs save task outputs under:
- `runs/<run_id>/tasks/<task_key>/`

## 6.2 Current full-study M4 runtime defaults
Current full `M4` train configs use:
- `batch_size=8`
- `gradient_accumulation_steps=4`
- effective batch unchanged relative to the earlier `4 x 8` setup
- `checkpoint_interval_optimizer_steps=500`
- dataset-versioned tokenized feature caches under `data/processed/squad_jcq_v1/_feature_cache/`

Current full `M4` eval and residualization configs use:
- greedy decoding
- `eval_batch_size=4`
- unchanged `max_length=768`
- unchanged condition-specific `max_new_tokens`

Current runtime-safe eval implementation defaults now also include:
- dataset-versioned eval feature caches under `data/processed/squad_jcq_v1/_feature_cache/`
- length-aware eval ordering by cached prompt/target token lengths
- cache metadata saved into eval/residualization manifests
- optional load-once bundled eval sessions via `overlay_algebra.eval.bundle`
- optional decode-budget audits via `overlay_algebra.analysis.decode_audit`

Current code-ready `M5` defaults use:
- the same `batch_size=8`, `gradient_accumulation_steps=4`, and checkpoint cadence as `M4` for direct upper-bound training
- pair/triple composition validation on the full core validation split
- `alpha_grid = {0.5, 0.75, 1.0, 1.25}`
- merge simplex step `0.25`
- `eval_batch_size=4`
- `max_new_tokens=64` for `JC`
- `max_new_tokens=96` for quote-containing pair/triple conditions (`JQ`, `CQ`, `JCQ`)

## 7. What must be saved for later inspection
For any reported experiment:
- resolved config
- seed
- adapter checkpoint or reproducible pointer to it
- dense delta manifest if residualization/composition is involved
- validation tuning records
- final metrics
- predictions
- parser version / metric version
- dataset version
- wall-clock and hardware summary if available

## 8. Minimal rerun instructions
A minimally reproducible result means a new session can:
1. rebuild the processed dataset from raw + config,
2. train one adapter from a saved config,
3. evaluate it with deterministic parsers,
4. reproduce the saved metrics within numerical tolerance.

For the full paper path, minimally reproducible also means:
- rebuild one residual overlay,
- rerun one pair composition,
- regenerate the corresponding metrics table row.

Current clean rerun path for the next session:
1. run `python -m overlay_algebra.train.fit --config configs/train/scaffold.yaml --json`
2. verify the new `runs/oa_m4_full_scaffold_17/metrics.json` has finite training summaries
3. then launch `configs/train/full_J.yaml`, `configs/train/full_C.yaml`, and `configs/train/full_Q.yaml`
4. residualize with `configs/analysis/full_residuals_*.yaml`
5. run `python -m overlay_algebra.analysis.decode_audit --config configs/analysis/decode_audit_q.yaml --json` before trusting the final full `Q` decode budget
6. evaluate with either `configs/eval/full_*.yaml` or the new bundled configs `configs/eval/full_bundle_{J,C,Q}.yaml`
7. build reports with `configs/reports/full_single_overlay.yaml` and `configs/reports/c1_c4_c5.yaml`

Once `M4` is clean, the already-implemented next execution path is:
1. train `configs/train/full_JC.yaml`, `full_JQ.yaml`, `full_CQ.yaml`, `full_JCQ.yaml`
2. tune compositions with `configs/analysis/pairs.yaml` and `configs/analysis/triple.yaml`
3. evaluate prompt-only / merge / SOAR / direct with `configs/eval/full_*_{JC,JQ,CQ,JCQ}.yaml`
4. build `configs/reports/full_pair_composition.yaml` and `configs/reports/full_triple_composition.yaml`
5. run `configs/analysis/ablations.yaml`
6. build `configs/reports/all_claims.yaml`
7. optionally run `configs/eval/transfer.yaml` and export `configs/reports/paper.yaml`

## 9. Experiment tracking expectations
You do not need a heavyweight tracking platform at first.
But every run must leave behind:
- a machine-readable metrics file,
- a manifest of inputs,
- enough information to trace which claim it supports.

If pilot results are weak:
- keep the protocol strict and preserve the same task/evaluation story
- debug in this order: compiler/parsers and dataset filters; dense-delta subtraction coefficient and residual rank; target modules; then already-approved repair paths
- do not widen scope early just to chase a stronger positive result

## 10. Reproducibility-specific failure conditions
A result is not reproducible enough for the paper if:
- the processed dataset cannot be regenerated,
- parser versions are unknown,
- coefficient tuning choices were not saved,
- claim tables rely on numbers that cannot be traced to artifacts.
