# Experiment Pipeline Changes

Last updated: March 3, 2026

This file tracks changes made to the MVC experiment pipeline during this session.

## Summary

### Synthetic MVC evaluation
- Hardened `code/s2v_mvc/evaluate.py` so relative paths are resolved against the script directory.
- Added clearer missing-path errors for `save_dir` and `data_test`.
- Changed missing-checkpoint behavior from a crash to a clean skip for ranges with no `log-<min>-<max>.txt` or no usable checkpoint entry.
- Added support for paper-style generalization evaluation by letting one fixed training range model be tested across many dataset ranges via `MODEL_RANGE=<min>-<max>`.
- When `MODEL_RANGE` is set, output CSV names include both the training and test ranges to avoid collisions.
- Added `TEST_RANGES` filtering in `code/s2v_mvc/run_eval.sh` so evaluation can be restricted to one or more explicit buckets.
- Added `SYNTHETIC_MVC_RESULT_ROOT` to `code/s2v_mvc/run_eval.sh` so paper reproductions can write into a separate result directory.
- Added legacy `networkx` pickle normalization in `code/s2v_mvc/evaluate.py` so the downloaded paper graph pickles load correctly under the current environment.

### Synthetic MVC training
- Added resume support in `code/s2v_mvc/main.py` when `-load_model` is provided.
- Training now resumes from the parsed checkpoint iteration instead of restarting at iteration `0`.
- `code/s2v_mvc/main.py` can now load a fixed validation set from `-data_valid` instead of always generating validation graphs online.
- Added legacy `networkx` pickle normalization in `code/s2v_mvc/main.py` so the downloaded paper validation pickles can be rebuilt into fresh graphs.
- Updated `code/s2v_mvc/run_nstep_dqn.sh` to:
  - honor `G_TYPE` and `SYNTHETIC_MVC_TEST_DATA_DIR` environment overrides
  - honor `SYNTHETIC_MVC_VALID_DATA_DIR` to pick the matching fixed validation file per bucket
  - honor `SYNTHETIC_MVC_RESULT_ROOT` so paper reproductions do not reuse non-paper checkpoints
  - train all available synthetic `barabasi_albert` ranges discovered from `data/test_mvc_maxcut`
  - optionally restrict training to one paper-style range with `TRAIN_RANGE=<min>-<max>`
  - resume interrupted ranges from the latest `nrange_<min>_<max>_iter_*.model`
  - append to existing range logs when resuming
  - create `nrange_<min>_<max>.done` marker files after successful completion
- Reduced the default synthetic MVC training budget from `1,000,000` iterations to `100,000`.
- Made the synthetic training budget configurable with `SYNTHETIC_MVC_MAX_ITER`.

### Real-world MVC training
- Added resume support in `code/realworld_s2v_mvc/main.py` when `-load_model` is provided.
- Training now resumes from the parsed `iter_<n>.model` checkpoint iteration.
- Updated `code/realworld_s2v_mvc/run_nstep_dqn.sh` to:
  - resume from the latest `iter_*.model`
  - append to the existing training log when resuming
- Reduced the default real-world MVC training budget from `1,000,000` iterations to `100,000`.
- Made the real-world training budget configurable with `REALWORLD_MVC_MAX_ITER`.

### GPU device handling
- Updated these scripts to honor an environment override via `DEV_ID`:
  - `code/s2v_mvc/run_eval.sh`
  - `code/realworld_s2v_mvc/run_nstep_dqn.sh`
  - `code/realworld_s2v_mvc/eval.sh`
- Updated `run_mvc_slurm.sh` to export `DEV_ID=0` by default for the batch job.

### SLURM defaults
- Updated `run_mvc_slurm.sh` to export shorter default training budgets for batch runs:
  - `SYNTHETIC_MVC_MAX_ITER=100000`
  - `REALWORLD_MVC_MAX_ITER=100000`
- Added `MVC_PIPELINE_MODE=paper_synthetic` support in `run_mvc_slurm.sh` for the paper-style synthetic MVC workflow:
  - defaults to `PAPER_G_TYPE=barabasi_albert`
  - defaults to `PAPER_BUCKETS="15-20 40-50 50-100 100-200 400-500"`
  - uses the downloaded `data/paper_synthetic_data` test/validation files and `data/paper_synthetic_opt` optimal files by default
  - writes synthetic model outputs to a separate `results/paper-dqn-<g_type>` tree
  - runs synthetic training, S2V evaluation, and greedy baselines bucket-by-bucket
  - skips the real-world stages in that mode
- Added `MVC_PIPELINE_MODE=paper_generalization` support in `run_mvc_slurm.sh`:
  - defaults to `GENERALIZATION_TRAIN_RANGES="50-100"`
  - evaluates the fixed trained model(s) across all discovered paper test buckets unless `GENERALIZATION_TEST_RANGES` is set
  - runs the greedy baselines once on the same paper test buckets
- Added `MVC_PIPELINE_MODE=paper_full` support in `run_mvc_slurm.sh`:
  - defaults to `PAPER_GRAPH_FAMILIES="barabasi_albert erdos_renyi"`
  - runs the same-bucket synthetic study, the synthetic generalization study, and then the real-world MVC pipeline in one batch job
  - uses separate synthetic result roots per graph family under `results/paper-dqn-<g_type>`

### Greedy baseline plumbing
- Updated `code/greedy_mvc/run_greedy.sh` to honor:
  - `G_TYPE` / `G_TYPES` for graph-family selection
  - `TEST_RANGES` for bucket filtering
  - `SYNTHETIC_MVC_TEST_DATA_DIR` and `SYNTHETIC_MVC_OPT_DIR` for custom datasets / optimal-solution files
- Added legacy `networkx` pickle normalization in `code/greedy_mvc/evaluate_greedy.py` so the downloaded paper test pickles load correctly.

### Failure handling
- Updated the shell entrypoints to fail fast with `set -euo pipefail`:
  - `run_mvc_slurm.sh`
  - `code/s2v_mvc/run_nstep_dqn.sh`
  - `code/s2v_mvc/run_eval.sh`
  - `code/greedy_mvc/run_greedy.sh`
  - `code/realworld_s2v_mvc/run_nstep_dqn.sh`
  - `code/realworld_s2v_mvc/eval.sh`
  - `code/realworld_greedy_mvc/run_greedy.sh`
- Synthetic MVC training now treats a range as complete only when the expected final checkpoint file exists; stale `.done` markers from failed runs are ignored.

## Operational Notes
- Re-submitting `sbatch run_mvc_slurm.sh` starts a new job from the top of the script, but the pipeline is now substantially more restart-friendly.
- Completed synthetic training ranges are skipped automatically.
- Interrupted synthetic and real-world training runs resume from the newest available checkpoint.
- Synthetic evaluation no longer aborts the whole job when a trained checkpoint is missing for one range.
