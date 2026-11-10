# Running the FireWorkers experiments on Kaggle (GPU T4 x2)

Kaggle gives about 30 GPU-hours per week, and a session stops after at most 12 hours. The full
E1–E9 plan (about 230 training runs) may need more than one session. The pipeline handles this
for you:

- It checkpoints training every 200 updates.
- It stops cleanly before the session budget runs out.
- It skips finished phases and resumes an interrupted batch exactly where it stopped.

## One-time setup

1. **Get the code onto Kaggle.** Choose one:
   - **Dataset.** Zip the `wildfire-comm-marl` folder (without `runs/` and `results/`). Go to
     Kaggle → *Datasets* → *New Dataset* and upload it.
   - **GitHub.** Push the folder to a repository and put its URL in `GIT_URL` (cell 1 of the
     notebook).
2. **Create a notebook** from `kaggle/fireworkers_kaggle.ipynb`: *File → Import Notebook*.
3. **Notebook settings** (right panel):
   - *Accelerator* = **GPU T4 x2**
   - *Internet* = **On** (needed for the pip install)
   - If you use the dataset route, use *Add Input* → your code dataset.

## Each session

1. **Commit the notebook:** *Save Version → Save & Run All (Commit)*. It runs in the
   background, so you can close the tab.
   - Cell 4 installs `jax[cuda12]==0.6.2` and checks that both GPUs are visible.
   - Cell 5 runs the fast tests.
   - Cell 6 benchmarks this session's GPUs and prints the projected hours for the remaining
     phases.
   - Cell 7 runs `scripts/run_plan.py --budget-hours 11`.
2. **When the version finishes,** its output (`/kaggle/working`) contains:
   - `wildfire-comm-marl/runs`: trained weights and evaluation logs;
   - `wildfire-comm-marl/results`: tables, JSON reports and figures;
   - `fireworkers_results.zip`.
3. **To continue:** open the notebook, *Add Input → Notebook Output Files →* this notebook
   (latest version), and commit again. Cell 3 copies `runs/` and `results/` back.
   `results/plan_status.json` records which phases are done.

Exit code **3** from `run_plan.py` means "budget reached, progress saved", not an error.

## Choosing the budget

- **Measure before you commit.** Run cell 6 first, interactively if you like (a short
  interactive session also uses quota). It prints env-steps/s for 1, 8 and 24 runs per program,
  and the projected hours per phase at the default `algo.total_updates`.
- **If the plan is too big for your quota,** set `UPDATES` (cell 1). Fewer updates gives less
  converged policies; check that the E1 learning curves (`runs/core/*/seed_*/train_metrics.csv`)
  have flattened. Or run a subset of phases: `PHASES = ["e1", "e2_pilot", "e2"]`, then E3–E8,
  then E9 (the most expensive one, because of N = 50).
- **The plan starts with the task choice** (`e1_calibrate`). It trains NoComm and FullComm on the
  8×8, 10×10 and 12×12 variants (3 seeds each) and writes `results/e1_calibrate/task_choice.json`.
  Every later phase runs on the chosen variant. If you want to look at the choice before spending
  more quota, run `PHASES = ["e1_calibrate", "e1"]` first.
  - If `any_eligible` is false, no variant showed a communication benefit that passes the checks.
  - If `results/e1/e1_report.json → all_pass` is false, adjust the task (see `if_fail`), delete
    `runs/core`, and re-run.
  - Cached runs trained with different settings are refused with a message (not silently reused).
- **Two GPUs** are used automatically: one process trains a batch of runs with the run axis
  split across both T4s (`pmap`). `WF_MAX_DEVICES=1` restricts it to one GPU, and `--max-batch`
  caps the runs per program if memory is tight (it isn't at N = 4; at N = 50 lower
  `--large-num-envs` in E9).

## Commands the notebook runs (also usable in a Kaggle terminal)

```bash
python scripts/benchmark.py --batches 1 8 24
python scripts/run_plan.py --estimate --env-steps-per-s <measured>
python scripts/run_plan.py --budget-hours 11 --seeds 0 1 2 3 4 [--updates N] [--phases e1 e2_pilot ...]
python scripts/run_plan.py --status
```
