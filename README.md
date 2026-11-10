# Learned gated communication in cooperative MARL: the FireWorkers wildfire task

This codebase runs every experiment (E1–E9) of `Detailed Experiment Specs, Code-Generation Prompt, and References.md` on a **wildfire-suppression task written in pure JAX**. The task is a port of free-range-zoo's `wildfire_v0` (OASYS lab). The pipeline is built to run on **Kaggle (GPU T4 x2)**.

It compares no communication, full communication and learned binary gating, with a communication-cost penalty and a switching-cost penalty. For each model it measures:

- performance (return, success, burned area);
- link sparsity;
- link persistence;
- the association between open links and a task event;
- whether receivers actually use the messages.

It also includes all of v2's methodological fixes from the earlier spread-task study ([`Roopansha/MasCourseProject`](https://github.com/Roopansha/MasCourseProject)): target link density, λs applied after sparsification, sampled-gate evaluation, time × distance stratified event statistics, and 5 seeds with Holm-corrected tests.

Stack: JAX 0.6.2, Flax 0.10.7, optax 0.2.8, Hydra, pytest. The spread task from the original repo is still included (`env=spread`), so older results stay reproducible.

## Quick start

```bash
pip install -r requirements.txt                      # CPU; on a GPU machine: pip install "jax[cuda12]==0.6.2"
python -m pytest tests -q                            # ~120 tests, ~6 min on CPU
python scripts/e1_task_validation.py --quick         # every script has --quick (tiny budget, ~1-3 min)
python scripts/run_plan.py --quick                   # the whole plan on a tiny budget (~30 min on CPU)
python scripts/benchmark.py                          # throughput on this machine + projected plan hours
python scripts/run_plan.py --budget-hours 11         # the whole E1-E9 plan, resumable
```

**On Kaggle,** open `kaggle/fireworkers_kaggle.ipynb` and follow [`kaggle/README_KAGGLE.md`](kaggle/README_KAGGLE.md).

## The task: FireWorkers (`envs/wildfire.py`)

This is a faithful JAX port of free-range-zoo `wildfire_v0`.

**Grid encoding.** The grid uses v0's encoding:

- `fires > 0` means burning, and the value is the fire type, which equals the attack power needed to suppress it.
- `fires < 0` means flammable but not burning.
- `fires = 0` means not flammable.

Each cell also has an intensity and a fuel count.

**One step.** Each step applies v0's transitions in v0's order:

1. Agents act.
2. Suppressant decrease.
3. Equipment.
4. Refill.
5. Tank capacity.
6. Fire intensity increase (unless enough power attacks the cell).
7. Fire intensity decrease (when attacked with enough power).
8. Spread (v0's 3×3 cross filter, optionally wind-driven).
9. Rewards (put-out reward per fire type, burnout penalty).
10. Termination.

**Fidelity check.** `tests/test_wildfire_v0_parity.py` feeds the same states and the same random numbers to these JAX functions and to free-range-zoo's own PyTorch transition modules. It covers every stochastic/deterministic switch, and all **30 cases match exactly**.

**Extensions for the communication study.** These are switches; v0 behaviour is `env=wildfire_v0`.

| Change | Why |
| --- | --- |
| **Local view** (`obs_radius`, Chebyshev, default 1 = a 3×3 window, the same as the attack range); v0 observes the full grid | With full observability messages carry no private information |
| **Mobile firefighters** (`mobile`, move N/S/W/E); v0 agents are fixed | Gives fires outside an agent's view a reason to matter |
| **Procedural layouts** each episode: fire-type map, initial fires, agent cells. The grid side is `round(sqrt(N / agent_density))` | Generalisation; constant agents-per-cell and fires-per-cell as N grows (8×8 at N = 4, 13×13 at N = 10, 28×28 at N = 50, 40×40 at N = 100) |
| **Fixed-size masked action space**: 0 = noop (refills the tank, as in v0), 1–4 = moves, then "suppress cell at offset (dy, dx)" within range | `jit`/`vmap`; the masks hide invalid actions, as v0's default `show_bad_actions=False` does |
| **Fixed-length episodes** (50 steps); after v0's termination the state is absorbing | Recurrent MAPPO with full-episode rollouts |
| **Team reward divided by N** | Returns are comparable across N |
| **Success** = all fires out at some step with ≤ 10 % of flammable cells burned | "All fires out" alone is also reached by doing nothing, because fires burn out on their own |

**Task variants and calibration.** Communication can only matter if learned agents can't do just as well without it. Three variants are provided; they differ in grid size per agent, initial fires and the share of fires that need 2–3 firefighters at once.

| Variant (N = 4) | Grid | Fire types 1/2/3 | Scripted greedy, own view | Scripted greedy, whole grid | Info gap |
| --- | --- | --- | --- | --- | --- |
| `wildfire_8x8` | 8×8 | 40/40/10 % | −1.6 | +1.0 | +2.6 |
| `wildfire_10x10` | 10×10 | 20/50/20 % | −4.5 | −1.6 | +2.9 |
| `wildfire_12x12` (default) | 12×12 | 20/50/20 % | −4.5 | −1.1 | +3.4 |

The scripted greedy teams don't coordinate, so the gap shows what perfect information sharing alone is worth.

A CPU pilot on `wildfire_8x8` (2 seeds, 400 updates) showed the problem with relying on scripted gaps alone:

- Learned NoComm agents reached +0.87, versus +0.77 for FullComm. Learned search habits made messages unnecessary there.
- The FullComm agents still relied on their messages (zeroing them cost 0.8–1.4 return).

So the plan starts with a **pre-registered task choice**, in the same way v2 chose its observation radius. `e1_task_validation.py --calibrate` trains NoComm and FullComm on each variant (3 seeds) and picks the variant with the largest standardised effect d among those that pass the checks (FullComm success < 0.9, event rate 10–30 %, FullComm > NoComm). `run_plan.py` then passes `env=<chosen>` to every later phase. The default `env=wildfire` (12×12) is used only if calibration is skipped.

**Communication event.** E[t, i, j] = sender i sees a burning cell that is outside receiver j's view (permitted links only). It is computed in O(N²) with a summed-area table, and tested equal to the generic O(N²L) definition.

**Interface mapping.** Grid cells play the role of the spread task's "landmarks", and a burning cell plays the role of an "uncovered landmark". So all analysis code (events, probes, faithfulness, statistics) is shared between the two tasks.

## Model (unchanged from the spread study except where noted)

- **Backbone.** MAPPO with a shared GRU actor, gated attention over messages and a learned null key. The centralized critic never sees messages. For the wildfire task the critic is a narrow per-cell MLP with mean+max pooling (`GridCritic`). It is 4× faster than a 128-wide per-cell DeepSets critic, and it uses only dense layers, so vmapping over runs gives batched matmuls rather than slow grouped convolutions.
- **Gates.** Hard-concrete gates (β = 0.5, γ = −0.1, ζ = 1.1) on links inside a K = 10 nearest-neighbour mask. The sender-side gate net reads the receiver's relative position.
- **Action masking.** Invalid actions are masked with `-1e9` in the same way when sampling, when PPO replays the policy, and in the teacher-forced message tests. The entropy term covers valid actions only.
- **Loss.** `L = L_MAPPO + λc · mean P(g>0) + λs · mean[p_t(1−p_{t−1}) + p_{t−1}(1−p_t)]`.

## Design (v2 fixes applied to every experiment)

| v1 problem (spread task) | Here |
| --- | --- |
| Fixed λc gave all-on or almost-off gates, so densities could not be matched | `penalties=density_target`: a PI controller (Ki 1e-3, Kp 0.05, anti-windup) sets λc so that the mean P(g>0) tracks a target ρ that ramps 1 → ρ over 20–50 % of training |
| λs > 0 blocked sparsification | λs ramps in only after the target is reached (50 → 70 % of training) |
| The positive event association was a time-of-episode confound | Mantel–Haenszel log odds ratio within time-step × distance strata; episode-shuffle null (1000 permutations) plus the spec's circular-shift null |
| Deterministic gate thresholding gave densities the policy never saw in training | Sampled gates are the primary evaluation (`--tag stoch`); deterministic gates are a robustness check |
| One seed | 5 seeds per core condition (3 in E9), one statistic per seed, t-intervals, Holm correction within each test family |

## Experiments

| Script | What it trains / measures | Default conditions |
| --- | --- | --- |
| `e1_task_validation.py --calibrate` | Task choice: NoComm vs FullComm on each variant, pre-set rule → `results/e1_calibrate/task_choice.json` | 3 × 2 × 3 runs |
| `e1_task_validation.py` | NoComm, FullComm, HandRule d ∈ {1.5, 3, 5} cells; pass checks; event rate by distance; scripted reference rows | 5 × 5 runs |
| `e2_sparsity_sweep.py` | RQ1: return vs density at targets ρ ∈ {1, .75, .5, .25, .1, .05}; pre-registered 25 % loss margin; learned vs random fixed mask (C1) at each ρ. `--pilot` checks the controller first | 11 × 5 runs |
| `e3_persistence_grid.py` | RQ2: ρ ∈ {.5, .25, .1} × λs ∈ {0, mid, high}. `--pilot` picks mid/high (switch rate halved / ÷4). Paired-by-seed tests, manipulation checks, density-adjusted version | 9 × 5 (3 cached from E2) |
| `e4_matched_controls.py` | C1–C4 at the learned density, at test time (on FullComm) and trained from scratch | 4 × 5 + cached |
| `e5_event_association.py` | RQ3 on the E3 grid: MH log OR, nulls, Holm; persistence comparison | cached |
| `e6_message_use.py` | F1–F4 teacher-forced KL, pre-registered F4 direction test, closed-loop return drops | cached + 5 |
| `e7_message_probe.py` | Linear/MLP probes: private fire offset, "sees a fire", human label; shuffled baselines | cached |
| `e8_method_comparison.py` | NoComm, FullComm, AE-Comm, 9-token language, gated and IC3Net at the same ρ; cross-play | 3 × 5 + cached |
| `e9_transfer_scale.py` | N ∈ {4, 10, 50}, zero-shot transfer 50 → 100, straight-through / REINFORCE gates, second tasks on top of the chosen variant (`r2`: 5×5 view; `wind`: v0 wind-driven spread); fps and peak memory | ~60 runs, 3 seeds |
| `run_plan.py` | All of the above in order, with a time budget and resume | |
| `benchmark.py` | env-steps/s on the current GPUs and projected hours | |

Runs are cached in `runs/<exp>/<run>/seed_<s>/` and shared between experiments. Results and figures go to `results/<exp>/`. Every script accepts `--seeds`, `--quick`, `--updates`, `--tag {stoch,det}`, `--max-batch`, `--train-only`, `--set <hydra overrides>` and `--retrain`.

## Speed and Kaggle features

- **One program, many runs, two GPUs.** `train.run_training_batch` vmaps the PPO update over independent runs (seeds and penalty settings that share static settings). It splits the run axis across all local GPUs with `pmap`; the runs don't communicate. `scripts/common.train_runs` groups pending runs this way automatically.
- **Resumable.** The full training state (params, optimizer, PRNG keys, PI controller, metrics) is checkpointed atomically every 200 updates to `runs/<exp>/_resume/`. If the environment variable `WF_DEADLINE` (unix time) is close, training checkpoints and exits with code 3, and re-running resumes. Tests check that interrupted-then-resumed training and 2-device training reproduce uninterrupted 1-device training.
- **Compile once per configuration.** Evaluation programs are cached per static configuration, and parameters are passed in as arguments. Evaluating many seeds and conditions doesn't recompile.
- **Small logs.** Observations aren't saved (no analysis uses them), and constant cell positions are stored once. Deterministic-gate checks keep summaries only.

## Repository layout

| Path | Purpose |
| --- | --- |
| `envs/wildfire.py` | The JAX wildfire env (v0 transitions + the extensions above) |
| `envs/wildfire_baselines.py` | Scripted policies (v0 baselines and greedy local/global) |
| `envs/spread.py` | The original cooperative-navigation task |
| `envs/__init__.py` | Env registry (`env.kind`) |
| `configs/env/wildfire*.yaml` | `wildfire_base` (all v0 parameters), the variants `wildfire_8x8` / `_10x10` / `_12x12`, `wildfire` (default = 12×12), `wildfire_r2` (5×5 view), `wildfire_wind` (v0 realistic wind spread), `wildfire_v0` (static, fully observed) |
| `configs/penalties/density_target.yaml` | The PI-controller settings |
| `core/`, `models/`, `train.py`, `eval.py` | Rollout, MAPPO, gates, actor-critic, batched multi-GPU training, cached evaluation |
| `analysis/association.py` | Per-run RQ3 statistics (MH log OR, nulls) |
| `analysis/` (rest) | Link metrics, events, nulls, faithfulness, probes, plots |
| `scripts/` | E1–E9, `run_plan.py`, `benchmark.py` |
| `kaggle/` | Notebook and guide |
| `tests/` | Env unit tests, the v0 parity test, comm/analysis tests, integration tests (resume, multi-device) |

Link convention: arrays indexed `[i, j]` are the directed link from **sender i** to **receiver j**.

## Assumptions and known limitations

- **These extensions change v0.** The local view, mobility, procedural layouts and fixed-length absorbing episodes are needed for a communication study. Results are about this task, not about v0 as published.
- **Task difficulty is chosen by a rule, not tuned by hand.** The task-choice phase picks among three variants with learned agents. If no variant passes, the choice is flagged (`any_eligible: false`); then E1's `if_fail` hint applies before running the rest.
- **Hard-concrete gates are not strictly binary.** "Open" means g > 0, and an open gate can pass a scaled message (0 < g < 1).
- **Messages are not fed back into the GRU.** So the E6 tests are one-step tests.
- **Association is not causation.** Faithfulness tests measure message influence, not the reasons behind an action.
- **Large N is expensive.** N = 50 and 100 cost far more than N = 4. E9's large-N settings (`--large-num-envs`, `--large-updates`) trade budget for training quality; check large-N success before reading scaling results.
- **The parity test needs extra installs.** It requires PyTorch and a free-range-zoo checkout (`FREE_RANGE_ZOO_SRC`), and is skipped otherwise.
- **This code was generated with an assistant.** The env transitions are differential-tested against v0, and the event definition, nulls and statistics are unit-tested. Review them before relying on results.
# mas
