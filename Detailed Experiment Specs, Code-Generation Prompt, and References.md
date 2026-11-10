# Detailed Experiment Specs, Code-Generation Prompt, and References

This document expands each experiment (E1 to E9) into an implementable spec, gives a structured prompt you can paste into a coding assistant, and lists the original papers and code. Reference status is marked in section 4: some links were checked during this project, others are from memory and must be verified before you cite or depend on them.

## 1. Shared specification (applies to every experiment)

**Task (default).** Cooperative navigation in JAX (JaxMARL-style simple spread): N agents must cover N landmarks in continuous 2D. Each agent sees only entities within radius R\_obs. Shared team reward is the negative sum over landmarks of the distance to the nearest agent, minus a collision penalty. Entity density is held constant as N grows. A landmark counts as covered when an agent is within a fixed distance threshold. Success means all landmarks covered before the time limit.

**Backbone.** MAPPO with parameter sharing, GRU actors, and a centralized critic that never receives messages (Yu et al. 2022).

**Communication module.** Each agent i produces a message vector m\_i. A gate g\_ij on the directed link i to j decides whether j can read m\_i. The receiver uses attention (query: own observation embedding, keys and values: incoming messages). Attention logits are masked where g=0, and a learned null key prevents NaNs when every link is closed. Links exist only inside a spatial K-nearest-neighbor mask, and links outside it count as off.

**Gate.** Hard-concrete gate (Louizos et al. 2018) with parameters beta=0.5, gamma=-0.1, zeta=1.1. Active means g>0. Use deterministic gates at evaluation, and also report stochastic-gate statistics separately.

**Penalties.**

- Communication cost: mean over permitted links of P(g>0).
- Switching cost: expected flip probability, p\_t(1-p\_{t-1}) + p\_{t-1}(1-p\_t), over links permitted at both steps.
- Warm-up: both lambdas are zero for the first part of training, then annealed to target. Initialize gates near on.

**Evaluation logging (save for every evaluation episode).** Positions, actions, action logits, gate logits alpha\[t,i,j\], sampled and deterministic gates g\[t,i,j\], neighbor mask\[t,i,j\], messages\[t,i,:\], attention weights, visible-landmark flags vis\[t,i,L\], covered\[t,L\], event flags E\[t,i,j\], and team reward\[t\].

**Seeds and statistics.** 10 seeds for core conditions if throughput allows (5 minimum). Report per-seed points with bootstrap confidence intervals and the interquartile mean (rliable). With only 5 seeds the IQM is fragile, so always show the raw points.

**Matched-density rule.** Compare persistence effects only at equal link density, because all-off and all-on gates both have zero switching cost.

## 2. Experiment specifications

### E1. Task validation

- **Aim:** show the task needs communication and has headroom, so later differences mean something.
- **Conditions:** No-Comm (same graph, messages multiplied by 0 for parameter and FLOP parity), Full-Comm (all neighbor gates on), Hand-Rule (gate on when distance is below d, with d swept).
- **Procedure:** train each with identical MAPPO settings. Evaluate reward, success rate, and event frequency.
- **Outputs:** table of reward and success per condition, event frequency per distance bin.
- **Pass criteria:** Full-Comm clearly beats No-Comm. Full-Comm is not at ceiling. Hand-Rule beats No-Comm. The event occurs in roughly 10 to 30 percent of link-timesteps, with variation across distance bins.
- **If it fails:** shrink R\_obs, hide landmarks more often, or add more landmarks than the local view can cover.

### E2. Sparsity sweep (RQ1)

- **Aim:** can agents use fewer links without a substantial loss in performance?
- **Conditions:** lambda\_c swept over about 6 values with lambda\_s = 0. Calibrate on a pilot so link density spans roughly 100 percent down to 10 to 20 percent.
- **Procedure:** train each setting, evaluate with deterministic gates, record density and reward.
- **Outputs:** reward versus active-link-ratio curve with per-seed points and CIs.
- **Readout:** the density below which reward drops, and whether the curve is smooth or shows gate collapse.
- **Failure modes:** gate collapse (all off early). Fix with warm-up, annealing, near-on initialization, or an entropy bonus on the gate.

### E3. Persistence at matched density (RQ2)

- **Aim:** does encouraging links to persist change performance independently of link count?
- **Conditions:** 3x3 grid of lambda\_c in {0, small, large} and lambda\_s in {0, small, large}. The cell with both zero is the learned, unpenalized model.
- **Procedure:** train all cells. Bin models by link density and compare reward and switch rate within a bin.
- **Outputs:** heatmap of reward, density, and switch rate per cell. Scatter of reward versus switch rate colored by density.
- **Metrics:** gate-driven switch rate, mask-driven switch rate (neighbor set changes), and mean link-run length. Report them separately.
- **Readout:** if reward or event lift differs between lambda\_s settings at equal density, persistence matters independently. If only the switch rate changes, persistence affects stability but not performance.

### E4. Matched-mask controls

- **Aim:** show whether learned gating matters beyond having fewer links.
- **Controls:**
  - C1 random fixed mask at the learned density.
  - C2 random time-varying mask matched in density and switch rate.
  - C3 distance-rule mask matched in density.
  - C4 learned gates with message contents zeroed.
- **Procedure:** run each control both at evaluation time (apply the mask to a trained Full-Comm model) and trained from scratch with the mask fixed.
- **Outputs:** reward of each control against the learned model at the same density.
- **Readout:** learned beats C1 and C3 means the learning found useful links. Learned equals C1 means it only removed links. C4 near learned means the messages carry nothing.

### E5. Event association (RQ3)

- **Aim:** are active links associated with a task event more than chance, and does this differ for persistence-penalized models?
- **Event:** E\[t,i,j\] = 1 when sender i sees an uncovered landmark that is outside receiver j's observation radius. Computed only for links inside the neighbor mask.
- **Statistics:**
  - Signed phi coefficient (and Cramer's V for magnitude).
  - Lift = P(g=1 | E=1) / P(g=1 | E=0).
  - AUROC for predicting E from g.
- **Null:** circular time-shifts within each (episode, link), at least 1,000 shifts, preserving each link's autocorrelation and density. Do not use plain timestep shuffling, which makes persistent models look significant.
- **Confound control:** compute within distance bins, since distance affects both gates and events.
- **Aggregation:** one statistic per seed, then bootstrap across seeds. Never treat link-timesteps as independent samples.
- **Outputs:** effect sizes with CIs per model, null distribution plots, comparison across lambda\_s settings at matched density.
- **Wording:** report association, not causation.

### E6. Message-use test (faithfulness)

- **Aim:** are messages actually used, or only aligned with events? This tests the case where an agent says something that never changes the receiver's next action.
- **Interventions at test time:**
  - F1 zero all messages.
  - F2 ablate active links at event times versus the same number of active links at random times.
  - F3 swap messages with those from another sender or episode.
  - F4 targeted counterfactual: replace a message with one encoding a different landmark position and check whether the receiver's action moves toward the new target (define the predicted direction before running).
- **Metrics:** receiver policy KL (with versus without intervention), reward drop, and the fraction of swaps where the action change matches the prediction.
- **Outputs:** per-model faithfulness scores, plotted against event lift from E5.
- **Readout:** high lift with near-zero KL means decorative messages. Low lift with large KL means useful but event-unrelated messages.

### E7. Message content probe

- **Aim:** do messages carry the sender's private information?
- **Procedure:** train a linear probe (and a small MLP for comparison) from the message to the positions of landmarks the receiver cannot see. Use a held-out split by episode, and a shuffled-label baseline.
- **Outputs:** probe R-squared or accuracy against baseline, per model and per link-density level.
- **Readout:** interpret with E6. Information present in a message does not show that it is used.

### E8. Method comparison

- **Aim:** compare communication methods on performance, readability, and faithfulness under one backbone and training budget.
- **Methods:** No-Comm, RL-trained messages, AE-Comm (autoencoder-grounded messages), a language-style variant (fixed vocabulary of human-readable tokens, or LangGround-style alignment to language embeddings), and the gated model. Optionally IC3Net-style gating as a literature baseline.
- **Procedure:** equal tuning budget per method, 5+ seeds.
- **Metrics:** reward, probe accuracy (E7), faithfulness (E6), vocabulary size or bits per step, and cross-play (pair a sender and receiver from different seeds).
- **Outputs:** reward versus faithfulness, and readability versus faithfulness, one point per method with error bars.
- **Caution:** language-grounded methods score well on readability by construction, so weigh faithfulness more heavily.

### E9. Transfer and scale

- **Aim:** check whether findings are specific to one N, one gate estimator, or one task.
- **Runs:**
  - N in {4, 10, 50}. The neighbor limit only bites at N of about 12 or more (K=10), so make scaling claims from N=50.
  - Zero-shot transfer from N=50 to N=100 with frozen weights.
  - Gate estimator swap: straight-through or REINFORCE gate on key conditions.
  - A second task or observation radius.
- **Metrics:** throughput (frames per second), peak memory, and the main results of E3 and E5 repeated.
- **Fallback:** if memory runs out, reduce the number of parallel environments and use gradient accumulation.

## 3. Structured prompt for code generation

Use the master prompt once to generate the shared codebase, then use the per-experiment template for each experiment.

### 3.1 Master prompt

```
ROLE
You are an expert JAX/Flax research engineer. Write clean, tested, reproducible code.

PROJECT
Learned communication in cooperative MARL: compare no communication, full communication,
and learned binary gating with a communication-cost penalty and a switching-cost penalty.
We measure performance, link sparsity, link persistence, event association, and whether
messages are actually used by receivers.

STACK
Python 3.10+, JAX + Flax (or PyTorch if JAX unavailable), optax, Hydra configs,
Weights & Biases logging, pytest. Pin versions in requirements.txt and TEST the install
before writing code that depends on any API.

ENVIRONMENT
Cooperative continuous 2D navigation, N agents and N landmarks, observation radius R_obs,
shared reward = -sum_landmarks min_agent_distance - collision_penalty, constant entity
density as N grows. Expose in info: positions, vis[t,i,L] (landmark visibility per agent),
covered[t,L]. Unit-test visibility and reward logic.

MODEL
- MAPPO, parameter sharing, GRU actor, centralized critic (critic never sees messages).
- Message module: sender message m_i; receiver attention over incoming messages with logits
  masked where gate=0 and a learned null key to avoid NaNs.
- Spatial K-nearest-neighbor mask restricts which links exist (jax.lax.top_k).
- Hard-concrete gate (beta=0.5, gamma=-0.1, zeta=1.1), active means g>0, deterministic
  gates at eval. Gate init near on.
- Losses: L = L_MAPPO + lambda_c * mean P(g>0) + lambda_s * mean[p_t(1-p_{t-1}) + p_{t-1}(1-p_t)]
  over links permitted at both steps. Lambda warm-up at zero then linear anneal.
- Baselines: NoComm (messages * 0, identical graph and parameter count), FullComm (all
  neighbor gates on), HandRule (gate on if distance < d).
- Controls: random fixed mask, random time-varying mask (matched density and switch rate),
  distance mask (matched density), zeroed-message model.

LOGGING (eval episodes, saved as .npz)
positions, actions, action_logits, alpha[t,i,j], g[t,i,j], mask[t,i,j], messages[t,i,:],
attn[t,i,j], vis[t,i,L], covered[t,L], E[t,i,j], reward[t].

ANALYSIS UTILITIES
- metrics: active_link_ratio, gate_switch_rate, mask_switch_rate, mean_run_length.
- event stats: phi, Cramer's V, lift, AUROC, within distance bins.
- null: circular time-shift within each (episode, link), >= 1000 shifts, jit/vmap.
- aggregation: per-seed statistic, then bootstrap CI across seeds, IQM via rliable.

REPO LAYOUT
configs/{algo,env,penalties}/, envs/, models/, scripts/, tests/, train.py, eval.py,
analysis/, requirements.txt, README.md

REQUIREMENTS
1. Seed everything (JAX PRNGKey, numpy, random). 2. Hydra configs only, no hardcoded
hyperparameters. 3. Log git hash and config to W&B. 4. Include pytest tests for: visibility,
reward, gate shapes, mask correctness, penalty values on toy inputs, null-model sanity
(a persistent random gate should NOT show significant association under circular shifts).
5. State assumptions explicitly. Do not invent library APIs: if unsure, say so and write
the code against a thin wrapper I can adapt.

OUTPUT FORMAT
For each file: path, purpose, then full code. End with a run-order checklist and known
limitations.
```

### 3.2 Per-experiment prompt template

```
EXPERIMENT: <ID and name>
AIM: <one sentence>
INPUTS: <config keys, checkpoints, logged arrays used>
PROCEDURE: <numbered steps>
METRICS: <exact formulas>
NULL / CONTROLS: <what to compare against and how it is generated>
OUTPUTS: <files, tables, figures with axis definitions>
ACCEPTANCE TESTS: <what must be true for the code to be considered correct>
CONSTRAINTS: reuse the shared codebase and logging schema, no new dependencies without asking.
```

### 3.3 Filled example (E5, event association)

```
EXPERIMENT: E5 Event association
AIM: Measure whether active gates are associated with the event
     "sender i sees an uncovered landmark outside receiver j's observation radius".
INPUTS: eval logs with g[t,i,j], mask[t,i,j], vis[t,i,L], covered[t,L], positions.
PROCEDURE:
 1. Compute E[t,i,j] = any_L( vis[t,i,L] & ~covered[t,L] & ~(landmark L within R_obs of j) ),
    restricted to links where mask[t,i,j]=1.
 2. Per seed, compute phi, Cramer's V, lift, AUROC pooled over episodes and links, and
    separately within distance bins.
 3. Null: for each (episode, link), circularly shift g by a random offset, recompute the
    statistics, repeat >= 1000 times. Report observed minus null mean and the percentile.
 4. Aggregate one number per seed, bootstrap CI across seeds.
METRICS: phi = (n11*n00 - n10*n01)/sqrt(row/col marginals); lift = P(g|E=1)/P(g|E=0).
OUTPUTS: results table per model; null histograms; plot of lift vs gate switch rate at
     matched density.
ACCEPTANCE TESTS: (a) synthetic gates generated from E give high phi; (b) a persistent
     random gate gives phi near the circular-shift null; (c) results unchanged under
     relabeling of agents.
CONSTRAINTS: vectorized with jax.vmap, no Python loops over shifts.
```

## 4. References and code

**Verified during this project (checked via search):**

- **AE-Comm.** Lin, Huh, Stauffer, Lim, Isola. *Learning to Ground Multi-Agent Communication with Autoencoders.* NeurIPS 2021. arXiv:2110.15349. Code (PyTorch, includes CIFAR Game, FindGoal, RedBlueDoors): https://github.com/ToruOwO/marl-ae-comm. Project page: https://toruowo.github.io/marl-ae-comm/
- **LangGround.** Li et al. *Language Grounded Multi-agent Reinforcement Learning with Human-interpretable Communication.* NeurIPS 2024. arXiv:2409.17348. The paper points to a code page at https://romanlee6.github.io/langground\_web/ (check there for the repository link). The paper evaluates against IC3Net and AE-Comm on tasks including USAR and Predator-Prey, which makes it a useful template for your method comparison.
- **GLC.** Du et al. *Learning Efficient and Interpretable Multi-Agent Communication.* ICLR 2026. Frames a trilemma between performance, communication efficiency, and human interpretability. I did not find a code link, so check the paper page.

**From memory, verify before citing or relying on them:**

| Topic | Reference | Code (if known) |
| --- | --- | --- |
| CommNet | Sukhbaatar et al. 2016, arXiv:1605.07736 | github.com/facebookresearch/CommNet |
| IC3Net (learned binary gates) | Singh et al. 2019, arXiv:1812.09755 | github.com/IC3Net/IC3Net |
| TarMAC | Das et al. 2019, arXiv:1810.11187 | check authors' page |
| TMC (Temporal Message Control) | Zhang et al. 2020, arXiv:2010.14391 | check authors' page |
| VBC | Zhang et al. 2019 | check authors' page |
| NDQ | Wang et al. 2020 | check authors' page |
| Hard-concrete L0 gates | Louizos et al. 2018, arXiv:1712.01312 | github.com/AMLab-Amsterdam/L0\_regularization |
| Gumbel-softmax / Concrete | Jang et al. 2017; Maddison et al. 2017 | many implementations |
| MAPPO | Yu et al. 2022, arXiv:2103.01955 | github.com/marlbenchmark/on-policy |
| JaxMARL | Rutherford et al. 2024, arXiv:2311.10090 | github.com/FLAIROx/JaxMARL |
| Pitfalls of measuring emergent communication | Lowe et al. 2019 | n/a |
| Social influence as intrinsic motivation | Jaques et al. 2019 | n/a |
| rliable (IQM, bootstrap CIs) | Agarwal et al. 2021 | github.com/google-research/rliable |
| Deep RL that matters | Henderson et al. 2018 | n/a |

**Notes on reuse.** The AE-Comm repository is the most direct starting point for a message-based baseline and for the FindGoal and RedBlueDoors tasks. For gated communication, IC3Net is the closest public precedent, though it uses a REINFORCE-trained gate, so your hard-concrete gate is a different estimator. Check licenses, library versions, and whether each repository still runs before building on it.

## 5. Caveats

- Event analysis shows association, not that a message caused a behavior change or what it means.
- Faithfulness tests measure message influence and consistency, not the reasons behind an individual agent's action.
- Pin and test library versions (JAX, JaxMARL, jaxlib) before committing to the stack. Version and throughput numbers in earlier planning documents were not verified.
- Generated code must be reviewed and unit-tested, especially the visibility logic, the event definition, and the null model, since silent bugs there would invalidate the results.
