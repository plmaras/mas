"""E9 Transfer and scale: are the findings specific to one N, one gate estimator, or one task?

Runs (gated models at target density ``--rho`` with lambda_s in {0, ``--ls``}):
  * N in {4, 10, 50}; the grid grows with N at constant agents-per-cell and fires-per-cell.
    The K=10 neighbour mask only bites at N >= 12, so scaling claims come from N = 50.
  * zero-shot transfer N=50 -> N=100 with frozen weights (observations have a fixed size);
  * gate estimator swap: straight-through and REINFORCE gates at the same target density;
  * second tasks, applied on top of the current task variant: a wider local view (``r2``:
    5x5 instead of 3x3) and v0's wind-driven spread (``wind``: downwind ~0.16, crosswind ~0.07).
Metrics: training throughput (env-steps/s), peak memory, return / success / burned area, and the
E3/E5 headline numbers (density, gate- and mask-driven switch rate, run length, MH log odds
ratio vs the episode-shuffle null).
Memory fallback: lower --large-num-envs and raise --grad-accum. For clean peak-memory numbers,
run one N per process (e.g. --ns 50).

    python scripts/e9_transfer_scale.py --rho 0.25 --ls 0.03 --seeds 0 1 2
"""
from common import (association, eval_runs, finalize_args, gated_overrides, gname, parser, results_dir, run_main,
                    save_json, train_runs)

import json

import numpy as np

from analysis import plots
from analysis.stats import summarize_seeds
from eval import evaluate_run

TASKS = {  # second tasks as overrides on top of the current env (works for every calibrated variant)
    "r2": ["env.obs_radius=2"],
    "wind": ["env.realistic_fire_spread=true", "env.base_spread_rate=13.5", "env.max_spread_rate=32.0",
             "env.wind_direction=1.5707963"],
}

def main():
    ap = parser(__doc__, default_seeds=(0, 1, 2))
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 10, 50])
    ap.add_argument("--transfer-from", type=int, default=50)
    ap.add_argument("--transfer-to", type=int, default=100)
    ap.add_argument("--transfer-episodes", type=int, default=50)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--ls", type=float, default=0.03)
    ap.add_argument("--estimator-n", type=int, default=4)
    ap.add_argument("--tasks", nargs="*", default=["r2", "wind"], choices=list(TASKS))
    ap.add_argument("--large-n", type=int, default=20, help="N at/above which the large-N settings apply")
    ap.add_argument("--large-num-envs", type=int, default=16)
    ap.add_argument("--large-updates", type=int, default=None, help="training updates at large N (default: same)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--large-eval-episodes", type=int, default=50)
    a = finalize_args(ap.parse_args())
    if a.quick:
        a.ns = [n for n in a.ns if n <= 10] + [12]
        a.transfer_from, a.transfer_to, a.transfer_episodes = 12, 16, 4
    out = results_dir(a, "e9")
    rho, ls = a.rho, a.ls

    def size_ov(n):
        ov = [f"env.n_agents={n}"]
        if n >= a.large_n and not a.quick:
            ov += [f"algo.num_envs={a.large_num_envs}", f"algo.grad_accum_steps={a.grad_accum}",
                   f"eval.episodes={a.large_eval_episodes}", f"eval.batch_size={min(a.large_eval_episodes, 25)}",
                   f"algo.num_minibatches={min(4, a.large_num_envs)}"]
            if a.large_updates:
                ov.append(f"algo.total_updates={a.large_updates}")
        return ov

    gated_models = {}
    conds = {}
    for n in a.ns:
        conds[f"fullcomm_N{n}"] = ["comm=fullcomm"] + size_ov(n)
        conds[f"nocomm_N{n}"] = ["comm=nocomm"] + size_ov(n)
        for s in (0.0, ls):
            gated_models[f"{gname(rho, s)}_N{n}"] = gated_overrides(rho, s) + size_ov(n)
    for est in ("gated_ste", "gated_reinforce"):
        for s in (0.0, ls):
            gated_models[f"{gname(rho, s, est)}_N{a.estimator_n}"] = gated_overrides(rho, s, comm=est) + size_ov(a.estimator_n)
    for task in a.tasks:
        conds[f"fullcomm_{task}"] = ["comm=fullcomm", *TASKS[task]]
        conds[f"nocomm_{task}"] = ["comm=nocomm", *TASKS[task]]
        for s in (0.0, ls):
            gated_models[f"{gname(rho, s)}_{task}"] = gated_overrides(rho, s) + TASKS[task]
    conds.update(gated_models)

    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag, light=True)  # E9 needs gates/events/summaries only (memory at N = 50)

    report = {}
    for name, dirs in runs.items():
        metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
        r = {k: summarize_seeds([s.get(k, np.nan) for s in summ[name]]) for k in
             ("episode_return", "success_rate", "final_burned_frac", "active_link_ratio", "gate_switch_rate",
              "mask_switch_rate", "mean_run_length")}
        r["fps_env_steps"] = summarize_seeds([m["fps_env_steps"] for m in metas])
        r["fps_env_steps_batch"] = summarize_seeds([m.get("fps_env_steps_batch", np.nan) for m in metas])
        r["peak_memory_mb"] = summarize_seeds([m["peak_memory_mb"] for m in metas])
        if name in gated_models:
            ev = [association(d, a, seed=k).get("events") or {} for k, d in enumerate(dirs)]
            r["log_or_time_dist"] = summarize_seeds([e.get("log_or_time_dist", np.nan) for e in ev])
            r["shuffle_p_per_seed"] = [(e.get("shuffle_vs_log_or_time_dist") or {}).get("p_value") for e in ev]
            r["phi"] = summarize_seeds([e.get("phi", np.nan) for e in ev])
            r["lift"] = summarize_seeds([e.get("lift", np.nan) for e in ev])
        report[name] = r

    transfer = {}
    for base in ("fullcomm", "nocomm", gname(rho, 0.0), gname(rho, ls)):
        src = f"{base}_N{a.transfer_from}"
        if src not in runs:
            continue
        ss = [evaluate_run(d, tag=f"transfer_N{a.transfer_to}_{a.tag}", save=False, light=True, deterministic=(a.tag == "det"),
                           n_episodes=a.transfer_episodes, batch_size=min(a.transfer_episodes, 25),
                           env_overrides={"n_agents": a.transfer_to}) for d in runs[src]]
        transfer[base] = dict(
            in_distribution_return=report[src]["episode_return"],
            zero_shot_return=summarize_seeds([s["episode_return"] for s in ss]),
            zero_shot_success=summarize_seeds([s["success_rate"] for s in ss]),
            zero_shot_burned=summarize_seeds([s.get("final_burned_frac", np.nan) for s in ss]),
            zero_shot_density=summarize_seeds([s["active_link_ratio"] for s in ss]))
    save_json(out / "e9_report.json", dict(
        per_condition=report, zero_shot_transfer=transfer, gate_eval=a.tag,
        note="team reward is divided by N, so returns are comparable across N. fps_env_steps is per run inside a "
             "batched multi-GPU program (fps_env_steps_batch = whole program). peak_memory_mb is the process-lifetime "
             "peak: run one N per process (e.g. --ns 50) for clean memory numbers"))

    ys = {}
    for base in ("nocomm", "fullcomm", gname(rho, 0.0), gname(rho, ls)):
        ys[base] = np.array([[summ[f"{base}_N{n}"][k]["episode_return"] for n in a.ns] for k in range(len(a.seeds))])
    plots.lines_by_group(np.arange(len(a.ns)), ys, "number of agents N", "episode return (team reward / N)",
                         out / "e9_return_vs_N.png", "E9 return vs N", xticklabels=[str(n) for n in a.ns])
    fps = {base: np.array([[json.loads((runs[f"{base}_N{n}"][k] / "meta.json").read_text())["fps_env_steps"]
                            for n in a.ns] for k in range(len(a.seeds))]) for base in ys}
    plots.lines_by_group(np.arange(len(a.ns)), fps, "number of agents N", "training throughput per run (env-steps / s)",
                         out / "e9_fps_vs_N.png", "E9 throughput vs N", xticklabels=[str(n) for n in a.ns])
    for n, r in report.items():
        extra = f"  logOR {r['log_or_time_dist']['mean']:+.3f}" if "log_or_time_dist" in r else ""
        print(f"{n:34s} return {r['episode_return']['mean']:8.3f}  density {r['active_link_ratio']['mean']:.2f}  "
              f"switch {r['gate_switch_rate']['mean']:.3f}  fps {r['fps_env_steps']['mean']:.0f}  "
              f"mem {r['peak_memory_mb']['mean']:.0f}MB{extra}")
    for b, t in transfer.items():
        print(f"transfer {b}: N={a.transfer_from} {t['in_distribution_return']['mean']:.3f} -> "
              f"N={a.transfer_to} {t['zero_shot_return']['mean']:.3f}")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
