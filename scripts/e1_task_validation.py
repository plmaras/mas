"""E1 Task validation: does the wildfire task need communication, and does it leave headroom?

Conditions (same MAPPO settings): NoComm (messages * 0, same graph and parameters), FullComm
(all neighbour gates on), HandRule (gate on when the link distance is below d cells, d swept).
Reference rows (not trained): free-range-zoo's scripted baselines (noop, strongest) and greedy
firefighters with local vs. global (whole-grid) information.

Pass criteria (spec): FullComm clearly beats NoComm (Welch test over seeds, p < 0.05),
FullComm is not at ceiling, the best HandRule beats NoComm, the event (sender sees a burning
cell outside the receiver's view) occurs in 10-30 % of permitted link-steps and varies across
distance bins. If a check fails: shrink env.obs_radius, raise env.base_spread_rate or lower
env.agent_density (more cells per agent).

Task calibration (run first; v2-style pre-set rule): ``--calibrate`` trains NoComm and FullComm
on each candidate variant (3 seeds) and picks the variant with the largest standardised
communication effect d = (FullComm - NoComm) / pooled SD among those with FullComm success < 0.9
and an event rate in 10-30 % (ties -> the first listed). The choice is written to
results/e1_calibrate/task_choice.json and used by scripts/run_plan.py for every later phase.

    python scripts/e1_task_validation.py --calibrate
    python scripts/e1_task_validation.py --seeds 0 1 2 3 4 --set env=wildfire_10x10
    python scripts/e1_task_validation.py --quick
"""
from common import (baseline_conditions, eval_runs, finalize_args, fmt_ci, parser, per_seed, results_dir, run_main,
                    save_json, table, t_ci, train_runs, welch, write_table_csv)

import jax
import numpy as np
from omegaconf import OmegaConf

from analysis import plots
from analysis.metrics import distance_bins, per_bin_rate
from eval import load_logs


def scripted_reference(cfg, n_episodes: int):
    from envs import make_params
    from envs.wildfire_baselines import POLICIES, rollout_stats

    _, p = make_params(cfg.env)
    return {name: rollout_stats(POLICIES[name], p, jax.random.PRNGKey(123), n_episodes)
            for name in ("noop", "random", "strongest", "greedy_local", "greedy_global")}


def calibrate(a):
    a.seeds = a.seeds[:3]
    a.exp_name = "calib_" + a.exp_name
    a = finalize_args(a)
    out = results_dir(a, "e1_calibrate")
    conds = {}
    for v in a.variants:
        conds[f"nocomm_{v}"] = ["comm=nocomm", f"env={v}"]
        conds[f"fullcomm_{v}"] = ["comm=fullcomm", f"env={v}"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    ret, succ, ev = per_seed(summ, "episode_return"), per_seed(summ, "success_rate"), per_seed(summ, "event_rate")
    rows = {}
    for v in a.variants:
        f, n = np.array(ret[f"fullcomm_{v}"]), np.array(ret[f"nocomm_{v}"])
        sd = np.sqrt((f.var(ddof=1) + n.var(ddof=1)) / 2) if f.size > 1 else np.nan
        rows[v] = dict(fullcomm=t_ci(f), nocomm=t_ci(n), gap=float(f.mean() - n.mean()),
                       cohens_d=float((f.mean() - n.mean()) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
                       welch_p=welch(f, n), fullcomm_success=float(np.mean(succ[f"fullcomm_{v}"])),
                       event_rate=float(np.mean(ev[f"fullcomm_{v}"])))
    ok = [v for v in a.variants if rows[v]["fullcomm_success"] < 0.9 and 0.10 <= rows[v]["event_rate"] <= 0.30
          and rows[v]["gap"] > 0]
    pool = ok or a.variants
    key = lambda v: np.nan_to_num(rows[v]["cohens_d"], nan=-np.inf)
    best = max(pool, key=lambda v: (round(key(v), 2), -a.variants.index(v)))
    choice = dict(chosen=best, eligible=ok, rule="max Cohen's d among variants with FullComm success < 0.9, event rate in "
                  "10-30 % and FullComm > NoComm (ties -> first listed); if none is eligible, max d overall",
                  any_eligible=bool(ok), per_variant=rows, seeds=a.seeds)
    save_json(out / "task_choice.json", choice)
    for v, r in rows.items():
        print(f"{v:16s} FullComm {r['fullcomm']['mean']:+.3f}  NoComm {r['nocomm']['mean']:+.3f}  gap {r['gap']:+.3f}  "
              f"d {r['cohens_d']:+.2f}  p {r['welch_p']:.3g}  success {r['fullcomm_success']:.2f}  events {r['event_rate']:.2f}")
    print(f"-> chosen task: env={best}" + ("" if ok else "   (WARNING: no variant passes the checks)"))
    print(f"results in {out}")


def main():
    ap = parser(__doc__)
    ap.add_argument("--calibrate", action="store_true", help="choose the task variant (see docstring)")
    ap.add_argument("--variants", nargs="+", default=["wildfire_8x8", "wildfire_10x10", "wildfire_12x12"])
    ap.add_argument("--hand-d", type=float, nargs="+", default=[1.5, 3.0, 5.0], help="hand-rule distances (cells)")
    ap.add_argument("--ceiling", type=float, default=0.9, help="success rate counted as ceiling")
    a = ap.parse_args()
    if a.calibrate:
        return calibrate(a)
    a = finalize_args(a)
    out = results_dir(a, "e1")

    conds = baseline_conditions()
    for d in a.hand_d:
        conds[f"handrule_d{d:g}"] = ["comm=handrule", f"comm.hand_rule_d={d}"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    keys = ["episode_return", "success_rate", "final_burned_frac", "final_putouts", "active_link_ratio", "event_rate"]
    tab = table(summ, keys)
    write_table_csv(out / "e1_table.csv", tab, keys)

    edges, rates = None, []
    for d in runs["fullcomm"]:
        logs, _ = load_logs(d / f"eval_{a.tag}.npz")
        cfg = OmegaConf.load(d / "config.yaml")
        edges = list(cfg.eval.distance_bins)
        bins = distance_bins(logs["positions"], edges, float(cfg.env.obs_radius))
        rates.append(per_bin_rate(logs["E"], logs["mask"], bins, len(edges) - 1))
    rates = np.array(rates)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" if edges[i + 1] < 1e6 else f">{edges[i]:g}" for i in range(len(edges) - 1)]

    ret, succ = per_seed(summ, "episode_return"), per_seed(summ, "success_rate")
    hand = [k for k in ret if k.startswith("handrule")]
    best_hand = max(hand, key=lambda k: np.mean(ret[k]))
    p_full = welch(ret["fullcomm"], ret["nocomm"])
    p_hand = welch(ret[best_hand], ret["nocomm"])
    ev = float(np.mean(per_seed(summ, "event_rate")["fullcomm"]))
    gap = float(np.mean(ret["fullcomm"]) - np.mean(ret["nocomm"]))
    sd = np.sqrt((np.var(ret["fullcomm"], ddof=1) + np.var(ret["nocomm"], ddof=1)) / 2) if len(a.seeds) > 1 else np.nan
    bin_means = np.nanmean(rates, 0)
    checks = {
        "fullcomm_beats_nocomm": bool(gap > 0 and (p_full < 0.05 if np.isfinite(p_full) else True)),
        "fullcomm_not_at_ceiling": bool(np.mean(succ["fullcomm"]) < a.ceiling),
        "handrule_beats_nocomm": bool(np.mean(ret[best_hand]) > np.mean(ret["nocomm"])
                                      and (p_hand < 0.05 if np.isfinite(p_hand) else True)),
        "event_rate_in_10_30pct": bool(0.10 <= ev <= 0.30),
        "event_rate_varies_across_bins": bool(np.nanmax(bin_means) - np.nanmin(bin_means) > 0.05),
    }
    ref_cfg = OmegaConf.load(runs["fullcomm"][0] / "config.yaml")
    scripted = scripted_reference(ref_cfg, 64 if a.quick else 512)
    report = dict(
        table=tab, best_handrule=best_hand, comm_benefit=dict(fullcomm_minus_nocomm=gap, welch_p=p_full,
                                                              cohens_d=float(gap / sd) if sd and np.isfinite(sd) else None,
                                                              per_seed=dict(fullcomm=ret["fullcomm"], nocomm=ret["nocomm"])),
        besthand_minus_nocomm=dict(diff=float(np.mean(ret[best_hand]) - np.mean(ret["nocomm"])), welch_p=p_hand),
        event_rate_fullcomm=ev, event_rate_per_distance_bin=dict(zip(labels, bin_means.tolist())),
        scripted_reference=scripted, checks=checks, all_pass=all(checks.values()), gate_eval=a.tag,
        if_fail="shrink env.obs_radius, raise env.base_spread_rate, or lower env.agent_density (more cells per agent)")
    save_json(out / "e1_report.json", report)

    plots.points_with_ci(ret, "episode return", out / "e1_return.png", "E1 return per condition (points = seeds)")
    plots.points_with_ci(succ, "success rate", out / "e1_success.png", "E1 success per condition")
    plots.lines_by_group(np.arange(len(labels)), {"event rate (FullComm)": rates}, "link distance / obs_radius",
                         "P(E=1 | permitted link)", out / "e1_event_rate_by_distance.png",
                         "E1 event frequency per distance bin", xticklabels=labels)
    for name, row in tab.items():
        print(f"{name:16s} return {fmt_ci(row['episode_return'])}  success {fmt_ci(row['success_rate'])}")
    for name, r in scripted.items():
        print(f"[scripted] {name:14s} return {r['ret']:.3f}  success {r['success']:.2f}  burned {r['final_burned_frac']:.2f}")
    print(f"FullComm - NoComm: {gap:+.3f} (Welch p = {p_full:.3g}); event rate {ev:.3f}")
    print("checks:", checks)
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
