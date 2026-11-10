"""E2 Sparsity sweep (RQ1): can agents use fewer links without a substantial loss?

v2 design: instead of sweeping lambda_c (v1 on the spread task: gates were all-on or all-off,
with one intermediate point), each condition is trained to a TARGET link density rho; a PI
controller sets lambda_c (configs/penalties/density_target.yaml). lambda_s = 0 here.
rho = 1 is the learned but unpenalised gate.

Pre-specified RQ1 readout (from the v2 analysis plan): with the communication benefit
D = FullComm - NoComm, the loss of a gated model is L = FullComm - Gated(rho); a loss is
"substantial" when L / D > 0.25. Verdict from the bootstrap CI of L / D: upper < 0.25 -> no
substantial loss, lower > 0.25 -> substantial loss, else inconclusive. Secondary: learned gates
vs. a random fixed mask trained from scratch at the same density (C1; Welch, Holm over rho).

    python scripts/e2_sparsity_sweep.py --pilot                       # 1 seed, 1/3 budget: does the controller hold?
    python scripts/e2_sparsity_sweep.py --seeds 0 1 2 3 4
"""
from common import (baseline_conditions, eval_runs, finalize_args, fmt_ci, gated_overrides, gname, holm, parser,
                    per_seed, ratio_ci, results_dir, run_main, save_json, t_ci, table, train_runs, welch,
                    write_table_csv)

import numpy as np

from analysis import plots

DEFAULT_RHOS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
MARGIN = 0.25


def main():
    ap = parser(__doc__)
    ap.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS, help="target link densities (1 = no target)")
    ap.add_argument("--no-c1", action="store_true", help="skip the random-mask (C1) controls")
    ap.add_argument("--pilot", action="store_true", help="1 seed, 1/3 of the training budget, separate namespace")
    a = ap.parse_args()
    if a.pilot:
        a.seeds = a.seeds[:1]
        a.exp_name = "pilot_" + a.exp_name
        a.no_c1 = True
        if a.quick:
            pass  # the quick budget already applies
        elif a.updates:
            a.updates = max(a.updates // 3, 1)
        else:
            from common import compose_cfg
            a.updates = max(int(compose_cfg([]).algo.total_updates) // 3, 1)
    a = finalize_args(a)
    out = results_dir(a, "e2_pilot" if a.pilot else "e2")

    gated = {gname(r, 0.0): gated_overrides(r, 0.0) for r in a.rhos}
    conds = dict(gated)
    if not a.pilot:
        conds.update(baseline_conditions())
    if not a.no_c1:
        for r in a.rhos:
            if r < 1:
                conds[f"c1fixed_rho{r:g}"] = ["comm=random_fixed", f"comm.density={r}"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    summ_det = eval_runs({n: d for n, d in runs.items() if n in gated}, tag="det", save=False)  # summaries only
    keys = ["episode_return", "success_rate", "final_burned_frac", "active_link_ratio", "gate_switch_rate",
            "mean_run_length"]
    tab = table(summ, keys)
    write_table_csv(out / f"e2_table_{a.tag}.csv", tab, keys)
    write_table_csv(out / "e2_table_det.csv", table(summ_det, keys), keys)

    dens, ret = per_seed(summ, "active_link_ratio"), per_seed(summ, "episode_return")
    names = list(gated)
    controller = {n: dict(target=r, achieved=t_ci(dens[n]), abs_error=float(abs(np.mean(dens[n]) - r)) if r < 1 else None)
                  for n, r in zip(names, a.rhos)}
    collapse = {n: bool(np.mean(dens[n]) < 0.02) for n in names}
    report = dict(rhos=a.rhos, gate_eval=a.tag, table=tab, controller=controller, gate_collapse=collapse)
    if not a.pilot:
        f_ret, n_ret = ret["fullcomm"], ret["nocomm"]
        rq1, p_c1 = {}, {}
        for n, r in zip(names, a.rhos):
            rc = ratio_ci(f_ret, n_ret, ret[n])
            verdict = ("no substantial loss" if rc["hi"] < MARGIN else "substantial loss" if rc["lo"] > MARGIN
                       else "inconclusive")
            e = dict(target=r, density=t_ci(dens[n]), loss_fraction=rc, verdict=verdict, ret=t_ci(ret[n]))
            c1 = f"c1fixed_rho{r:g}"
            if c1 in ret:
                e["vs_c1"] = dict(diff=float(np.mean(ret[n]) - np.mean(ret[c1])), welch_p=welch(ret[n], ret[c1]),
                                  c1_return=t_ci(ret[c1]), c1_loss_fraction=ratio_ci(f_ret, n_ret, ret[c1]))
                p_c1[n] = e["vs_c1"]["welch_p"]
            rq1[n] = e
        for n, p in holm(p_c1).items():
            rq1[n]["vs_c1"]["holm_p"] = p
        report.update(benefit=dict(diff=float(np.mean(f_ret) - np.mean(n_ret)), welch_p=welch(f_ret, n_ret)),
                      rq1=rq1, margin=MARGIN)
    save_json(out / "e2_report.json", report)

    pts = {"learned gates": [(dens[n], ret[n]) for n in names]}
    c1n = [n for n in runs if n.startswith("c1fixed")]
    if c1n:
        pts["random fixed mask (C1)"] = [(dens[n], ret[n]) for n in c1n]
    if "fullcomm" in ret:
        pts["FullComm / NoComm"] = [([1.0] * len(ret["fullcomm"]), ret["fullcomm"]), ([0.0] * len(ret["nocomm"]), ret["nocomm"])]
    plots.xy_by_group(pts, "active link ratio", f"episode return ({a.tag} gates)", out / f"e2_return_vs_density_{a.tag}.png",
                      "E2 return vs link density (points = seeds)", xlim=(-0.02, 1.02))
    dd, rd = per_seed(summ_det, "active_link_ratio"), per_seed(summ_det, "episode_return")
    plots.xy_by_group({"learned gates (deterministic eval)": [(dd[n], rd[n]) for n in names]}, "active link ratio",
                      "episode return", out / "e2_return_vs_density_det.png", "E2, deterministic gates", xlim=(-0.02, 1.02))
    for n in names:
        extra = f"  loss/benefit {report['rq1'][n]['loss_fraction']['point']:.2f} -> {report['rq1'][n]['verdict']}" if "rq1" in report else ""
        print(f"{n:20s} density {fmt_ci(tab[n]['active_link_ratio'])}  return {fmt_ci(tab[n]['episode_return'])}{extra}")
    print(f"gate collapse: {collapse}\nresults in {out}")


if __name__ == "__main__":
    run_main(main)
