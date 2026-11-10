"""E3 Persistence at matched density (RQ2): does encouraging links to persist change performance
independently of link count?

Grid: target density rho in {large, mid, small} x lambda_s in {0, mid, high}. The density is
held by the PI controller, and lambda_s ramps in only after the target is reached (v2 fix: with
lambda_s on from the start, gates stay all-on or collapse, so densities cannot be matched).

``--pilot`` applies the v2 rule for choosing lambda_s (1 seed, rho = 0.25): "mid" is the smallest
lambda_s that at least halves the gate switch rate of lambda_s = 0, "high" the smallest that cuts
it by >= 4x, both keeping density within 0.05 of the target.

Analysis (pre-specified in the v2 plan): at each rho, lambda_s > 0 vs lambda_s = 0, paired by
seed (same seed = same initialisation and warm-up); paired t-test on return, Holm over all
comparisons; manipulation checks (switch rate down, run length up, |delta density| <= 0.05);
pooled over rho (and averaged within seed first, the conservative version); a density-adjusted
version (return - slope * delta density). Metrics reported separately: gate-driven switch rate,
mask-driven switch rate (neighbour-set changes; nonzero only once K < N - 1), mean run length.

    python scripts/e3_persistence_grid.py --pilot
    python scripts/e3_persistence_grid.py --rho 0.5 0.25 0.1 --ls 0 0.01 0.03 --seeds 0 1 2 3 4
"""
from common import (eval_runs, finalize_args, gated_overrides, gname, holm, parser, results_dir, run_main, save_json,
                    t_ci, table, train_runs, ttest_1samp, write_table_csv)

import numpy as np

from analysis import plots

KEYS = ["episode_return", "success_rate", "final_burned_frac", "active_link_ratio", "gate_switch_rate",
        "mask_switch_rate", "mean_run_length", "event_rate"]
PILOT_LS = [0.0, 0.003, 0.01, 0.03, 0.1]


def pilot(a):
    a.seeds = a.seeds[:1]
    a.exp_name = "pilot_" + a.exp_name
    a = finalize_args(a)
    out = results_dir(a, "e3_pilot")
    conds = {gname(0.25, ls): gated_overrides(0.25, ls) for ls in PILOT_LS}
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    rows = {n: dict(ls=ls, density=summ[n][0]["active_link_ratio"], switch=summ[n][0]["gate_switch_rate"],
                    ret=summ[n][0]["episode_return"]) for n, ls in zip(conds, PILOT_LS)}
    sw0 = rows[gname(0.25, 0.0)]["switch"]
    ok = [r for r in rows.values() if r["ls"] > 0 and abs(r["density"] - 0.25) <= 0.05]
    mid = min((r["ls"] for r in ok if r["switch"] <= sw0 / 2), default=None)
    high = min((r["ls"] for r in ok if r["switch"] <= sw0 / 4 and (mid is None or r["ls"] > mid)), default=None)
    if high is None and ok:
        high = max(r["ls"] for r in ok)
    dec = dict(rule="mid: smallest ls halving the switch rate; high: smallest cutting it >= 4x; density within 0.05",
               rows=rows, switch_ls0=sw0, ls_mid=mid, ls_high=high)
    save_json(out / "e3_pilot_decisions.json", dec)
    for n, r in rows.items():
        print(f"{n:22s} density {r['density']:.3f} switch {r['switch']:.4f} return {r['ret']:.3f}")
    print(f"-> lambda_s mid = {mid}, high = {high}   (use: --ls 0 {mid} {high})\nresults in {out}")


def rq2(rows_by, rhos, lss):
    """Paired-by-seed persistence comparisons. ``rows_by[(rho, ls)] = {seed: summary}``."""
    base = lss[0]
    xs, ys = [], []
    for rho in rhos:  # return-density slope from the lambda_s = 0 runs (density-adjusted version)
        for s in rows_by[(rho, base)].values():
            xs.append(s["active_link_ratio"])
            ys.append(s["episode_return"])
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(np.round(xs, 3))) > 1 else 0.0
    per, p_raw, p_adj, pooled = {}, {}, {}, {}
    for ls in lss[1:]:
        pooled_d, by_seed = [], {}
        for rho in rhos:
            b, t = rows_by[(rho, base)], rows_by[(rho, ls)]
            seeds = sorted(set(b) & set(t))
            d = {k: np.array([t[s][k] - b[s][k] for s in seeds]) for k in
                 ("episode_return", "success_rate", "active_link_ratio", "gate_switch_rate", "mean_run_length")}
            adj = d["episode_return"] - slope * d["active_link_ratio"]
            key = f"rho{rho:g}_ls{ls:g}"
            per[key] = dict(rho=rho, ls=ls, seeds=seeds, p=ttest_1samp(d["episode_return"]), p_adjusted=ttest_1samp(adj),
                            density_matched=bool(abs(np.mean(d["active_link_ratio"])) <= 0.05),
                            persistence_increased=bool(np.mean(d["gate_switch_rate"]) < 0 < np.mean(d["mean_run_length"])),
                            d_return_adjusted=t_ci(adj), **{f"d_{k}": t_ci(v) for k, v in d.items()})
            p_raw[key], p_adj[key] = per[key]["p"], per[key]["p_adjusted"]
            pooled_d += d["episode_return"].tolist()
            for s, v in zip(seeds, d["episode_return"]):
                by_seed.setdefault(s, []).append(v)
        seed_means = [np.mean(v) for v in by_seed.values()]
        pooled[f"ls{ls:g}"] = dict(d_return=t_ci(pooled_d), p=ttest_1samp(pooled_d),
                                   seed_averaged=dict(d_return=t_ci(seed_means), p=ttest_1samp(seed_means)))
    for k, p in holm(p_raw).items():
        per[k]["holm_p"] = p
    for k, p in holm(p_adj).items():
        per[k]["holm_p_adjusted"] = p
    return dict(per=per, pooled=pooled, density_slope=slope)


def main():
    ap = parser(__doc__)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.25, 0.1], help="target densities")
    ap.add_argument("--ls", type=float, nargs="+", default=[0.0, 0.01, 0.03], help="lambda_s: 0, mid, high")
    ap.add_argument("--pilot", action="store_true", help="choose lambda_s mid/high (1 seed)")
    a = ap.parse_args()
    if a.pilot:
        return pilot(a)
    a = finalize_args(a)
    out = results_dir(a, "e3")
    conds = {gname(r, ls): gated_overrides(r, ls) for r in a.rho for ls in a.ls}
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    summ_det = eval_runs(runs, tag="det", save=False)  # robustness check: summaries only
    tab = table(summ, KEYS)
    write_table_csv(out / f"e3_table_{a.tag}.csv", tab, KEYS)
    write_table_csv(out / "e3_table_det.csv", table(summ_det, KEYS), KEYS)

    rows_by, rows = {}, []
    mats = {k: np.full((len(a.rho), len(a.ls)), np.nan) for k in ("return", "density", "gate switch rate", "run length")}
    for i, r in enumerate(a.rho):
        for j, ls in enumerate(a.ls):
            n = gname(r, ls)
            rows_by[(r, ls)] = {s: summ[n][k] for k, s in enumerate(a.seeds)}
            mats["return"][i, j] = tab[n]["episode_return"]["mean"]
            mats["density"][i, j] = tab[n]["active_link_ratio"]["mean"]
            mats["gate switch rate"][i, j] = tab[n]["gate_switch_rate"]["mean"]
            mats["run length"][i, j] = tab[n]["mean_run_length"]["mean"]
            for s, x in zip(a.seeds, summ[n]):
                rows.append(dict(name=n, rho=r, ls=ls, seed=s, density=x["active_link_ratio"], switch=x["gate_switch_rate"],
                                 run_length=x["mean_run_length"], mask_switch=x["mask_switch_rate"],
                                 **{"return": x["episode_return"]}))
    res = rq2(rows_by, a.rho, a.ls)
    res_det = rq2({k: {s: summ_det[gname(*k)][i] for i, s in enumerate(a.seeds)} for k in rows_by}, a.rho, a.ls)
    save_json(out / "e3_report.json", dict(
        grid_rho=a.rho, grid_ls=a.ls, gate_eval=a.tag, table=tab, per_run=rows, rq2=res, rq2_deterministic_gates=res_det,
        readout="return differing between lambda_s settings at matched density => persistence matters independently of "
                "link count; only switch rate / run length differing => it changes stability, not performance"))
    lab = lambda v: [f"{x:g}" for x in v]
    plots.heatmaps(mats, lab(a.rho), lab(a.ls), "target density", "lambda_s", out / "e3_heatmaps.png", fmt="{:.2f}")
    plots.scatter_colored([r["switch"] for r in rows], [r["return"] for r in rows], [r["density"] for r in rows],
                          "gate switch rate", "episode return", "active link ratio", out / "e3_return_vs_switch.png",
                          "E3 return vs switch rate (color = density, one point per seed)")
    for k, v in res["per"].items():
        print(f"{k:16s} d_return {v['d_episode_return']['mean']:+.3f} [{v['d_episode_return']['lo']:+.3f}, "
              f"{v['d_episode_return']['hi']:+.3f}] holm p {v['holm_p']:.3g}  d_density {v['d_active_link_ratio']['mean']:+.3f} "
              f"d_switch {v['d_gate_switch_rate']['mean']:+.4f}  matched {v['density_matched']}")
    for k, v in res["pooled"].items():
        print(f"pooled {k}: d_return {v['d_return']['mean']:+.3f} p {v['p']:.3g}; seed-averaged p {v['seed_averaged']['p']:.3g}")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
