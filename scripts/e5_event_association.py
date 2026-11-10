"""E5 Event association (RQ3): are active links associated with the task event more than
chance, and does this differ for persistence-penalised models (at matched density)?

Event E[t,i,j] = sender i sees a burning cell outside receiver j's view (permitted links only).

Per run (analysis/association.py):
  * primary: Mantel-Haenszel log odds ratio (gate open vs event) within time-step x distance
    strata (v2: the pooled statistic is confounded by time in the episode, because gates and
    events both change over an episode);
  * null: episode shuffle (whole gate sequences re-paired with other episodes' events;
    keeps time profile and persistence), 1000 permutations; also the spec's circular shift;
  * also: pooled phi, Cramer's V, lift, AUROC (binary g and rank-AUROC of alpha).
Across seeds: mean log OR with 95 % t-CI; one-sample t-test vs 0, Holm over conditions;
persistence: paired-by-seed difference of log OR (lambda_s > 0 minus 0) at each density, Holm.
Wording: association, not causation.

    python scripts/e5_event_association.py --rho 0.5 0.25 0.1 --ls 0 0.01 0.03 --seeds 0 1 2 3 4
"""
from common import (association, eval_runs, finalize_args, gated_overrides, gname, holm, parser, results_dir,
                    run_main, save_json, t_ci, train_runs, ttest_1samp)

import numpy as np

from analysis import plots

STATS = ["log_or_time_dist", "log_or_time", "log_or_pooled", "phi", "phi_time_stratified", "cramers_v", "lift",
         "auroc", "auroc_alpha"]


def main():
    ap = parser(__doc__)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    ap.add_argument("--ls", type=float, nargs="+", default=[0.0, 0.01, 0.03])
    ap.add_argument("--n-perm", type=int, default=1000)
    a = finalize_args(ap.parse_args())
    n_perm = min(a.n_perm, 50) if a.quick else a.n_perm
    out = results_dir(a, "e5")
    conds = {gname(r, ls): gated_overrides(r, ls) for r in a.rho for ls in a.ls}
    runs = train_runs(conds, a)
    eval_runs(runs, tag=a.tag)

    per_run, report, pvals = {}, {}, {}
    for name, dirs in runs.items():
        rs = [association(d, a, seed=k, n_perm=n_perm) for k, d in enumerate(dirs)]
        per_run[name] = rs
        ev = [r.get("events") or {} for r in rs]
        agg = {s: t_ci([e.get(s, np.nan) for e in ev]) for s in STATS}
        agg["shuffle_p_per_seed"] = [(e.get("shuffle_vs_log_or_time_dist") or {}).get("p_value") for e in ev]
        agg["shuffle_excess_per_seed"] = [(e.get("shuffle_vs_log_or_time_dist") or {}).get("excess") for e in ev]
        agg["circular_p_per_seed"] = [(e.get("circular_vs_phi") or {}).get("p_value") for e in ev]
        agg["n_runs_negative"] = int(sum(1 for e in ev if e.get("log_or_time_dist", 0) < 0))
        agg["n_runs_shuffle_p05"] = int(sum(1 for p in agg["shuffle_p_per_seed"] if p is not None and p <= 0.05))
        agg["density"] = t_ci([r["active_link_ratio"] for r in rs])
        agg["switch_rate"] = t_ci([r["gate_switch_rate"] for r in rs])
        agg["p"] = ttest_1samp([e.get("log_or_time_dist", np.nan) for e in ev])
        pvals[name] = agg["p"]
        report[name] = agg
        if ev and ev[0].get("null_log_or_time_dist_samples"):
            plots.null_histogram(ev[0]["null_log_or_time_dist_samples"], ev[0]["log_or_time_dist"],
                                 "MH log odds ratio (time x distance strata)", out / f"e5_null_{name}_seed{a.seeds[0]}.png",
                                 f"E5 null vs observed: {name}", null_label="episode-shuffle null")
    for n, p in holm(pvals).items():
        report[n]["holm_p"] = p

    persistence, p_pers = {}, {}
    for r in a.rho:
        base = {s: (x.get("events") or {}).get("log_or_time_dist", np.nan) for s, x in zip(a.seeds, per_run[gname(r, a.ls[0])])}
        for ls in a.ls[1:]:
            t = {s: (x.get("events") or {}).get("log_or_time_dist", np.nan) for s, x in zip(a.seeds, per_run[gname(r, ls)])}
            d = np.array([t[s] - base[s] for s in a.seeds])
            key = f"rho{r:g}_ls{ls:g}"
            persistence[key] = dict(d_log_or=t_ci(d), p=ttest_1samp(d),
                                    d_density=t_ci([x["active_link_ratio"] - y["active_link_ratio"] for x, y in
                                                    zip(per_run[gname(r, ls)], per_run[gname(r, a.ls[0])])]))
            p_pers[key] = persistence[key]["p"]
    for k, p in holm(p_pers).items():
        persistence[k]["holm_p"] = p
    all_lo = [((x.get("events") or {}).get("log_or_time_dist", np.nan)) for rs in per_run.values() for x in rs]
    by_seed = [np.nanmean([(per_run[n][k].get("events") or {}).get("log_or_time_dist", np.nan) for n in runs])
               for k in range(len(a.seeds))]
    summary = dict(n_runs=len(all_lo), n_negative=int(np.sum(np.asarray(all_lo) < 0)),
                   n_positive=int(np.sum(np.asarray(all_lo) > 0)),
                   seed_averaged=dict(log_or=t_ci(by_seed), p=ttest_1samp(by_seed)))
    save_json(out / "e5_report.json", dict(per_condition=report, persistence=persistence, overall=summary, gate_eval=a.tag,
                                           per_run={n: [{k: v for k, v in (x.get("events") or {}).items()
                                                         if not k.startswith("null_")} for x in rs] for n, rs in per_run.items()},
                                           note="association, not causation; one statistic per seed; Holm within families"))
    pts = [(n, x) for n, rs in per_run.items() for x in rs if x.get("events")]
    if pts:
        plots.scatter_colored([x["gate_switch_rate"] for _, x in pts], [x["events"]["log_or_time_dist"] for _, x in pts],
                              [x["active_link_ratio"] for _, x in pts], "gate switch rate",
                              "MH log OR (time x distance)", "active link ratio", out / "e5_logor_vs_switch.png",
                              "E5 association vs switch rate (color = density, one point per seed)")
        plots.points_with_ci({n: [(x.get("events") or {}).get("log_or_time_dist", np.nan) for x in rs] for n, rs in per_run.items()},
                             "MH log odds ratio (time x distance)", out / "e5_logor_per_model.png",
                             "E5 association per model (points = seeds)")
    for n, r in report.items():
        lo = r["log_or_time_dist"]
        print(f"{n:20s} logOR {lo['mean']:+.3f} [{lo['lo']:+.3f}, {lo['hi']:+.3f}]  holm p {r['holm_p']:.3g}  "
              f"neg {r['n_runs_negative']}/{len(a.seeds)}  shuffle p<=.05 {r['n_runs_shuffle_p05']}/{len(a.seeds)}  "
              f"phi {r['phi']['mean']:+.3f}  lift {r['lift']['mean']:.2f}")
    for k, v in persistence.items():
        print(f"persistence {k}: d logOR {v['d_log_or']['mean']:+.3f} holm p {v['holm_p']:.3g}")
    print(f"overall: {summary['n_negative']} negative / {summary['n_positive']} positive of {summary['n_runs']}; "
          f"seed-averaged log OR {summary['seed_averaged']['log_or']['mean']:+.3f} (p {summary['seed_averaged']['p']:.3g})")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
