"""E4 Matched-mask controls: does learned gating matter beyond having fewer links?

C1 random fixed mask at the learned density
C2 random time-varying (Markov) mask matched in density and switch rate
C3 distance-rule mask matched in density
C4 learned gates with message contents zeroed
Each control is run (a) at evaluation time on a trained FullComm model (C4: on the learned
model) and (b) trained from scratch with the mask fixed. Matching uses the learned model of
the same seed for (a) and the across-seed mean for (b).

The learned model is the target-density gate of E2/E3 (``--rho``, ``--ls``); its achieved
density and switch rate (per seed) set the controls' parameters.

    python scripts/e4_matched_controls.py --rho 0.25 --ls 0 --seeds 0 1 2 3 4
"""
from common import (eval_runs, finalize_args, gated_overrides, gname, holm, parser, per_seed, results_dir, run_main,
                    save_json, train_runs, welch)

import numpy as np

from analysis import plots
from analysis.metrics import link_distance
from analysis.stats import summarize_seeds
from core.rollout import Interv
from eval import evaluate_run, load_logs


def matched_distance(run_dir, density, tag):
    """Distance d such that P(|p_i - p_j| < d | link permitted) = density, from the learned model's
    logs. Grid distances are discrete, so d is placed between the two neighbouring values."""
    logs, _ = load_logs(run_dir / f"eval_{tag}.npz")
    d = np.sort(link_distance(logs["positions"])[logs["mask"]])
    q = float(np.quantile(d, np.clip(density, 0, 1)))
    above = d[d > q]
    return float((q + above[0]) / 2) if above.size else q + 0.5


def main():
    ap = parser(__doc__)
    ap.add_argument("--rho", type=float, default=0.25, help="target density of the learned model")
    ap.add_argument("--ls", type=float, default=0.0, help="lambda_s of the learned model")
    ap.add_argument("--skip-scratch", action="store_true", help="only evaluation-time controls")
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e4")

    learned = gname(a.rho, a.ls)
    runs = train_runs({learned: gated_overrides(a.rho, a.ls), "fullcomm": ["comm=fullcomm"]}, a)
    summ = eval_runs(runs, tag=a.tag)
    det = a.tag == "det"
    rho = per_seed(summ, "active_link_ratio")[learned]
    sw = per_seed(summ, "gate_switch_rate")[learned]
    dists = [matched_distance(d, r, a.tag) for d, r in zip(runs[learned], rho)]

    res = {"learned": per_seed(summ, "episode_return")[learned], "fullcomm (all on)": per_seed(summ, "episode_return")["fullcomm"]}
    ev = {"C1 random fixed": [], "C2 random Markov": [], "C3 distance rule": [], "C4 zeroed messages": []}
    for k, (fd, ld) in enumerate(zip(runs["fullcomm"], runs[learned])):
        tagp = f"{learned}_s{a.seeds[k]}"
        c1 = evaluate_run(fd, tag=f"e4_c1_{tagp}", save=False, comm_overrides=dict(gate_mode="random_fixed", density=rho[k]))
        c2 = evaluate_run(fd, tag=f"e4_c2_{tagp}", save=False,
                          comm_overrides=dict(gate_mode="random_markov", density=rho[k], switch_rate=sw[k]))
        c3 = evaluate_run(fd, tag=f"e4_c3_{tagp}", save=False, comm_overrides=dict(gate_mode="hand_rule", hand_rule_d=dists[k]))
        c4 = evaluate_run(ld, tag=f"e4_c4_{a.tag}", save=False, deterministic=det, interv=Interv(zero_messages=True))
        for name, s in zip(ev, (c1, c2, c3, c4)):
            ev[name].append(s)
    res.update({f"{n} [eval]": [s["episode_return"] for s in ss] for n, ss in ev.items()})
    match = {n: dict(density=[s["active_link_ratio"] for s in ss], switch=[s["gate_switch_rate"] for s in ss]) for n, ss in ev.items()}

    if not a.skip_scratch:
        r_, s_, d_ = float(np.mean(rho)), float(np.mean(sw)), float(np.mean(dists))
        scratch = {
            "C1_scratch": ["comm=random_fixed", f"comm.density={r_:.4f}"],
            "C2_scratch": ["comm=random_markov", f"comm.density={r_:.4f}", f"comm.switch_rate={s_:.4f}"],
            "C3_scratch": ["comm=handrule", f"comm.hand_rule_d={d_:.4f}"],
            "C4_scratch": gated_overrides(a.rho, a.ls) + ["comm.zero_messages=true"],
        }
        sruns = train_runs({f"{k}_{learned}": v for k, v in scratch.items()}, a)
        ssum = eval_runs(sruns, tag=a.tag)
        for k in scratch:
            res[f"{k.split('_')[0]} [scratch]"] = per_seed(ssum, "episode_return")[f"{k}_{learned}"]
            match[f"{k} [scratch]"] = dict(density=per_seed(ssum, "active_link_ratio")[f"{k}_{learned}"],
                                           switch=per_seed(ssum, "gate_switch_rate")[f"{k}_{learned}"])

    diffs = {n: summarize_seeds(np.array(res["learned"]) - np.array(v)) for n, v in res.items() if n != "learned"}
    # eval-time controls share the seed with the learned model (paired); scratch controls are independent (Welch)
    from common import ttest_1samp
    pvals = {n: (ttest_1samp(np.array(res["learned"]) - np.array(v)) if "[eval]" in n or n.startswith("fullcomm")
                 else welch(res["learned"], v)) for n, v in res.items() if n != "learned"}
    hp = holm(pvals)
    tests = {n: dict(p=p, holm_p=hp[n]) for n, p in pvals.items()}
    save_json(out / "e4_report.json", dict(learned=learned, gate_eval=a.tag, learned_density=rho, learned_switch=sw,
                                           matched_distance=dists, returns={n: summarize_seeds(v) for n, v in res.items()},
                                           learned_minus_control=diffs, tests=tests, achieved_matching=match,
                                           readout="learned > C1 and C3 => learning found useful links; learned == C1 => "
                                                   "it only removed links; C4 near learned => messages carry nothing"))
    plots.points_with_ci(res, "episode return", out / "e4_controls.png", f"E4 controls at matched density ({learned})")
    for n, s in diffs.items():
        print(f"learned - {n:26s}: {s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
