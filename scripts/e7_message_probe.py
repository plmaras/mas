"""E7 Message content probe: do messages carry the sender's private information?

Linear and small-MLP probes from the message to (a) the offset (in cells, / obs_radius) of a
burning cell the receiver cannot see, (b) whether the sender sees a burning cell, (c) the
human-readable label (compass octant of the nearest visible fire, or "none").
Held-out split by episode and a shuffled-label baseline. Reported per model and per
link-density level (models from the E2 sweep). Interpret together with E6: information
present in a message does not show that it is used.

Note: the probe reads the message the sender PRODUCES, whether or not a link delivers it, so a
model with collapsed gates can still score (v2 correction): compare against NoComm and read
together with E6.

    python scripts/e7_message_probe.py --rhos 1 0.75 0.5 0.25 0.1 0.05 --seeds 0 1 2 3 4
"""
from common import (eval_runs, finalize_args, gated_overrides, gname, parser, per_seed, results_dir, run_main,
                    save_json, train_runs)

import json

import numpy as np

from analysis import plots
from analysis.probe import probe_all
from analysis.stats import summarize_seeds
from eval import load_logs


def main():
    ap = parser(__doc__)
    ap.add_argument("--rhos", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.1, 0.05])
    ap.add_argument("--probe-steps", type=int, default=500)
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e7")

    conds = {gname(r, 0.0): gated_overrides(r, 0.0) for r in a.rhos}
    conds["fullcomm"] = ["comm=fullcomm"]
    conds["nocomm"] = ["comm=nocomm"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)

    report, series = {}, {}
    for name, dirs in runs.items():
        per = []
        for k, d in enumerate(dirs):
            cache = d / f"probe_{a.tag}.json"
            if cache.exists() and not a.retrain:
                per.append(json.loads(cache.read_text()))
                continue
            logs, meta = load_logs(d / f"eval_{a.tag}.npz")
            r = probe_all(logs, meta["obs_radius"], seed=k, steps=a.probe_steps)
            save_json(cache, r)
            per.append(r)
        rep = {}
        for target, metric_keys in (("private_landmark_offset", ("linear", "mlp", "linear_shuffled", "mlp_shuffled")),
                                    ("sees_uncovered", ("linear", "mlp", "linear_shuffled", "majority_baseline")),
                                    ("human_label", ("linear", "mlp", "linear_shuffled", "majority_baseline"))):
            rep[target] = {m: summarize_seeds([p[target].get(m, np.nan) for p in per]) for m in metric_keys}
        rep["density"] = summarize_seeds(per_seed(summ, "active_link_ratio")[name])
        report[name] = rep
    save_json(out / "e7_report.json", dict(per_model=report, gate_eval=a.tag,
                                           note="R^2 for offsets, accuracy for labels; held out by episode"))

    gated = [n for n in runs if n.startswith("gated")]
    dens = per_seed(summ, "active_link_ratio")
    for target, lab in (("private_landmark_offset", "probe R² (held-out episodes)"), ("human_label", "probe accuracy")):
        series = {}
        for m in ("linear", "mlp", "linear_shuffled"):
            series[m] = [(dens[n], report[n][target][m]["values"]) for n in gated]
        plots.xy_by_group(series, "active link ratio", lab, out / f"e7_{target}_vs_density.png",
                          f"E7 probe: {target.replace('_', ' ')}")
    for n, r in report.items():
        t = r["private_landmark_offset"]
        print(f"{n:20s} density {r['density']['mean']:.2f}  R2 lin {t['linear']['mean']:.3f} mlp {t['mlp']['mean']:.3f} "
              f"shuffled {t['linear_shuffled']['mean']:.3f}")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
