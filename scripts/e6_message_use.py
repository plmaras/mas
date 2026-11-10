"""E6 Message-use test (faithfulness): are messages used, or only aligned with events?

Open-loop (teacher-forced, receiver policy KL on the next action): F1 zero all messages,
F2 event-time vs random-time ablation (same number of active links), F3 swap messages with
another sender / episode, F4 targeted counterfactual (pre-registered direction test).
Closed-loop (full episodes): return drop under F1, F2-event, F2-random (rate matched), F3.

The event lift / MH log odds ratio on the x-axis come from E5's per-run association statistics.

    python scripts/e6_message_use.py --rho 1 0.25 0.1 --ls 0 0.03 --seeds 0 1 2 3 4
"""
from common import (association, eval_runs, finalize_args, gated_overrides, gname, parser, results_dir, run_main,
                    save_json, train_runs)

import json

import numpy as np

from analysis import plots
from analysis.faithfulness import faithfulness_scores
from analysis.stats import summarize_seeds
from core.rollout import Interv
from core.utils import load_checkpoint
from eval import evaluate_run, load_logs


def closed_loop(run_dir, logs, base_return, det: bool):
    act = (logs["g"] > 0) & logs["mask"]
    rate = float((act & logs["E"]).sum() / max(act.sum(), 1))  # share of active links at event times
    res = {}
    for name, iv in (("F1_zero", Interv(zero_messages=True)), ("F2_event", Interv(ablate="event")),
                     ("F2_random", Interv(ablate="random", ablate_rate=rate)), ("F3_shuffle", Interv(shuffle_senders=True))):
        s = evaluate_run(run_dir, tag=f"e6_{name}_{'det' if det else 'stoch'}", save=False, deterministic=det, interv=iv)
        res[name] = dict(return_drop=base_return - s["episode_return"], success=s["success_rate"])
    res["random_ablation_rate"] = rate
    return res


def main():
    ap = parser(__doc__)
    ap.add_argument("--rho", type=float, nargs="+", default=[1.0, 0.25, 0.1])
    ap.add_argument("--ls", type=float, nargs="+", default=[0.0, 0.03])
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e6")

    conds = {gname(r, ls): gated_overrides(r, ls) for r in a.rho for ls in a.ls}
    conds["fullcomm"] = ["comm=fullcomm"]
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)
    det = a.tag == "det"

    report, pts = {}, []
    for name, dirs in runs.items():
        per = []
        for k, d in enumerate(dirs):
            cache = d / f"faithfulness_{a.tag}.json"
            if cache.exists() and not a.retrain:
                r = json.loads(cache.read_text())
            else:
                cfg, params = load_checkpoint(d)
                logs, _ = load_logs(d / f"eval_{a.tag}.npz")
                r = faithfulness_scores(cfg, params, logs, seed=k)
                r["closed_loop"] = closed_loop(d, logs, summ[name][k]["episode_return"], det)
                save_json(cache, r)
            ev = association(d, a, seed=k).get("events") or {}
            r["lift"] = ev.get("lift", 1.0 if summ[name][k]["active_link_ratio"] >= 1 else np.nan)
            r["log_or"] = ev.get("log_or_time_dist", np.nan)
            per.append(r)
            pts.append(dict(name=name, lift=r["lift"], log_or=r["log_or"], kl_f1=r["F1"]["kl_mean_affected"],
                            density=summ[name][k]["active_link_ratio"]))
        report[name] = dict(
            F1_kl=summarize_seeds([p["F1"]["kl_mean_affected"] for p in per]),
            F2_event_kl_per_link=summarize_seeds([p["F2"]["event"].get("kl_per_ablated_link", np.nan) for p in per]),
            F2_random_kl_per_link=summarize_seeds([p["F2"]["random"].get("kl_per_ablated_link", np.nan) for p in per]),
            F3_other_sender_kl=summarize_seeds([p["F3"]["other_sender"]["kl_mean_affected"] for p in per]),
            F3_other_episode_kl=summarize_seeds([p["F3"]["other_episode"]["kl_mean_affected"] for p in per]),
            F4_match_rate=summarize_seeds([p["F4"].get("match_rate", np.nan) for p in per]),
            closed_loop={k: summarize_seeds([p["closed_loop"][k]["return_drop"] for p in per])
                         for k in ("F1_zero", "F2_event", "F2_random", "F3_shuffle")},
            consistency_max_abs=[p["consistency_max_abs"] for p in per],
            lift=summarize_seeds([p["lift"] for p in per]),
            log_or_time_dist=summarize_seeds([p["log_or"] for p in per]),
        )
    save_json(out / "e6_report.json", dict(per_model=report, per_run=pts, gate_eval=a.tag,
                                           readout="high lift + ~zero KL => decorative messages; low lift + large KL => "
                                                   "useful but event-unrelated messages; F4 match rate chance = 0.5"))
    groups = {}
    for p in pts:
        groups.setdefault(p["name"], []).append(p)
    names = list(groups)
    series = {n: [([p["lift"] for p in groups[n]], [p["kl_f1"] for p in groups[n]])] for n in names[:8]}
    plots.xy_by_group(series, "event lift (E5)", "F1 receiver KL (zero messages)", out / "e6_faithfulness_vs_lift.png",
                      "E6 message use vs event association", connect=False)
    plots.points_with_ci({n: [p["kl_f1"] for p in groups[n]] for n in names}, "KL(pi || pi with messages zeroed)",
                         out / "e6_f1_kl.png", "E6 F1 receiver KL per model")
    for n, r in report.items():
        print(f"{n:22s} F1 KL {r['F1_kl']['mean']:.4f}  F2 ev/rand {r['F2_event_kl_per_link']['mean']:.4f}/"
              f"{r['F2_random_kl_per_link']['mean']:.4f}  F4 match {r['F4_match_rate']['mean']:.2f}  "
              f"drop F1 {r['closed_loop']['F1_zero']['mean']:.2f}")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
