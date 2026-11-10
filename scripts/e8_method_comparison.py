"""E8 Method comparison under one backbone (MAPPO) and one training budget.

Methods: NoComm, RL-trained continuous messages (FullComm), AE-Comm (autoencoder-grounded
messages, RL gradient blocked), a language-style variant (fixed 9-token vocabulary aligned to
human-readable compass labels), the gated model, and optionally IC3Net-style gating
(REINFORCE, one gate per sender).
Metrics: return, probe R^2 (E7), readability, faithfulness (E6 F1 KL), bits per step,
cross-play (sender from seed s+1, receiver from seed s).
Caution: language-grounded methods score well on readability by construction, so weigh
faithfulness more heavily.

The gated model and IC3Net-style gating are trained to the SAME target link density (v2: in
v1 they used different lambda_c and were not comparable). Cross-play pairs the receiver of seed
s with the sender of seed s+1.

    python scripts/e8_method_comparison.py --rho 0.25 --seeds 0 1 2 3 4 --with-ic3net
"""
from common import (eval_runs, finalize_args, gated_overrides, gname, parser, per_seed, results_dir, run_main,
                    save_json, train_runs)

import json

import numpy as np

from analysis import plots
from analysis.faithfulness import f1_zero_messages, make_receiver_fn
from analysis.probe import probe_all
from analysis.stats import summarize_seeds
from core.utils import load_checkpoint
from eval import evaluate, load_logs


def bits_per_message(cfg):
    if cfg.comm.zero_messages:
        return 0.0
    if cfg.comm.message_type == "language":
        return float(np.log2(cfg.comm.vocab_size))
    return float(cfg.comm.msg_dim * 32)  # nominal: float32 per dimension


def main():
    ap = parser(__doc__)
    ap.add_argument("--rho", type=float, default=0.25, help="target link density of the gated models")
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--with-ic3net", action="store_true")
    a = finalize_args(ap.parse_args())
    out = results_dir(a, "e8")

    gated = gname(a.rho, a.ls)
    conds = {"nocomm": ["comm=nocomm"], "fullcomm": ["comm=fullcomm"], "aecomm": ["comm=aecomm"],
             "language": ["comm=language"], gated: gated_overrides(a.rho, a.ls)}
    if a.with_ic3net:
        conds[gname(a.rho, a.ls, "ic3net")] = gated_overrides(a.rho, a.ls, comm="ic3net")
    runs = train_runs(conds, a)
    summ = eval_runs(runs, tag=a.tag)

    rows = {}
    for name, dirs in runs.items():
        per = []
        for k, d in enumerate(dirs):
            cache = d / f"e8_metrics_{a.tag}.json"
            if cache.exists() and not a.retrain:
                per.append(json.loads(cache.read_text()))
                continue
            cfg, params = load_checkpoint(d)
            logs, meta = load_logs(d / f"eval_{a.tag}.npz")
            pr = probe_all(logs, meta["obs_radius"], seed=k)
            f1 = f1_zero_messages(make_receiver_fn(cfg, params), logs)
            r = dict(probe_r2=pr["private_landmark_offset"].get("linear", np.nan),
                     readability=pr["human_label"].get("linear", np.nan),
                     faithfulness_kl=f1["kl_mean_affected"],
                     bits_per_agent_step=summ[name][k]["links_per_agent_step"] * bits_per_message(cfg))
            if cfg.comm.message_type == "language":  # tokens are directly readable: no probe needed
                r["readability_direct"] = float((logs["messages"].argmax(-1) == logs["human_label"]).mean())
            # cross-play: messages from another seed's sender, receiver/policy from this seed
            if len(dirs) > 1:
                _, p_other = load_checkpoint(dirs[(k + 1) % len(dirs)])
                _, s_x = evaluate(cfg, params, sender_params=p_other["actor"], light=True, deterministic=(a.tag == "det"))
                r["crossplay_return"] = s_x["episode_return"]
                r["crossplay_drop"] = summ[name][k]["episode_return"] - s_x["episode_return"]
            save_json(cache, r)
            per.append(r)
        rows[name] = per

    metrics = ["probe_r2", "readability", "faithfulness_kl", "bits_per_agent_step", "crossplay_return", "crossplay_drop"]
    rep = {n: {m: summarize_seeds([p.get(m, np.nan) for p in per]) for m in metrics + ["readability_direct"]}
           for n, per in rows.items()}
    for n in rep:
        rep[n]["episode_return"] = summarize_seeds(per_seed(summ, "episode_return")[n])
    save_json(out / "e8_report.json", dict(per_method=rep, gate_eval=a.tag, caution="language-grounded methods are readable by construction; "
                                                                    "weigh faithfulness more heavily"))
    ret = per_seed(summ, "episode_return")
    names = list(runs)
    plots.xy_by_group({n: [([p["faithfulness_kl"] for p in rows[n]], ret[n])] for n in names},
                      "faithfulness: F1 receiver KL", "episode return", out / "e8_return_vs_faithfulness.png",
                      "E8 return vs faithfulness (one marker per method)", connect=False)
    read = lambda n, p: p.get("readability_direct", p["readability"])
    plots.xy_by_group({n: [([p["faithfulness_kl"] for p in rows[n]], [read(n, p) for p in rows[n]])] for n in names},
                      "faithfulness: F1 receiver KL", "readability (human-label accuracy)",
                      out / "e8_readability_vs_faithfulness.png", "E8 readability vs faithfulness", connect=False)
    plots.points_with_ci(ret, "episode return", out / "e8_return.png", "E8 return per method")
    for n, r in rep.items():
        print(f"{n:20s} return {r['episode_return']['mean']:8.3f}  KL {r['faithfulness_kl']['mean']:.4f}  "
              f"R2 {r['probe_r2']['mean']:.3f}  read {r['readability']['mean']:.3f}  bits {r['bits_per_agent_step']['mean']:.1f}  "
              f"xplay drop {r['crossplay_drop']['mean']:.3f}")
    print(f"results in {out}")


if __name__ == "__main__":
    run_main(main)
