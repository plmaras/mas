"""Per-run event-association statistics (E5 / RQ3), v2 design.

Primary statistic: the Mantel-Haenszel log odds ratio between "gate open" and the event,
stratified by time step x sender-receiver distance bin (removes the time-in-episode confound
found in v1, where gates were open early and events happened early). Null distributions:
  * episode shuffle: whole gate sequences re-paired with another episode's events (same link,
    same time steps) -> keeps every gate series' time profile and persistence;
  * circular time shift within (episode, link) (the spec's null), reported for comparison.
Also reported: pooled phi, lift and AUROC (spec), and phi within time strata.
One number per run; aggregate across seeds afterwards (never pool link-timesteps).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

DIST_EDGES = [0.0, 1.5, 2.5, 3.5, 5.0, 7.0, 1.0e9]  # link distance in units of obs_radius (cells at R=1)


def association_run(run_dir, tag: str = "stoch", n_perm: int = 1000, overwrite: bool = False, seed: int = 0,
                    edges=None) -> dict:
    """Cached as ``association_<tag>.json`` in the run directory. Returns ``{"events": None}``
    when the gates are constant (density 0 or 1: association undefined)."""
    from analysis.events import event_stats, mantel_haenszel, stratum_counts, time_strata
    from analysis.metrics import distance_bins
    from analysis.null import circular_shift_null, compare_to_null, episode_shuffle_null
    from eval import load_logs

    run_dir = Path(run_dir)
    cache = run_dir / f"association_{tag}.json"
    if cache.exists() and not overwrite:
        return json.loads(cache.read_text())
    logs, summary = load_logs(run_dir / f"eval_{tag}.npz")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    g, E, mask = logs["g"], logs["E"], logs["mask"]
    act = (g > 0) & mask
    T = g.shape[1]
    out = {k: summary.get(k) for k in ("episode_return", "success_rate", "active_link_ratio", "gate_switch_rate",
                                       "mean_run_length", "event_rate")}
    out["density_by_t"] = (act.sum((0, 2, 3)) / np.maximum(mask.sum((0, 2, 3)), 1)).tolist()
    out["event_by_t"] = ((E & mask).sum((0, 2, 3)) / np.maximum(mask.sum((0, 2, 3)), 1)).tolist()
    dens = float(act.sum() / max(mask.sum(), 1))
    if dens <= 0 or dens >= 1 or not (E & mask).any() or (E | ~mask).all():
        out["events"] = None
        cache.write_text(json.dumps(out, default=float))
        return out
    edges = list(edges or DIST_EDGES)
    nb = len(edges) - 1
    ts = time_strata(g.shape)
    db = distance_bins(logs["positions"], edges, float(cfg.env.obs_radius))
    st_td = ts * nb + db
    obs = event_stats(g, E, mask, ts, T, alpha=logs["alpha"])
    mh_pooled = mantel_haenszel(stratum_counts(g, E, mask, np.zeros_like(ts), 1))
    mh_t = mantel_haenszel(stratum_counts(g, E, mask, ts, T))
    mh_td = mantel_haenszel(stratum_counts(g, E, mask, st_td, T * nb))
    shuf = episode_shuffle_null(g, E, mask, st_td, T * nb, n_perm=n_perm, seed=seed)
    null_td = mantel_haenszel(shuf["counts"])["log_or"]
    null_pooled = mantel_haenszel(shuf["counts"].sum(-1, keepdims=True))["log_or"]
    circ = circular_shift_null(g, E, mask, n_shifts=n_perm, seed=seed)
    out["events"] = dict(
        phi=obs["phi"], cramers_v=obs.get("cramers_v"), lift=obs["lift"], auroc=obs.get("auroc"),
        auroc_alpha=obs.get("auroc_alpha"), phi_time_stratified=obs["phi_stratified"],
        log_or_pooled=float(mh_pooled["log_or"]), log_or_time=float(mh_t["log_or"]),
        log_or_time_dist=float(mh_td["log_or"]), log_or_time_dist_se=mh_td.get("se"), distance_edges=edges,
        shuffle_vs_log_or_time_dist=compare_to_null(float(mh_td["log_or"]), null_td),
        shuffle_vs_log_or_pooled=compare_to_null(float(mh_pooled["log_or"]), null_pooled),
        circular_vs_phi=compare_to_null(obs["phi"], circ["null_pooled"]["phi"]),
        null_log_or_time_dist_samples=np.asarray(null_td).tolist(), n_perm=n_perm,
    )
    cache.write_text(json.dumps(out, default=float))
    return out
