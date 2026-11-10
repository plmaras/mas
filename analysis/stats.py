"""Across-seed aggregation: one statistic per seed, then bootstrap CIs and IQM (rliable).

Never pool link-timesteps across seeds as independent samples; always reduce to one number
per seed first. With 5 seeds the IQM is fragile, so plots always show the raw per-seed points.
"""
from __future__ import annotations

import numpy as np

try:
    from rliable import library as rly
    from rliable import metrics as rl_metrics

    HAVE_RLIABLE = True
except Exception:  # pragma: no cover
    HAVE_RLIABLE = False


def iqm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if HAVE_RLIABLE:
        return float(rl_metrics.aggregate_iqm(x[:, None]))
    from scipy.stats import trim_mean

    return float(trim_mean(x, 0.25))


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of ``stat`` over seeds."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(stat(x)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([stat(row) for row in x[idx]]) if stat is not np.mean else x[idx].mean(1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(stat(x)), float(lo), float(hi)


def rliable_iqm_ci(scores_by_method: dict, reps: int = 2000):
    """Stratified-bootstrap IQM CIs via rliable. ``scores_by_method[name] = [n_seeds]``."""
    if not HAVE_RLIABLE:
        return {k: (iqm(v),) + bootstrap_ci(v, iqm, reps)[1:] for k, v in scores_by_method.items()}
    data = {k: np.asarray(v, float)[:, None] for k, v in scores_by_method.items()}
    point, interval = rly.get_interval_estimates(data, lambda x: np.array([rl_metrics.aggregate_iqm(x)]), reps=reps)
    return {k: (float(point[k][0]), float(interval[k][0, 0]), float(interval[k][1, 0])) for k in data}


def summarize_seeds(values, n_boot: int = 10000) -> dict:
    v = np.asarray(values, float)
    mean, lo, hi = bootstrap_ci(v, np.mean, n_boot)
    return dict(mean=mean, ci_lo=lo, ci_hi=hi, iqm=iqm(v), n=int(np.isfinite(v).sum()), values=v.tolist())
