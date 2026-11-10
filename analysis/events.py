"""Association between active gates and the task event (E5).

2x2 table on permitted link-timesteps (g = gate active, E = event):
              E=1    E=0
     g=1      n11    n10
     g=0      n01    n00
phi    = (n11 n00 - n10 n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))   (signed)
V      = |phi|                (Cramer's V of a 2x2 table)
lift   = P(g=1 | E=1) / P(g=1 | E=0)
AUROC  = 0.5 * (1 + TPR - FPR) for the binary score g (TPR = P(g=1|E=1), FPR = P(g=1|E=0));
         ``auroc_from_scores`` gives the rank-based AUROC for a continuous score (e.g. alpha).
All functions work on numpy or jax arrays (counts may carry leading batch dimensions).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def counts(g_active, E, valid, xp=np):
    g = g_active.astype(bool) & valid
    e = E.astype(bool) & valid
    v = valid.astype(bool)
    n11 = xp.sum(g & e)
    n10 = xp.sum(g & ~e & v)
    n01 = xp.sum(~g & e & v)
    n00 = xp.sum(~g & ~e & v)
    return n11, n10, n01, n00


def stats_from_counts(n11, n10, n01, n00, xp=np):
    n11, n10, n01, n00 = (xp.asarray(x, dtype=xp.float64 if xp is np else xp.float32) for x in (n11, n10, n01, n00))
    den = xp.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = xp.where(den > 0, (n11 * n00 - n10 * n01) / xp.where(den > 0, den, 1.0), 0.0)
    p_g_e1 = n11 / xp.maximum(n11 + n01, 1.0)  # TPR
    p_g_e0 = n10 / xp.maximum(n10 + n00, 1.0)  # FPR
    lift = xp.where(p_g_e0 > 0, p_g_e1 / xp.where(p_g_e0 > 0, p_g_e0, 1.0), xp.nan)
    auroc = 0.5 * (1.0 + p_g_e1 - p_g_e0)
    return dict(phi=phi, cramers_v=xp.abs(phi), lift=lift, auroc=auroc, p_g_given_e=p_g_e1, p_g_given_not_e=p_g_e0)


def auroc_from_scores(scores, labels) -> float:
    """Mann-Whitney AUROC of a continuous score for predicting binary labels."""
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, bool).ravel()
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def event_stats(g, E, mask, strata=None, n_strata: int = 0, alpha=None) -> dict:
    """Observed statistics, pooled over episodes and links, plus per-stratum (distance bin)
    statistics and a count-weighted stratified average."""
    act = np.asarray(g) > 0
    valid = np.asarray(mask, bool)
    out = {k: float(v) for k, v in stats_from_counts(*counts(act, E, valid)).items()}
    out["n"] = int(valid.sum())
    out["event_rate"] = float((np.asarray(E, bool) & valid).sum() / max(valid.sum(), 1))
    if alpha is not None:
        out["auroc_alpha"] = auroc_from_scores(np.asarray(alpha)[valid], np.asarray(E, bool)[valid])
    if strata is not None:
        per = []
        for b in range(n_strata):
            vb = valid & (strata == b)
            s = {k: float(v) for k, v in stats_from_counts(*counts(act, E, vb)).items()}
            s["n"] = int(vb.sum())
            per.append(s)
        out["per_bin"] = per
        w = np.array([p["n"] for p in per], float)
        for k in ("phi", "lift", "auroc"):
            vals = np.array([p[k] for p in per])
            ok = np.isfinite(vals) & (w > 0)
            # only bins where both E=1 and E=0 occur are informative
            out[f"{k}_stratified"] = float(np.sum(vals[ok] * w[ok]) / w[ok].sum()) if ok.any() else float("nan")
    return out


def time_strata(shape) -> np.ndarray:
    """Timestep index of every entry of a ``[ep, T, N, N]`` array (strata for event_stats)."""
    ep, T = shape[:2]
    return np.broadcast_to(np.arange(T)[None, :, None, None], shape).astype(np.int32)


def stratum_counts(g_active, E, valid, strata, n_strata: int) -> np.ndarray:
    """2x2 counts per stratum, shape ``[4, n_strata]`` (n11, n10, n01, n00)."""
    g = np.asarray(g_active) > 0
    e = np.asarray(E, bool)
    v = np.asarray(valid, bool)
    s = np.asarray(strata).ravel()
    cells = (g & e & v, g & ~e & v, ~g & e & v, ~g & ~e & v)
    return np.stack([np.bincount(s, weights=c.ravel().astype(float), minlength=n_strata)[:n_strata] for c in cells])


def mantel_haenszel(c) -> dict:
    """Mantel-Haenszel common odds ratio over strata from counts ``[..., 4, n_strata]``.
    Strata where the gate (or the event) is constant contribute nothing, so time steps at
    which every link is open do not dilute the estimate. Returns log OR and the
    Robins-Breslow-Greenland standard error (``se`` only for unbatched input)."""
    c = np.asarray(c, float)
    a, b, cc, d = c[..., 0, :], c[..., 1, :], c[..., 2, :], c[..., 3, :]
    n = a + b + cc + d
    nz = np.maximum(n, 1.0)
    R, S = a * d / nz, b * cc / nz
    Rs, Ss = R.sum(-1), S.sum(-1)
    log_or = np.where((Rs > 0) & (Ss > 0), np.log(np.maximum(Rs, 1e-300)) - np.log(np.maximum(Ss, 1e-300)), np.nan)
    out = dict(log_or=log_or, odds_ratio=np.exp(log_or))
    if c.ndim == 2 and Rs > 0 and Ss > 0:
        P, Q = (a + d) / nz, (b + cc) / nz
        var = ((P * R).sum() / (2 * Rs**2) + ((P * S + Q * R).sum()) / (2 * Rs * Ss) + (Q * S).sum() / (2 * Ss**2))
        out["se"] = float(np.sqrt(var))
    return out
