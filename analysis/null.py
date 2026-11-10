"""Circular time-shift null model for gate/event association (E5).

For every (episode, link) sequence the gate series is rotated by a random offset *among the
timesteps at which the link is permitted*; the event series stays fixed. This preserves each
link's density and autocorrelation while breaking its alignment with events. Plain
timestep shuffling would destroy autocorrelation and make persistent gates look significant.

Fully vectorised: one shift is a gather; shifts are mapped with ``jax.lax.map`` (batched
vmap), no Python loop over shifts.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from analysis.events import stats_from_counts


def to_sequences(arr, time_axis: int = 1):
    """``[ep, T, N, N]`` -> ``[ep*N*N, T]`` (one row per (episode, link))."""
    a = np.moveaxis(np.asarray(arr), time_axis, -1)
    return a.reshape(-1, a.shape[-1])


def circular_shift_valid(x, valid, offsets):
    """Rotate each row of ``x`` by ``offsets`` positions among its valid entries.
    ``x, valid: [S, T]``, ``offsets: [S]`` (taken modulo the number of valid entries)."""
    S, T = x.shape
    order = jnp.argsort(~valid, axis=1, stable=True)  # valid time indices first, in time order
    n_valid = jnp.maximum(valid.sum(1), 1)
    rank = jnp.cumsum(valid, axis=1) - 1  # rank of each valid step
    src_rank = jnp.mod(rank - offsets[:, None], n_valid[:, None])
    src_t = jnp.take_along_axis(order, src_rank, axis=1)
    shifted = jnp.take_along_axis(x, src_t, axis=1)
    return jnp.where(valid, shifted, x)


def _stratified_counts(g, e, valid, strata, n_strata):
    flat = lambda a: a.reshape(-1)
    s = flat(strata)
    v = flat(valid)
    gg, ee = flat(g) & v, flat(e) & v
    seg = lambda m: jax.ops.segment_sum(m.astype(jnp.float32), s, num_segments=n_strata)
    return seg(gg & ee), seg(gg & ~ee & v), seg(~gg & ee & v), seg(~gg & ~ee & v)


@partial(jax.jit, static_argnames=("n_shifts", "n_strata", "batch_size"))
def _null_counts(key, g, e, valid, strata, n_shifts, n_strata, batch_size):
    n_valid = valid.sum(1)

    def one(k):
        offs = jnp.floor(jax.random.uniform(k, (g.shape[0],)) * jnp.maximum(n_valid, 1)).astype(jnp.int32)
        gs = circular_shift_valid(g, valid, offs)
        return jnp.stack(_stratified_counts(gs, e, valid, strata, n_strata))  # [4, n_strata]

    keys = jax.random.split(key, n_shifts)
    return jax.lax.map(one, keys, batch_size=batch_size)  # [n_shifts, 4, n_strata]


def circular_shift_null(g_active, E, mask, strata=None, n_strata: int = 1, n_shifts: int = 1000,
                        seed: int = 0, batch_size: int = 50) -> dict:
    """Null distribution of phi / lift / AUROC (pooled and per stratum).

    Inputs ``[ep, T, N, N]``. Returns dict with ``null_pooled[stat] -> [n_shifts]``,
    ``null_per_bin[stat] -> [n_shifts, n_strata]`` and ``null_stratified[stat] -> [n_shifts]``.
    """
    g = to_sequences(np.asarray(g_active) > 0)
    e = to_sequences(np.asarray(E, bool))
    v = to_sequences(np.asarray(mask, bool))
    if strata is None:
        st = np.zeros_like(g, dtype=np.int32)
        n_strata = 1
    else:
        st = to_sequences(np.asarray(strata, np.int32))
    keep = v.any(1)  # drop links never permitted
    g, e, v, st = (jnp.asarray(a[keep]) for a in (g, e, v, st))
    # bound the working set of one vmapped batch to ~2e8 elements
    batch_size = int(max(1, min(batch_size, n_shifts, 2e8 // max(g.size, 1))))
    c = np.asarray(_null_counts(jax.random.PRNGKey(seed), g, e, v, st, int(n_shifts), int(n_strata), int(batch_size)))
    n11, n10, n01, n00 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]  # [S, B]
    per_bin = stats_from_counts(n11, n10, n01, n00)
    pooled = stats_from_counts(n11.sum(1), n10.sum(1), n01.sum(1), n00.sum(1))
    w = (n11 + n10 + n01 + n00)
    strat = {}
    for k in ("phi", "lift", "auroc"):
        val = per_bin[k]
        ok = np.isfinite(val) & (w > 0)
        strat[k] = np.where(ok, val * w, 0).sum(1) / np.maximum(np.where(ok, w, 0).sum(1), 1)
    return dict(null_pooled=pooled, null_per_bin=per_bin, null_stratified=strat)


def compare_to_null(observed: float, null: np.ndarray) -> dict:
    """Observed minus null mean, null percentile of the observed value and a two-sided
    permutation p-value."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return dict(observed=observed, null_mean=float("nan"), excess=float("nan"), percentile=float("nan"), p_value=float("nan"))
    mu = float(null.mean())
    pct = float((null < observed).mean() + 0.5 * (null == observed).mean())
    p = float((np.sum(np.abs(null - mu) >= abs(observed - mu)) + 1) / (null.size + 1))
    return dict(observed=float(observed), null_mean=mu, null_std=float(null.std()), excess=float(observed - mu),
                percentile=pct, p_value=p)


def episode_shuffle_null(g_active, E, mask, strata=None, n_strata: int = 1, n_perm: int = 1000,
                         seed: int = 0) -> dict:
    """Null that keeps time structure: the whole gate sequence of every (episode, link) is
    paired with the events of a *different* random episode (same link index, same timesteps).

    It preserves each gate series' autocorrelation and, unlike a circular shift, its position
    within the episode, so a gate that is simply "on early" and an event that is "frequent
    early" show no excess association. Inputs ``[ep, T, N, N]``; link-timesteps count only
    where both episodes permit the link. Returns the keys of ``circular_shift_null`` plus the
    per-stratum ``counts [n_perm, 4, n_strata]`` (for Mantel-Haenszel statistics)."""
    g = np.asarray(g_active) > 0
    e = np.asarray(E, bool)
    v = np.asarray(mask, bool)
    n_ep = g.shape[0]
    st = np.zeros(g.shape, np.int32) if strata is None else np.asarray(strata, np.int32)
    n_strata = 1 if strata is None else int(n_strata)
    rng = np.random.default_rng(seed)
    c = np.zeros((n_perm, 4, n_strata))
    for k in range(n_perm):
        perm = rng.permutation(n_ep)
        while n_ep > 1 and np.any(perm == np.arange(n_ep)):  # derangement: never pair an episode with itself
            fixed = perm == np.arange(n_ep)
            perm[fixed] = rng.permutation(perm[fixed]) if fixed.sum() > 1 else perm[fixed]
            if fixed.sum() == 1:
                i = int(np.flatnonzero(fixed)[0])
                j = int(rng.integers(n_ep - 1))
                j += j >= i
                perm[i], perm[j] = perm[j], perm[i]
        gs = g[perm]
        valid = v & v[perm]
        s = st.ravel()
        for q, m in enumerate((gs & e & valid, gs & ~e & valid, ~gs & e & valid, ~gs & ~e & valid)):
            c[k, q] = np.bincount(s, weights=m.ravel().astype(float), minlength=n_strata)[:n_strata]
    n11, n10, n01, n00 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    per_bin = stats_from_counts(n11, n10, n01, n00)
    pooled = stats_from_counts(n11.sum(1), n10.sum(1), n01.sum(1), n00.sum(1))
    w = n11 + n10 + n01 + n00
    strat = {}
    for k in ("phi", "lift", "auroc"):
        val = per_bin[k]
        ok = np.isfinite(val) & (w > 0)
        strat[k] = np.where(ok, val * w, 0).sum(1) / np.maximum(np.where(ok, w, 0).sum(1), 1)
    return dict(null_pooled=pooled, null_per_bin=per_bin, null_stratified=strat, counts=c)
