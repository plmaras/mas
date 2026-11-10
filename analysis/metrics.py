"""Link metrics computed from evaluation logs (numpy).

Array layout: gates/masks ``[..., T, N, N]`` (time axis = -3, link = sender i -> receiver j).
"""
from __future__ import annotations

import numpy as np


def _active(g, mask):
    return (np.asarray(g) > 0) & np.asarray(mask, bool)


def active_link_ratio(g, mask) -> float:
    """Fraction of permitted link-timesteps whose gate is active (g > 0)."""
    mask = np.asarray(mask, bool)
    return float(_active(g, mask).sum() / max(mask.sum(), 1))


def gate_switch_rate(g, mask) -> float:
    """Gate-driven switches: fraction of link-steps permitted at both t-1 and t whose
    active state flipped."""
    mask = np.asarray(mask, bool)
    a = _active(g, mask)
    both = mask[..., 1:, :, :] & mask[..., :-1, :, :]
    flips = (a[..., 1:, :, :] != a[..., :-1, :, :]) & both
    return float(flips.sum() / max(both.sum(), 1))


def mask_switch_rate(mask) -> float:
    """Mask-driven switches (neighbour-set changes): fraction of link-steps permitted at t-1
    or t whose permission changed."""
    mask = np.asarray(mask, bool)
    change = mask[..., 1:, :, :] != mask[..., :-1, :, :]
    either = mask[..., 1:, :, :] | mask[..., :-1, :, :]
    return float(change.sum() / max(either.sum(), 1))


def mean_run_length(g, mask) -> float:
    """Mean length (in steps) of maximal runs of consecutive active (and permitted) steps."""
    a = _active(g, mask)
    starts = a[..., :1, :, :].sum() + (a[..., 1:, :, :] & ~a[..., :-1, :, :]).sum()
    return float(a.sum() / max(starts, 1))


def event_rate(E, mask) -> float:
    mask = np.asarray(mask, bool)
    return float((np.asarray(E, bool) & mask).sum() / max(mask.sum(), 1))


def link_distance(positions) -> np.ndarray:
    """``dist[..., T, i, j] = |p_j - p_i|`` from positions ``[..., T, N, 2]``."""
    p = np.asarray(positions)
    return np.linalg.norm(p[..., None, :, :] - p[..., :, None, :], axis=-1)


def distance_bins(positions, edges, obs_radius: float) -> np.ndarray:
    """Bin index of each link-timestep; ``edges`` are in units of obs_radius."""
    d = link_distance(positions) / obs_radius
    return np.clip(np.digitize(d, np.asarray(edges)[1:-1]), 0, len(edges) - 2)


def per_bin_rate(values, mask, bins, n_bins) -> np.ndarray:
    mask = np.asarray(mask, bool)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = mask & (bins == b)
        if sel.sum():
            out[b] = np.asarray(values, bool)[sel].mean()
    return out


def summarize_logs(logs: dict) -> dict:
    """Headline numbers for one evaluation log."""
    g, mask = logs["g"], logs["mask"]
    out = dict(
        episode_return=float(logs["reward"].sum(-1).mean()),
        success_rate=float(np.mean(logs["success"])),
        final_coverage=float(logs["covered"][:, -1].mean()),
        active_link_ratio=active_link_ratio(g, mask),
        gate_switch_rate=gate_switch_rate(g, mask),
        mask_switch_rate=mask_switch_rate(mask),
        mean_run_length=mean_run_length(g, mask),
        event_rate=event_rate(logs["E"], mask),
        links_per_agent_step=float(_active(g, mask).sum() / (g.shape[0] * g.shape[1] * g.shape[2])),
    )
    for k, v in logs.items():  # env-specific episode outcomes (e.g. wildfire burned-area fraction)
        if k.startswith("final_") and np.ndim(v) == 1:
            out[k] = float(np.mean(v))
    if "g_sampled" in logs:
        out.update(
            sampled_active_link_ratio=active_link_ratio(logs["g_sampled"], mask),
            sampled_gate_switch_rate=gate_switch_rate(logs["g_sampled"], mask),
            sampled_mean_run_length=mean_run_length(logs["g_sampled"], mask),
        )
    return out
