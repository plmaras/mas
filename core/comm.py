"""Communication graph, non-learned gate modes, and the task event.

Link convention used everywhere in this repo: arrays indexed ``[i, j]`` describe the
directed link from **sender i** to **receiver j**.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.spread import BIG, pairwise_dist  # env-independent helpers

LEARNED_GATE_MODES = ("hard_concrete", "straight_through", "reinforce")
EXTERNAL_GATE_MODES = ("full", "hand_rule", "random_fixed", "random_markov")


def knn_mask(agent_pos: jax.Array, k: int, comm_radius: float | None = None) -> jax.Array:
    """Spatial K-nearest-neighbour mask. ``mask[i, j]`` is True when sender i is one of the
    k nearest agents of receiver j (self excluded). Links outside the mask count as off."""
    n = agent_pos.shape[0]
    kk = min(k, n - 1)
    d = pairwise_dist(agent_pos, agent_pos) + jnp.eye(n) * BIG
    _, idx = jax.lax.top_k(-d, kk)  # rows = receivers, idx = chosen senders
    m_recv = jnp.zeros((n, n), bool).at[jnp.arange(n)[:, None], idx].set(True)
    mask = m_recv.T  # [sender, receiver]
    if comm_radius is not None:
        mask = mask & (d <= comm_radius)
    return mask


def pair_geometry(agent_pos: jax.Array, obs_radius: float):
    """Returns ``rel[i, j] = (p_j - p_i) / R``, ``dist_n[i, j] = |p_j - p_i| / R`` and raw distances."""
    rel = agent_pos[None, :, :] - agent_pos[:, None, :]
    dist = jnp.linalg.norm(rel, axis=-1)
    return rel / obs_radius, dist / obs_radius, dist


class CommState(NamedTuple):
    fixed: jax.Array  # [N, N] random fixed mask (C1), resampled every episode
    prev: jax.Array  # [N, N] previous gate of the random Markov mask (C2)


def init_comm_state(key: jax.Array, n: int, density: float) -> CommState:
    k1, k2 = jax.random.split(key)
    return CommState(
        fixed=jax.random.bernoulli(k1, density, (n, n)).astype(jnp.float32),
        prev=jax.random.bernoulli(k2, density, (n, n)).astype(jnp.float32),
    )


def markov_rates(density: float, switch_rate: float):
    """Transition probabilities of a 2-state chain with stationary P(on)=density and
    per-step flip probability switch_rate:  P(off->on)=a, P(on->off)=b with
    density = a/(a+b) and switch_rate = 2ab/(a+b)."""
    rho = min(max(density, 1e-6), 1 - 1e-6)
    a = switch_rate / (2.0 * (1.0 - rho))
    b = switch_rate / (2.0 * rho)
    return min(a, 1.0), min(b, 1.0)


def external_gates(
    mode: str,
    cstate: CommState,
    key: jax.Array,
    dist_raw: jax.Array,
    hand_rule_d: float,
    density: float,
    switch_rate: float,
):
    """Gate values for non-learned modes (before applying the neighbour mask)."""
    n = dist_raw.shape[0]
    if mode == "full" or mode in LEARNED_GATE_MODES:
        return jnp.ones((n, n)), cstate
    if mode == "hand_rule":
        return (dist_raw < hand_rule_d).astype(jnp.float32), cstate
    if mode == "random_fixed":
        return cstate.fixed, cstate
    if mode == "random_markov":
        a, b = markov_rates(density, switch_rate)
        u = jax.random.uniform(key, (n, n))
        g = jnp.where(cstate.prev > 0, (u >= b), (u < a)).astype(jnp.float32)
        return g, cstate._replace(prev=g)
    raise ValueError(f"unknown gate mode {mode}")


def compute_events(vis: jax.Array, covered: jax.Array, mask: jax.Array) -> jax.Array:
    """Task event ``E[..., i, j]``: sender i sees an uncovered landmark that is outside
    receiver j's observation radius. Only defined (True) on links inside the neighbour mask.

    Shapes: ``vis[..., N, L]``, ``covered[..., L]``, ``mask[..., N, N]``. Works with numpy
    or jax arrays."""
    xp = jnp if isinstance(vis, jax.Array) else __import__("numpy")
    private = (vis & ~covered[..., None, :]).astype(xp.float32)  # sender sees, uncovered
    hidden = (~vis).astype(xp.float32)  # receiver does not see
    e = xp.einsum("...il,...jl->...ij", private, hidden) > 0
    return e & mask.astype(bool)
