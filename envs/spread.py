"""Cooperative navigation ("simple spread" style) written in pure JAX.

N agents must cover N landmarks in a continuous 2D square arena. Every function is
pure and works on a single environment; vectorise with ``jax.vmap``.

Conventions
-----------
* Agent positions ``agent_pos[N, 2]``, landmark positions ``landmark_pos[L, 2]``.
* ``vis[i, l]`` is True when landmark ``l`` is within ``obs_radius`` of agent ``i``.
* ``covered[l]`` is True when some agent is within ``cover_threshold`` of landmark ``l``.
* Team reward (shared by all agents) after taking the action:
      r = -(sum_l min_i ||p_i - l||  +  collision_penalty * n_colliding_pairs)
  divided by L when ``normalize_reward`` is set (keeps the reward scale constant as N grows).
* Constant entity density: arena side = sqrt(n_agents / density).
* Episodes have a fixed length ``max_steps``; success = all landmarks covered at the
  same time at some step of the episode.

Assumptions (stated explicitly):
* Discrete actions {noop, +x, -x, +y, -y} with MPE-like damped point-mass dynamics.
* Positions are clipped to the arena (no wall bounce).
* Observations are ego-centric and of fixed size regardless of N (k nearest landmarks and
  k nearest agents, zeroed when outside ``obs_radius``), which enables zero-shot transfer
  across N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

ACTION_DIRS = jnp.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=jnp.float32
)
N_ACTIONS = 5
BIG = 1e9


@dataclass(frozen=True)
class SpreadParams:
    name: str = "spread"
    n_agents: int = 4
    n_landmarks: int = 4
    density: float = 1.0
    obs_radius: float = 0.5
    max_steps: int = 50
    dt: float = 0.1
    damping: float = 0.25
    accel: float = 5.0
    max_speed: float = 1.0
    agent_radius: float = 0.05
    cover_threshold: float = 0.1
    collision_penalty: float = 1.0
    normalize_reward: bool = True
    k_obs_landmarks: int = 4
    k_obs_agents: int = 4

    @property
    def arena_size(self) -> float:
        return math.sqrt(self.n_agents / self.density)

    @property
    def half(self) -> float:
        return self.arena_size / 2.0

    @property
    def obs_dim(self) -> int:
        return 5 + 3 * self.k_obs_landmarks + 3 * self.k_obs_agents

    @classmethod
    def from_cfg(cls, cfg) -> "SpreadParams":
        n_l = cfg.n_landmarks if cfg.get("n_landmarks") is not None else cfg.n_agents
        return cls(
            n_agents=int(cfg.n_agents),
            n_landmarks=int(n_l),
            density=float(cfg.density),
            obs_radius=float(cfg.obs_radius),
            max_steps=int(cfg.max_steps),
            dt=float(cfg.dt),
            damping=float(cfg.damping),
            accel=float(cfg.accel),
            max_speed=float(cfg.max_speed),
            agent_radius=float(cfg.agent_radius),
            cover_threshold=float(cfg.cover_threshold),
            collision_penalty=float(cfg.collision_penalty),
            normalize_reward=bool(cfg.normalize_reward),
            k_obs_landmarks=int(cfg.k_obs_landmarks),
            k_obs_agents=int(cfg.k_obs_agents),
        )


class EnvState(NamedTuple):
    agent_pos: jax.Array  # [N, 2]
    agent_vel: jax.Array  # [N, 2]
    landmark_pos: jax.Array  # [L, 2]
    t: jax.Array  # int32 scalar
    ever_all_covered: jax.Array  # bool scalar


class GlobalState(NamedTuple):
    """Input of the centralized critic (never contains messages)."""

    agents: jax.Array  # [N, 5]
    landmarks: jax.Array  # [L, 4]
    time: jax.Array  # [1]


def pairwise_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    """Euclidean distances, ``out[i, j] = ||b_j - a_i||``."""
    diff = b[None, :, :] - a[:, None, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


def reset(key: jax.Array, p: SpreadParams) -> EnvState:
    k1, k2 = jax.random.split(key)
    agent_pos = jax.random.uniform(k1, (p.n_agents, 2), minval=-p.half, maxval=p.half)
    landmark_pos = jax.random.uniform(k2, (p.n_landmarks, 2), minval=-p.half, maxval=p.half)
    state = EnvState(
        agent_pos=agent_pos,
        agent_vel=jnp.zeros((p.n_agents, 2)),
        landmark_pos=landmark_pos,
        t=jnp.zeros((), jnp.int32),
        ever_all_covered=jnp.zeros((), bool),
    )
    return state._replace(ever_all_covered=jnp.all(coverage(state, p)))


def visibility(state: EnvState, p: SpreadParams) -> jax.Array:
    """``vis[i, l]``: landmark l within obs_radius of agent i."""
    return pairwise_dist(state.agent_pos, state.landmark_pos) <= p.obs_radius


def coverage(state: EnvState, p: SpreadParams) -> jax.Array:
    """``covered[l]``: some agent within cover_threshold of landmark l."""
    d = pairwise_dist(state.agent_pos, state.landmark_pos)
    return jnp.min(d, axis=0) <= p.cover_threshold


def n_collisions(state: EnvState, p: SpreadParams) -> jax.Array:
    """Number of unordered agent pairs closer than 2 * agent_radius."""
    d = pairwise_dist(state.agent_pos, state.agent_pos)
    upper = jnp.triu(jnp.ones_like(d, dtype=bool), k=1)
    return jnp.sum((d < 2.0 * p.agent_radius) & upper)


def reward_fn(state: EnvState, p: SpreadParams) -> jax.Array:
    d = pairwise_dist(state.agent_pos, state.landmark_pos)
    r = -jnp.sum(jnp.min(d, axis=0)) - p.collision_penalty * n_collisions(state, p)
    if p.normalize_reward:
        r = r / p.n_landmarks
    return r


def step(state: EnvState, actions: jax.Array, p: SpreadParams, key: jax.Array | None = None):
    """Apply integer actions ``[N]``. Returns ``(new_state, reward, done)``. ``key`` is unused
    (deterministic dynamics); it keeps the signature shared with the wildfire env."""
    acc = ACTION_DIRS[actions] * p.accel
    vel = state.agent_vel * (1.0 - p.damping) + acc * p.dt
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    vel = jnp.where(speed > p.max_speed, vel / jnp.maximum(speed, 1e-8) * p.max_speed, vel)
    pos = jnp.clip(state.agent_pos + vel * p.dt, -p.half, p.half)
    new = state._replace(agent_pos=pos, agent_vel=vel, t=state.t + 1)
    new = new._replace(ever_all_covered=state.ever_all_covered | jnp.all(coverage(new, p)))
    reward = reward_fn(new, p)
    done = new.t >= p.max_steps
    return new, reward, done


def _k_nearest_feats(rel: jax.Array, dist: jax.Array, radius: float, k: int) -> jax.Array:
    """Features of the k nearest entities: [dx/R * vis, dy/R * vis, vis], zero padded."""
    n, m = dist.shape
    kk = min(k, m)
    if kk == 0:
        return jnp.zeros((n, 3 * k))
    neg, idx = jax.lax.top_k(-dist, kk)
    r = jnp.take_along_axis(rel, idx[..., None], axis=1)
    vis = (-neg <= radius).astype(jnp.float32)
    feats = jnp.concatenate([r / radius * vis[..., None], vis[..., None]], axis=-1)
    if kk < k:
        feats = jnp.concatenate([feats, jnp.zeros((n, k - kk, 3))], axis=1)
    return feats.reshape(n, 3 * k)


def get_obs(state: EnvState, p: SpreadParams) -> jax.Array:
    """Local observation of every agent, shape ``[N, obs_dim]``."""
    n = p.n_agents
    own = jnp.concatenate(
        [
            state.agent_vel / p.max_speed,
            state.agent_pos / p.half,
            jnp.full((n, 1), state.t / p.max_steps),
        ],
        axis=-1,
    )
    rel_l = state.landmark_pos[None, :, :] - state.agent_pos[:, None, :]
    d_l = jnp.linalg.norm(rel_l, axis=-1)
    rel_a = state.agent_pos[None, :, :] - state.agent_pos[:, None, :]
    d_a = jnp.linalg.norm(rel_a, axis=-1) + jnp.eye(n) * BIG  # exclude self
    lm = _k_nearest_feats(rel_l, d_l, p.obs_radius, p.k_obs_landmarks)
    ag = _k_nearest_feats(rel_a, d_a, p.obs_radius, p.k_obs_agents)
    return jnp.concatenate([own, lm, ag], axis=-1)


def global_state(state: EnvState, p: SpreadParams) -> GlobalState:
    d = pairwise_dist(state.agent_pos, state.landmark_pos)  # [N, L]
    cap = 10.0
    agents = jnp.concatenate(
        [
            state.agent_pos / p.half,
            state.agent_vel / p.max_speed,
            jnp.minimum(jnp.min(d, axis=1, keepdims=True) / p.obs_radius, cap),
        ],
        axis=-1,
    )
    landmarks = jnp.concatenate(
        [
            state.landmark_pos / p.half,
            coverage(state, p)[:, None].astype(jnp.float32),
            jnp.minimum(jnp.min(d, axis=0)[:, None] / p.obs_radius, cap),
        ],
        axis=-1,
    )
    return GlobalState(agents=agents, landmarks=landmarks, time=(state.t / p.max_steps)[None].astype(jnp.float32))


def human_label(state: EnvState, p: SpreadParams, n_dirs: int = 8) -> jax.Array:
    """Human-readable token per agent: compass octant (0..n_dirs-1) of the nearest visible
    uncovered landmark, or ``n_dirs`` ("none") when the agent sees no uncovered landmark.
    Used by the language-style message variant and the readability metric."""
    vis = visibility(state, p)
    cov = coverage(state, p)
    cand = vis & ~cov[None, :]
    rel = state.landmark_pos[None, :, :] - state.agent_pos[:, None, :]
    d = jnp.where(cand, jnp.linalg.norm(rel, axis=-1), BIG)
    idx = jnp.argmin(d, axis=1)
    r = rel[jnp.arange(p.n_agents), idx]
    ang = jnp.arctan2(r[:, 1], r[:, 0])  # [-pi, pi]
    octant = jnp.floor((ang + jnp.pi) / (2 * jnp.pi / n_dirs)).astype(jnp.int32) % n_dirs
    return jnp.where(jnp.any(cand, axis=1), octant, n_dirs)


# ---------------------------------------------------------------------------- shared env interface
def n_actions(p: SpreadParams) -> int:
    return N_ACTIONS


def action_dirs(p: SpreadParams):
    return ACTION_DIRS


def action_mask(state: EnvState, p: SpreadParams) -> jax.Array:
    """Every action is always allowed in this task."""
    return jnp.ones((p.n_agents, N_ACTIONS), bool)


def events(state: EnvState, p: SpreadParams, mask: jax.Array) -> jax.Array:
    from core.comm import compute_events

    return compute_events(visibility(state, p), coverage(state, p), mask)


def final_stats(state: EnvState, p: SpreadParams) -> dict:
    return {}
