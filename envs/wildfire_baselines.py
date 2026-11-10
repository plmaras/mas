"""Scripted (non-learned) policies for the wildfire env, used for calibration and as reference
rows in E1. ``noop``, ``random``, ``strongest`` and ``weakest`` follow free-range-zoo's
``wildfire/baselines`` (choose among fires in range; noop when out of suppressant). The
mobile ``greedy_*`` policies additionally walk toward fires:

* ``greedy_local``  : walks toward the nearest burning cell inside its own view, else explores;
* ``greedy_global`` : the same with the whole grid visible (an information upper reference:
                      what perfect information sharing could buy a non-coordinating team).

All policies are pure functions ``(key, state, params) -> actions[N]``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from envs import wildfire as W


def _fight_choice(key, s, p, mode: str):
    """Index (into the fight block) of the chosen in-range fire, and whether any exists."""
    am = W.action_mask(s, p)[:, 5:]
    offs = jnp.asarray(W.fight_offsets(p))
    pos = s.agent_pos.astype(jnp.int32)
    tgt = jnp.clip(pos[:, None, :] + offs[None], 0, jnp.array([p.H - 1, p.W - 1]))
    inten = s.intensity[tgt[..., 0], tgt[..., 1]].astype(jnp.float32)
    noise = jax.random.uniform(key, am.shape) * 1e-3
    if mode == "strongest":
        score = inten + noise
    elif mode == "weakest":
        score = -inten + noise
    else:  # random fire
        score = noise
    score = jnp.where(am, score, -jnp.inf)
    return jnp.argmax(score, -1), am.any(-1)


def noop(key, s, p):
    return jnp.zeros((p.n_agents,), jnp.int32)


def random(key, s, p):
    am = W.action_mask(s, p)
    return jax.random.categorical(key, jnp.where(am, 0.0, -1e9)).astype(jnp.int32)


def _static(mode):
    def pol(key, s, p):
        idx, any_ = _fight_choice(key, s, p, mode)
        return jnp.where(any_, 5 + idx, 0).astype(jnp.int32)

    return pol


strongest = _static("strongest")
weakest = _static("weakest")


def _greedy(key, s, p, radius):
    k1, k2 = jax.random.split(key)
    idx, any_ = _fight_choice(k1, s, p, "strongest")
    lit = W.burning(s).ravel()
    rel = s.landmark_pos[None] - s.agent_pos[:, None]
    cheb = jnp.max(jnp.abs(rel), -1)
    cand = lit[None] & (cheb <= radius)
    d = jnp.where(cand, jnp.abs(rel).sum(-1), W.BIG)
    tgt = jnp.argmin(d, -1)
    seen = cand.any(-1)
    r = rel[jnp.arange(p.n_agents), tgt]
    # step along the larger axis toward the target
    vert = jnp.abs(r[:, 0]) >= jnp.abs(r[:, 1])
    move = jnp.where(vert, jnp.where(r[:, 0] < 0, 1, 2), jnp.where(r[:, 1] < 0, 3, 4))
    explore = jax.random.randint(k2, (p.n_agents,), 1, 5)
    am = W.action_mask(s, p)
    explore = jnp.where(am[jnp.arange(p.n_agents), explore], explore, 0)
    has_supp = s.suppressants > 0
    act = jnp.where(any_, 5 + idx, jnp.where(~has_supp, 0, jnp.where(seen, move, explore)))
    return act.astype(jnp.int32)


def greedy_local(key, s, p):
    return _greedy(key, s, p, p.obs_radius)


def greedy_global(key, s, p):
    return _greedy(key, s, p, 10**6)


POLICIES = dict(noop=noop, random=random, strongest=strongest, weakest=weakest,
                greedy_local=greedy_local, greedy_global=greedy_global)


def rollout_stats(policy, p, key, n_episodes: int = 256):
    """Mean return, success rate and episode outcomes of a scripted policy."""

    def episode(k):
        k0, k1 = jax.random.split(k)
        s = W.reset(k0, p)

        def body(s, kk):
            ka, ke = jax.random.split(kk)
            a = policy(ka, s, p)
            s, r, _ = W.step(s, a, p, ke)
            return s, r

        s, rs = jax.lax.scan(body, s, jax.random.split(k1, p.max_steps))
        return dict(ret=rs.sum(), success=s.ever_all_covered, **W.final_stats(s, p))

    out = jax.jit(jax.vmap(episode))(jax.random.split(key, n_episodes))
    return {k: float(jnp.mean(v)) for k, v in out.items()}
