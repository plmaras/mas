"""Gate estimators and communication penalties.

Hard-concrete gate (Louizos, Welling & Kingma 2018, arXiv:1712.01312):
    u ~ U(0, 1)
    s = sigmoid((log u - log(1 - u) + alpha) / beta)
    g = clip(s * (zeta - gamma) + gamma, 0, 1)
    P(g > 0) = sigmoid(alpha - beta * log(-gamma / zeta))
Deterministic (evaluation) gate: g = clip(sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1).
A gate is *active* when g > 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def hc_prob_on(alpha, beta=0.5, gamma=-0.1, zeta=1.1):
    return jax.nn.sigmoid(alpha - beta * jnp.log(-gamma / zeta))


def hc_sample(alpha, u, beta=0.5, gamma=-0.1, zeta=1.1):
    s = jax.nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + alpha) / beta)
    return jnp.clip(s * (zeta - gamma) + gamma, 0.0, 1.0)


def hc_deterministic(alpha, gamma=-0.1, zeta=1.1):
    return jnp.clip(jax.nn.sigmoid(alpha) * (zeta - gamma) + gamma, 0.0, 1.0)


def st_sample(alpha, u):
    """Straight-through Bernoulli: forward is a hard 0/1 sample, backward is d sigmoid."""
    p = jax.nn.sigmoid(alpha)
    hard = (u < p).astype(p.dtype)
    return hard + p - jax.lax.stop_gradient(p)


def bernoulli_logp(g, alpha):
    return g * jax.nn.log_sigmoid(alpha) + (1.0 - g) * jax.nn.log_sigmoid(-alpha)


def bernoulli_entropy(p, eps=1e-6):
    p = jnp.clip(p, eps, 1 - eps)
    return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))


def comm_cost(p_on, permitted):
    """Mean over permitted links of P(g > 0)."""
    permitted = permitted.astype(p_on.dtype)
    return jnp.sum(p_on * permitted) / jnp.maximum(jnp.sum(permitted), 1.0)


def switching_cost(p_on, permitted, time_axis: int = 0):
    """Expected flip probability p_t(1-p_{t-1}) + p_{t-1}(1-p_t), averaged over links
    permitted at both t and t-1."""
    p_on = jnp.moveaxis(p_on, time_axis, 0)
    permitted = jnp.moveaxis(permitted, time_axis, 0).astype(p_on.dtype)
    both = permitted[1:] * permitted[:-1]
    flip = p_on[1:] * (1 - p_on[:-1]) + p_on[:-1] * (1 - p_on[1:])
    return jnp.sum(flip * both) / jnp.maximum(jnp.sum(both), 1.0)


def lambda_schedule(progress, target, warmup_frac, anneal_frac):
    """Zero during warm-up, then a linear ramp to ``target`` over ``anneal_frac``."""
    ramp = jnp.clip((progress - warmup_frac) / max(anneal_frac, 1e-8), 0.0, 1.0)
    return target * ramp


def target_density_schedule(progress, target, warmup_frac, anneal_frac):
    """Density target for dual ascent: 1 during warm-up, then a linear ramp to ``target``."""
    ramp = min(max((progress - warmup_frac) / max(anneal_frac, 1e-8), 0.0), 1.0)
    return 1.0 - (1.0 - target) * ramp


def dual_step(lam, density, target, lr, clip):
    """One dual-ascent step on the equality constraint mean P(g>0) = target (numpy or float)."""
    return np.clip(lam + lr * (np.asarray(density) - target), -clip, clip)


def pi_step(integral, density, target, ki, kp, clip):
    """PI controller for the density constraint with anti-windup. Returns
    ``(new integral, lambda_c)``; with ``kp = 0`` this is plain dual ascent (``dual_step``)."""
    err = np.asarray(density) - target
    integral = np.clip(integral + ki * err, -clip, clip)
    return integral, np.clip(integral + kp * err, -clip, clip)
