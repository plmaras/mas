"""Environment registry. Every env module exposes the same functional interface:

``Params.from_cfg``, ``reset(key, p)``, ``step(state, actions, p, key)``, ``get_obs``,
``global_state``, ``visibility``, ``coverage``, ``events(state, p, mask)``, ``action_mask``,
``human_label``, ``final_stats``, ``n_actions(p)``, ``action_dirs(p)`` and ``pairwise_dist``.
"""
from __future__ import annotations

from envs import spread, wildfire

ENVS = {"spread": (spread, spread.SpreadParams), "wildfire": (wildfire, wildfire.WildfireParams)}


def get_env(name: str):
    """``(module, Params class)`` for an env name (``cfg.env.name``)."""
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose from {sorted(ENVS)}")
    return ENVS[name]


def make_params(env_cfg):
    """``env_cfg.kind`` selects the env (``name`` is just the config's label)."""
    mod, cls = get_env(str(env_cfg.get("kind", env_cfg.get("name", "spread"))))
    return mod, cls.from_cfg(env_cfg)
