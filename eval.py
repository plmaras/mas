"""Evaluate a trained run and save the full per-episode log as .npz.

    python eval.py --run runs/core/gated_lc0.0_ls0.0/seed_0                 # deterministic gates
    python eval.py --run ... --stochastic                                   # stochastic gates
    python eval.py --run ... --gate-mode random_fixed --density 0.4 --tag c1

Saved arrays (episode-major, ``[episodes, T, ...]``):
  positions[T,N,2], landmark_pos[T,L,2], actions[T,N], action_logits[T,N,A], alpha[T,i,j],
  g[T,i,j] (gates used), g_sampled[T,i,j] (a hard-concrete/Bernoulli sample from the same
  alpha, not fed back), p_on[T,i,j], mask[T,i,j], messages[T,i,:], attn[T,i,j] (weight receiver
  j puts on sender i), vis[T,i,L], covered[T,L], E[T,i,j], reward[T], hidden[T,N,H],
  human_label[T,N], success[], obs[T,N,D].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from core.comm import LEARNED_GATE_MODES
from core.rollout import NO_INTERV, Interv, policy_step, reset_batch
from core.setup import make_setup
from core.utils import load_checkpoint
from analysis.metrics import summarize_logs


def _sample_gates(alpha, g, gate_mode, mask, hc, rng):
    if gate_mode not in LEARNED_GATE_MODES:
        return g
    u = rng.uniform(1e-6, 1 - 1e-6, size=alpha.shape)
    if gate_mode == "hard_concrete":
        s = 1 / (1 + np.exp(-(np.log(u) - np.log1p(-u) + alpha) / hc.beta))
        gs = np.clip(s * (hc.zeta - hc.gamma) + hc.gamma, 0, 1)
    else:
        gs = (u < 1 / (1 + np.exp(-alpha))).astype(np.float32)
    return gs * mask


_EVAL_CACHE: dict = {}


def _eval_key(cfg, *parts) -> str:
    """Cache key: everything that changes the compiled evaluation program (not seeds/params)."""
    c = OmegaConf.to_container(cfg, resolve=True)
    c = {k: v for k, v in c.items() if k not in ("seed", "run_name", "out_dir", "exp_name", "penalties", "logging")}
    return json.dumps([c, *[repr(x) for x in parts]], sort_keys=True, default=str)


def _make_run_batch(setup, batch_size, deterministic, greedy, interv, light, cross):
    def run_batch(params, sender_params, key):
        k_reset, k_steps = jax.random.split(key)
        carry = reset_batch(setup, k_reset, batch_size)

        def body(carry, k):
            carry, rec = policy_step(setup, params, carry, k, deterministic=deterministic, greedy=greedy,
                                     interv=interv, sender_params=sender_params if cross else None)
            aux = rec["aux"]
            out = dict(
                positions=rec["agent_pos"], actions=rec["action"], reward=rec["reward"],
                g=aux["g"], mask=rec["comm"].mask, E=rec["events"], covered=rec["covered"],
                alpha=aux["alpha"],
            )
            if not light:  # vis is [N, cells] per step: large at big N, needed only by probes / F4
                out.update(
                    vis=rec["vis"], landmark_pos=rec["landmark_pos"],
                    action_logits=rec["logits"], action_mask=rec["amask"], p_on=aux["p_on"],
                    messages=aux["msg"], attn=aux["attn"], hidden=aux["hidden"],
                    human_label=rec["human_label"], obs=rec["obs"],
                )
            return carry, out

        carry, logs = jax.lax.scan(body, carry, jax.random.split(k_steps, setup.env.max_steps))
        logs = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), logs)  # [E, T, ...]
        logs["success"] = carry.env_state.ever_all_covered
        for k_, v in jax.vmap(setup.envm.final_stats, in_axes=(0, None))(carry.env_state, setup.env).items():
            logs[k_] = v
        return logs

    return jax.jit(run_batch)


def evaluate(cfg: DictConfig, params, *, n_episodes: int | None = None, seed: int = 12345,
             deterministic: bool = True, greedy: bool | None = None, interv: Interv = NO_INTERV,
             env_overrides: dict | None = None, comm_overrides: dict | None = None,
             sender_params=None, batch_size: int | None = None, light: bool = False):
    """Run evaluation episodes. Returns ``(logs, summary)`` with numpy arrays.
    ``light=True`` keeps only what summaries and event statistics need (gates, mask, events,
    positions, coverage): use it at large N, where vis [N, cells] per step is big.
    The compiled program is cached per static configuration, so evaluating many runs of the
    same condition (seeds) or many conditions with the same shapes compiles once."""
    setup = make_setup(cfg, env_overrides, comm_overrides)
    n_episodes = int(n_episodes or cfg.eval.episodes)
    batch_size = int(min(batch_size or cfg.eval.batch_size, n_episodes))
    greedy = bool(cfg.eval.greedy_actions) if greedy is None else greedy
    cross = sender_params is not None
    key_c = _eval_key(cfg, env_overrides, comm_overrides, batch_size, deterministic, greedy, interv, light, cross)
    if key_c not in _EVAL_CACHE:
        if len(_EVAL_CACHE) > 64:
            _EVAL_CACHE.clear()
        _EVAL_CACHE[key_c] = _make_run_batch(setup, batch_size, deterministic, greedy, interv, light, cross)
    run_batch = _EVAL_CACHE[key_c]
    sp = sender_params if cross else params["actor"]
    key = jax.random.PRNGKey(seed)
    chunks = []
    n_batches = -(-n_episodes // batch_size)
    for b in range(n_batches):
        chunks.append(jax.device_get(run_batch(params, sp, jax.random.fold_in(key, b))))
    logs = {k: np.concatenate([c[k] for c in chunks])[:n_episodes] for k in chunks[0]}
    for k in ("mask", "E", "vis", "covered", "success", "action_mask"):
        if k in logs:
            logs[k] = logs[k].astype(bool)
    rng = np.random.default_rng(seed)
    logs["g_sampled"] = _sample_gates(logs["alpha"], logs["g"], setup.gate_mode, logs["mask"],
                                      cfg.comm.hard_concrete, rng).astype(np.float32)
    summary = summarize_logs(logs)
    summary.update(deterministic_gates=deterministic, gate_mode=setup.gate_mode, n_agents=setup.env.n_agents,
                   obs_radius=setup.env.obs_radius, n_episodes=n_episodes, interv=interv.__dict__,
                   env_name=setup.env.name)
    return logs, summary


def save_logs(path: str | Path, logs: dict, meta: dict, float16: bool = False, save_obs: bool = False) -> None:
    """Compressed ``.npz``. Observations are dropped unless ``save_obs`` (no analysis uses them),
    and landmark/cell positions that never change are stored once (``landmark_pos_static``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for k, v in logs.items():
        if k == "obs" and not save_obs:
            continue
        if k == "landmark_pos" and v.ndim == 4 and np.all(v == v[:1, :1]):
            arrs["landmark_pos_static"] = v[0, 0]
            continue
        if float16 and v.dtype == np.float32 and k not in ("positions", "landmark_pos", "reward"):
            v = v.astype(np.float16)
        arrs[k] = v
    np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrs)


def load_logs(path: str | Path):
    with np.load(path, allow_pickle=False) as f:
        logs = {k: f[k] for k in f.files if k != "meta"}
        meta = json.loads(str(f["meta"])) if "meta" in f.files else {}
    for k, v in logs.items():
        if v.dtype == np.float16:
            logs[k] = v.astype(np.float32)
    if "landmark_pos_static" in logs:
        st = logs.pop("landmark_pos_static")
        E, T = logs["positions"].shape[:2]
        logs["landmark_pos"] = np.broadcast_to(st, (E, T) + st.shape)
    return logs, meta


def evaluate_run(run_dir: str | Path, tag: str = "det", overwrite: bool = False, save: bool = True, **kw):
    """Evaluate a saved run and cache ``eval_<tag>.npz`` + ``eval_<tag>.json`` in the run dir."""
    run_dir = Path(run_dir)
    npz, js = run_dir / f"eval_{tag}.npz", run_dir / f"eval_{tag}.json"
    if js.exists() and not overwrite and (npz.exists() or not save):
        return json.loads(js.read_text())
    cfg, params = load_checkpoint(run_dir)
    logs, summary = evaluate(cfg, params, **kw)
    summary.update(run_dir=str(run_dir), tag=tag)
    if save:
        save_logs(npz, logs, summary, float16=bool(cfg.eval.float16_logs), save_obs=bool(cfg.eval.get("save_obs", False)))
    js.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true", help="sample gates instead of deterministic gates")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gate-mode", default=None, help="override gate mode, e.g. random_fixed (E4 controls)")
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--switch-rate", type=float, default=None)
    ap.add_argument("--hand-rule-d", type=float, default=None)
    ap.add_argument("--zero-messages", action="store_true")
    ap.add_argument("--n-agents", type=int, default=None, help="zero-shot transfer to a different N")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    comm = {k: v for k, v in dict(gate_mode=a.gate_mode, density=a.density, switch_rate=a.switch_rate,
                                  hand_rule_d=a.hand_rule_d).items() if v is not None}
    env = {"n_agents": a.n_agents} if a.n_agents else None
    tag = a.tag or ("stoch" if a.stochastic else "det")
    s = evaluate_run(a.run, tag=tag, overwrite=True, n_episodes=a.episodes, seed=a.seed,
                     deterministic=not a.stochastic, interv=Interv(zero_messages=a.zero_messages),
                     comm_overrides=comm or None, env_overrides=env)
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
