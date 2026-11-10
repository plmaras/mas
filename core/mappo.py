"""MAPPO (Yu et al. 2022) with parameter sharing, recurrent actor and centralized critic.

Every update collects exactly one full episode (``max_steps`` steps) in each of
``num_envs`` parallel environments, so recurrent states always start at zero and there
are no mid-rollout resets. All agents share the team reward, so the per-agent advantage
equals the team advantage computed from the centralized critic.

Loss = L_PPO + vf_coef * L_V - ent_coef * H(pi)
       + lambda_c * mean P(g>0)                                  (communication cost)
       + lambda_s * mean[p_t(1-p_{t-1}) + p_{t-1}(1-p_t)]        (switching cost)
       - gate_entropy_coef * H(gate)   + lang_coef * CE(token, human label)  + ae_coef * recon
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from core.rollout import Carry, actor_forward, mask_logits, policy_step, reset_batch
from core.setup import Setup
from models import gates as G
from models.actor_critic import CommIn


class TrainState(NamedTuple):
    params: dict
    opt_state: optax.OptState


def make_optimizer(cfg: DictConfig):
    tx = optax.chain(optax.clip_by_global_norm(cfg.algo.max_grad_norm), optax.adam(cfg.algo.lr, eps=1e-5))
    k = int(cfg.algo.get("grad_accum_steps", 1))
    # gradient accumulation (memory fallback): average k minibatch gradients per optimizer step
    return optax.MultiSteps(tx, every_k_schedule=k) if k > 1 else tx


def init_train_state(setup: Setup, cfg: DictConfig, key: jax.Array) -> TrainState:
    p = setup.env
    k1, k2, k3 = jax.random.split(key, 3)
    carry = reset_batch(setup, k3, 1)
    obs = setup.envm.get_obs(jax.tree_util.tree_map(lambda x: x[0], carry.env_state), p)
    from core.rollout import build_comm_in, NO_INTERV

    c, _, _ = build_comm_in(setup, carry, k3, NO_INTERV)
    c0 = jax.tree_util.tree_map(lambda x: x[0], c)
    actor_params = setup.actor.init(k1, carry.h[0], obs, c0)
    gs = setup.envm.global_state(jax.tree_util.tree_map(lambda x: x[0], carry.env_state), p)
    critic_params = setup.critic.init(k2, gs)
    params = {"actor": actor_params, "critic": critic_params}
    return TrainState(params, make_optimizer(cfg).init(params))


def rollout(setup: Setup, params, key: jax.Array, n_envs: int):
    """Collect one full episode per environment. Returns stacked step records [T, E, ...]
    and the final carry."""
    k_reset, k_steps = jax.random.split(key)
    carry = reset_batch(setup, k_reset, n_envs)

    def body(carry, k):
        carry, rec = policy_step(setup, params, carry, k, with_value=True)
        keep = dict(
            obs=rec["obs"], comm=rec["comm"], action=rec["action"], logp=rec["logp"], value=rec["value"], amask=rec["amask"],
            reward=rec["reward"], done=rec["done"], gstate=rec["gstate"], human_label=rec["human_label"],
            g=rec["aux"]["g"], mask=rec["comm"].mask, events=rec["events"], p_on=rec["aux"]["p_on"],
        )
        return carry, keep

    carry, traj = jax.lax.scan(body, carry, jax.random.split(k_steps, setup.env.max_steps))
    return traj, carry


def gae(rewards, values, dones, gamma, lam):
    """Generalised advantage estimation over the time axis (axis 0). The rollout ends at the
    episode boundary, so the bootstrap value after the last step is zero."""

    def body(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nonterm = 1.0 - d.astype(jnp.float32)
        delta = r + gamma * v_next * nonterm - v
        adv = delta + gamma * lam * nonterm * adv_next
        return (adv, v), adv

    zeros = jnp.zeros_like(values[0])
    _, adv = jax.lax.scan(body, (zeros, zeros), (rewards, values, dones), reverse=True)
    return adv, adv + values


def make_update(setup: Setup, cfg: DictConfig):
    A = cfg.algo
    comm = cfg.comm
    tx = make_optimizer(cfg)
    n_envs = int(A.num_envs)
    n_mb = int(A.num_minibatches)
    # an env may fix its own reward scale (wildfire returns are O(1), spread uses algo.reward_scale)
    reward_scale = float(getattr(setup.env, "reward_scale", None) or A.reward_scale)
    assert n_envs % n_mb == 0, "num_envs must be divisible by num_minibatches"

    def loss_fn(params, mb, lam_c, lam_s):
        T, B, N = mb["action"].shape
        h0 = jnp.zeros((B, N, setup.actor.hidden_dim))

        def step(h, xs):
            obs, c, am = xs
            h, _, logits, aux = actor_forward(setup.actor, params["actor"], h, obs, c, False, True)
            logits = mask_logits(logits, am)
            out = dict(logits=logits, p_on=aux["p_on"], gate_logp=aux["gate_logp"], gate_entropy=aux["gate_entropy"])
            if "tok_logits" in aux:
                out["tok_logits"] = aux["tok_logits"]
            if "recon_err" in aux:
                out["recon_err"] = aux["recon_err"]
            return h, out

        _, out = jax.lax.scan(step, h0, (mb["obs"], mb["comm"], mb["amask"]))
        logp_all = jax.nn.log_softmax(out["logits"])
        logp = jnp.take_along_axis(logp_all, mb["action"][..., None], -1)[..., 0] + out["gate_logp"]
        ratio = jnp.exp(logp - mb["logp"])
        adv = mb["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv[..., None]
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - A.clip_eps, 1 + A.clip_eps) * adv))
        probs = jnp.exp(logp_all)
        ent = -jnp.mean(jnp.sum(jnp.where(mb["amask"], probs * logp_all, 0.0), -1)) + jnp.mean(out["gate_entropy"])

        values = jax.vmap(jax.vmap(lambda g: setup.critic.apply(params["critic"], g)))(mb["gstate"])
        v_clip = mb["value"] + jnp.clip(values - mb["value"], -A.vf_clip, A.vf_clip)
        v_loss = 0.5 * jnp.mean(jnp.maximum((values - mb["target"]) ** 2, (v_clip - mb["target"]) ** 2))

        permitted = mb["comm"].mask
        c_cost = G.comm_cost(out["p_on"], permitted)
        s_cost = G.switching_cost(out["p_on"], permitted, time_axis=0)
        perm_f = permitted.astype(jnp.float32)
        gate_ent = jnp.sum(G.bernoulli_entropy(out["p_on"]) * perm_f) / jnp.maximum(perm_f.sum(), 1.0)

        loss = pg + A.vf_coef * v_loss - A.ent_coef * ent + lam_c * c_cost + lam_s * s_cost
        loss = loss - comm.gate_entropy_coef * gate_ent
        aux_l = dict(pg_loss=pg, v_loss=v_loss, entropy=ent, comm_cost=c_cost, switch_cost=s_cost,
                     approx_kl=jnp.mean((ratio - 1) - jnp.log(ratio)))
        if "tok_logits" in out:
            ce = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(out["tok_logits"]), mb["human_label"][..., None], -1))
            loss = loss + comm.lang_coef * ce
            aux_l["lang_ce"] = ce
        if "recon_err" in out:
            rec = jnp.mean(out["recon_err"])
            loss = loss + comm.ae_coef * rec
            aux_l["ae_recon"] = rec
        return loss, aux_l

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(ts: TrainState, key: jax.Array, lam_c, lam_s):
        k_roll, k_perm = jax.random.split(key)
        traj, carry = rollout(setup, ts.params, k_roll, n_envs)
        adv, target = gae(traj["reward"] * reward_scale, traj["value"], traj["done"], A.gamma, A.gae_lambda)
        data = dict(obs=traj["obs"], comm=traj["comm"], action=traj["action"], logp=traj["logp"], amask=traj["amask"],
                    value=traj["value"], gstate=traj["gstate"], human_label=traj["human_label"],
                    adv=adv, target=target)

        def epoch(ts, k):
            perm = jax.random.permutation(k, n_envs).reshape(n_mb, -1)

            def minibatch(ts, idx):
                mb = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=1), data)
                (loss, aux_l), grads = grad_fn(ts.params, mb, lam_c, lam_s)
                updates, opt_state = tx.update(grads, ts.opt_state, ts.params)
                return TrainState(optax.apply_updates(ts.params, updates), opt_state), dict(aux_l, loss=loss)

            return jax.lax.scan(minibatch, ts, perm)

        ts, losses = jax.lax.scan(epoch, ts, jax.random.split(k_perm, int(A.update_epochs)))
        losses = jax.tree_util.tree_map(jnp.mean, losses)

        mask = traj["mask"]
        act = (traj["g"] > 0) & mask
        both = mask[1:] & mask[:-1]
        flips = (act[1:] != act[:-1]) & both
        metrics = dict(
            episode_return=traj["reward"].sum(0).mean(),
            success_rate=carry.env_state.ever_all_covered.mean(),
            final_coverage=jax.vmap(setup.envm.coverage, in_axes=(0, None))(carry.env_state, setup.env).mean(),
            **{k_: v.mean() for k_, v in jax.vmap(setup.envm.final_stats, in_axes=(0, None))(carry.env_state, setup.env).items()},
            active_link_ratio=act.sum() / jnp.maximum(mask.sum(), 1),
            p_on_mean=jnp.sum(traj["p_on"]) / jnp.maximum(mask.sum(), 1),
            gate_switch_rate=flips.sum() / jnp.maximum(both.sum(), 1),
            event_rate=traj["events"].sum() / jnp.maximum(mask.sum(), 1),
            lambda_c=lam_c,
            lambda_s=lam_s,
            **losses,
        )
        return ts, metrics

    return update
