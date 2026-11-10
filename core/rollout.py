"""One environment/policy step shared by training rollouts and evaluation episodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.comm import CommState, external_gates, init_comm_state, knn_mask, pair_geometry
from core.setup import Setup
from models.actor_critic import CommActor, CommIn, apply_interventions


@dataclass(frozen=True)
class Interv:
    """Test-time interventions (static, hashable)."""

    zero_messages: bool = False  # F1 / C4
    ablate: str = "none"  # none | event | random   (F2)
    ablate_rate: float = 0.0  # per-active-link ablation probability for ablate="random"
    shuffle_senders: bool = False  # F3: receivers read a random other sender's message


NO_INTERV = Interv()

NEG_INF = -1e9


def mask_logits(logits, amask):
    """Invalid actions (``amask`` False) get probability ~0. Used identically when sampling,
    when the PPO update replays the policy, and in the teacher-forced message tests."""
    return jnp.where(amask, logits, NEG_INF)


class Carry(NamedTuple):
    env_state: NamedTuple  # env state, batched over envs
    cstate: CommState
    h: jax.Array  # [E, N, H] hidden of the acting (receiver) model
    h_send: jax.Array  # [E, N, H] hidden of the sender model (cross-play only)


def reset_batch(setup: Setup, key: jax.Array, n_envs: int) -> Carry:
    k1, k2 = jax.random.split(key)
    p = setup.env
    env_state = jax.vmap(setup.envm.reset, in_axes=(0, None))(jax.random.split(k1, n_envs), p)
    cstate = jax.vmap(init_comm_state, in_axes=(0, None, None))(jax.random.split(k2, n_envs), p.n_agents, setup.density)
    h = jnp.zeros((n_envs, p.n_agents, setup.actor.hidden_dim))
    return Carry(env_state, cstate, h, h)


def build_comm_in(setup: Setup, carry: Carry, key: jax.Array, interv: Interv):
    """Graph, noise and intervention inputs for one step. Returns (CommIn, new cstate, extras)."""
    p = setup.env
    E, N = carry.h.shape[:2]
    V = setup.vocab_size
    k_u, k_tok, k_ext, k_abl, k_perm = jax.random.split(key, 5)
    es = carry.env_state
    mask = jax.vmap(knn_mask, in_axes=(0, None, None))(es.agent_pos, setup.k_neighbors, setup.comm_radius)
    rel, dist_n, dist_raw = jax.vmap(pair_geometry, in_axes=(0, None))(es.agent_pos, p.obs_radius)
    g_ext, cstate = jax.vmap(external_gates, in_axes=(None, 0, 0, 0, None, None, None))(
        setup.gate_mode, carry.cstate, jax.random.split(k_ext, E), dist_raw,
        setup.hand_rule_d, setup.density, setup.switch_rate,
    )
    vis = jax.vmap(setup.envm.visibility, in_axes=(0, None))(es, p)
    covered = jax.vmap(setup.envm.coverage, in_axes=(0, None))(es, p)
    events = jax.vmap(setup.envm.events, in_axes=(0, None, 0))(es, p, mask)

    keep = jnp.ones((E, N, N))
    if interv.ablate == "event":
        keep = 1.0 - events.astype(jnp.float32)
    elif interv.ablate == "random":
        keep = 1.0 - jax.random.bernoulli(k_abl, interv.ablate_rate, (E, N, N)).astype(jnp.float32)
    if interv.shuffle_senders:
        perm = jax.vmap(lambda k: jax.random.permutation(k, N))(jax.random.split(k_perm, E))
    else:
        perm = jnp.broadcast_to(jnp.arange(N), (E, N))
    scale = 0.0 if (setup.zero_messages or interv.zero_messages) else 1.0

    c = CommIn(
        mask=mask,
        rel=rel,
        dist=dist_n,
        gate_u=jax.random.uniform(k_u, (E, N, N), minval=1e-6, maxval=1 - 1e-6),
        tok_u=jax.random.gumbel(k_tok, (E, N, V)),
        g_ext=g_ext,
        g_given=jnp.zeros((E, N, N)),
        msg_scale=jnp.full((E,), scale),
        gate_keep=keep,
        msg_perm=perm,
    )
    extras = dict(vis=vis, covered=covered, events=events, dist_raw=dist_raw)
    return c, cstate, extras


def actor_forward(actor: CommActor, params, h, obs, c: CommIn, deterministic: bool, replay: bool,
                  sender_params=None, h_send=None):
    """Batched (over envs) actor call. With ``sender_params`` (cross-play), messages and gates
    come from the sender model while hidden state, attention and policy come from ``params``."""
    fwd = jax.vmap(lambda hh, oo, cc: actor.apply(params, hh, oo, cc, deterministic, replay))
    if sender_params is None:
        h_new, logits, aux = fwd(h, obs, c)
        return h_new, h_new, logits, aux

    def snd(pp, hh, oo, cc):
        return actor.apply(pp, hh, oo, cc, deterministic, replay, method=CommActor.sender)

    def rcv(hh, mm, gg, mk):
        return actor.apply(params, hh, mm, gg, mk, method=CommActor.receiver)

    hr, _, _, _, _, _ = jax.vmap(lambda hh, oo, cc: snd(params, hh, oo, cc))(h, obs, c)
    hs, m, alpha, p_on, g, aux = jax.vmap(lambda hh, oo, cc: snd(sender_params, hh, oo, cc))(h_send, obs, c)
    m, g = jax.vmap(apply_interventions)(m, g, c)
    logits, attn = jax.vmap(rcv)(hr, m, g, c.mask)
    aux = dict(aux, alpha=alpha, p_on=p_on, g=g, msg=m, attn=attn, hidden=hr)
    return hr, hs, logits, aux


def policy_step(setup: Setup, params, carry: Carry, key: jax.Array, *, deterministic: bool = False,
                greedy: bool = False, interv: Interv = NO_INTERV, sender_params=None, with_value: bool = False):
    """Advance a batch of environments by one step. Returns (new carry, step record)."""
    p = setup.env
    k_comm, k_act, k_env = jax.random.split(key, 3)
    es = carry.env_state
    E_ = setup.envm
    obs = jax.vmap(E_.get_obs, in_axes=(0, None))(es, p)
    amask = jax.vmap(E_.action_mask, in_axes=(0, None))(es, p)
    c, cstate, extras = build_comm_in(setup, carry, k_comm, interv)
    h_new, hs_new, logits, aux = actor_forward(
        setup.actor, params["actor"], carry.h, obs, c, deterministic, False, sender_params, carry.h_send
    )
    logits = mask_logits(logits, amask)
    if greedy:
        action = jnp.argmax(logits, -1)
    else:
        action = jax.random.categorical(k_act, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], -1)[..., 0] + aux["gate_logp"]
    gstate = jax.vmap(E_.global_state, in_axes=(0, None))(es, p)
    hlabel = jax.vmap(E_.human_label, in_axes=(0, None))(es, p)
    E = carry.h.shape[0]
    new_es, reward, done = jax.vmap(E_.step, in_axes=(0, 0, None, 0))(es, action, p, jax.random.split(k_env, E))
    rec = dict(
        obs=obs, comm=c._replace(g_given=aux["g_raw"]), action=action, logp=logp, logits=logits, amask=amask,
        reward=reward, done=done, gstate=gstate, human_label=hlabel, aux=aux,
        agent_pos=es.agent_pos, landmark_pos=es.landmark_pos, **extras,
    )
    if with_value:
        rec["value"] = jax.vmap(lambda g: setup.critic.apply(params["critic"], g))(gstate)
    new_carry = Carry(new_es, cstate, h_new, hs_new)
    return new_carry, rec
