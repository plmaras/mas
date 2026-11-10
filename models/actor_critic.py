"""MAPPO networks: shared-parameter GRU actor with gated attention communication, and a
centralized critic that never sees messages.

One ``CommActor`` call processes all N agents of ONE environment (vmap over environments
outside). Parameters are shared across agents (the same Dense layers are applied along
the agent axis).

Per step:
  sender side   e_i = Enc(o_i);  h_i = GRU(h_i, e_i);  m_i = Msg(h_i)
                alpha_ij = GateNet(h_i, (p_j - p_i)/R, |p_j - p_i|/R)   (sender-side gate)
                g_ij = gate(alpha_ij) * mask_ij
  receiver side q_j = Wq h_j,  k_i = Wk m_i,  v_i = Wv m_i (no bias, so m=0 gives v=0)
                logits_ij = q_j.k_i / sqrt(d), set to -inf where g_ij = 0 (or link not permitted)
                plus a learned null key (q_j.k_null) with zero value -> no NaN when all closed
                c_j = sum_i a_ij g_ij v_i       (g multiplies v so the task loss trains alpha)
                pi_j = Policy([h_j, c_j])

Assumption: one communication round per step, messages affect only the current action
(they are not fed back into the GRU). This keeps the message-use (E6) tests one-step.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from core.comm import LEARNED_GATE_MODES
from models import gates as G


class CommIn(NamedTuple):
    """Per-environment communication inputs (all [N, ...]); stored in the rollout so the
    PPO update can replay exactly the same gate noise."""

    mask: jax.Array  # [N, N] bool, sender -> receiver link permitted
    rel: jax.Array  # [N, N, 2] (p_j - p_i) / R
    dist: jax.Array  # [N, N] |p_j - p_i| / R
    gate_u: jax.Array  # [N, N] uniform noise for the gate
    tok_u: jax.Array  # [N, V] Gumbel noise for discrete (language) messages
    g_ext: jax.Array  # [N, N] gates for external modes (full / hand rule / random)
    g_given: jax.Array  # [N, N] stored REINFORCE gate samples (replayed in the update)
    msg_scale: jax.Array  # [] 0 => messages zeroed (NoComm, C4, F1)
    gate_keep: jax.Array  # [N, N] multiplicative gate ablation (F2)
    msg_perm: jax.Array  # [N] int, receiver reads m[msg_perm[i]] on link i->j (F3)


def apply_interventions(m, g, c: CommIn):
    m = m[c.msg_perm] * c.msg_scale
    g = g * c.gate_keep
    return m, g


class CommActor(nn.Module):
    n_actions: int
    obs_dim: int
    hidden_dim: int = 64
    msg_dim: int = 16
    attn_dim: int = 32
    gate_hidden: int = 16
    gate_mode: str = "hard_concrete"
    gate_scope: str = "link"  # "link" (per directed link) or "sender" (IC3Net-style)
    message_type: str = "continuous"  # continuous | language | ae
    vocab_size: int = 9
    hc_beta: float = 0.5
    hc_gamma: float = -0.1
    hc_zeta: float = 1.1
    gate_init_logit: float = 2.0
    gumbel_tau: float = 1.0

    def setup(self):
        H = self.hidden_dim
        self.enc1 = nn.Dense(H)
        self.enc2 = nn.Dense(H)
        self.gru = nn.GRUCell(features=H)
        if self.message_type == "continuous":
            self.msg_layer = nn.Dense(self.msg_dim)
        elif self.message_type == "language":
            self.tok_layer = nn.Dense(self.vocab_size)
        elif self.message_type == "ae":
            self.ae_enc1 = nn.Dense(H)
            self.ae_enc2 = nn.Dense(self.msg_dim)
            self.ae_dec1 = nn.Dense(H)
            self.ae_dec2 = nn.Dense(self.obs_dim)
        else:
            raise ValueError(self.message_type)
        self.gate_h = nn.Dense(self.gate_hidden)
        self.gate_geo = nn.Dense(self.gate_hidden, use_bias=False)
        self.gate_out = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.constant(self.gate_init_logit),
        )
        self.q = nn.Dense(self.attn_dim)
        self.k = nn.Dense(self.attn_dim)
        self.v = nn.Dense(H, use_bias=False)
        self.null_key = self.param("null_key", nn.initializers.normal(1.0), (self.attn_dim,))
        self.pi1 = nn.Dense(H)
        self.pi2 = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))

    # ------------------------------------------------------------------ sender
    def sender(self, h_prev, obs, c: CommIn, deterministic: bool = False, replay_gates: bool = False):
        n = obs.shape[0]
        x = nn.relu(self.enc2(nn.relu(self.enc1(obs))))
        h, _ = self.gru(h_prev, x)
        aux = {}

        # ---- message
        if self.message_type == "continuous":
            m = jnp.tanh(self.msg_layer(h))
        elif self.message_type == "language":
            tl = self.tok_layer(h)
            aux["tok_logits"] = tl
            if deterministic:
                m = jax.nn.one_hot(jnp.argmax(tl, -1), self.vocab_size)
            else:  # straight-through Gumbel-softmax
                y = jax.nn.softmax((tl + c.tok_u) / self.gumbel_tau)
                hard = jax.nn.one_hot(jnp.argmax(y, -1), self.vocab_size)
                m = hard + y - jax.lax.stop_gradient(y)
        else:  # AE-Comm: message trained only by reconstruction, RL gradient blocked
            z = jnp.tanh(self.ae_enc2(nn.relu(self.ae_enc1(obs))))
            recon = self.ae_dec2(nn.relu(self.ae_dec1(z)))
            aux["recon_err"] = jnp.mean((recon - obs) ** 2, axis=-1)
            m = jax.lax.stop_gradient(z)

        # ---- gate
        mask = c.mask.astype(jnp.float32)
        gate_logp = jnp.zeros((n,))
        gate_ent = jnp.zeros((n,))
        # The gate net is always evaluated so that every gate mode has identical parameters
        # (parameter parity between NoComm / FullComm / HandRule / gated models).
        if self.gate_scope == "sender":
            a_s = self.gate_out(jnp.tanh(self.gate_h(h)))[:, 0]
            alpha = jnp.broadcast_to(a_s[:, None], (n, n))
        else:
            geo = jnp.concatenate([c.rel, c.dist[..., None]], axis=-1)
            z = jnp.tanh(self.gate_h(h)[:, None, :] + self.gate_geo(geo))
            alpha = self.gate_out(z)[..., 0]
        if self.gate_mode in LEARNED_GATE_MODES:
            if self.gate_mode == "hard_concrete":
                p_on = G.hc_prob_on(alpha, self.hc_beta, self.hc_gamma, self.hc_zeta)
                if deterministic:
                    g = G.hc_deterministic(alpha, self.hc_gamma, self.hc_zeta)
                else:
                    g = G.hc_sample(alpha, c.gate_u, self.hc_beta, self.hc_gamma, self.hc_zeta)
            elif self.gate_mode == "straight_through":
                p_on = jax.nn.sigmoid(alpha)
                g = (p_on > 0.5).astype(jnp.float32) if deterministic else G.st_sample(alpha, c.gate_u)
            else:  # reinforce (score-function gradient through the PPO ratio)
                p_on = jax.nn.sigmoid(alpha)
                if deterministic:
                    g = (p_on > 0.5).astype(jnp.float32)
                elif replay_gates:
                    g = c.g_given
                else:
                    g = (c.gate_u < p_on).astype(jnp.float32)
                g = jax.lax.stop_gradient(g)
                if self.gate_scope == "sender":
                    gate_logp = G.bernoulli_logp(g[:, 0], alpha[:, 0])
                    gate_ent = G.bernoulli_entropy(p_on[:, 0])
                else:
                    gate_logp = jnp.sum(G.bernoulli_logp(g, alpha) * mask, axis=1)
                    gate_ent = jnp.sum(G.bernoulli_entropy(p_on) * mask, axis=1)
        else:
            alpha = jnp.zeros((n, n))  # not used by external gate modes
            p_on = c.g_ext
            g = c.g_ext
        aux.update(g_raw=g, gate_logp=gate_logp, gate_entropy=gate_ent)
        return h, m, alpha, p_on * mask, g * mask, aux

    # ---------------------------------------------------------------- receiver
    def receiver(self, h, m, g, mask):
        d = self.attn_dim
        q = self.q(h)
        k = self.k(m)
        v = self.v(m)
        logits = (k @ q.T) / jnp.sqrt(d)  # [sender i, receiver j]
        open_ = (g > 0) & (mask > 0)
        logits = jnp.where(open_, logits, -1e9)
        null = (q @ self.null_key) / jnp.sqrt(d)  # [receiver]
        w = jax.nn.softmax(jnp.concatenate([logits, null[None, :]], axis=0), axis=0)
        attn = w[:-1] * open_
        ctx = jnp.einsum("ij,ih->jh", attn * g, v)
        out = self.pi2(nn.relu(self.pi1(jnp.concatenate([h, ctx], axis=-1))))
        return out, attn

    def __call__(self, h_prev, obs, c: CommIn, deterministic: bool = False, replay_gates: bool = False):
        h, m, alpha, p_on, g, aux = self.sender(h_prev, obs, c, deterministic, replay_gates)
        m, g = apply_interventions(m, g, c)
        logits, attn = self.receiver(h, m, g, c.mask)
        aux.update(alpha=alpha, p_on=p_on, g=g, msg=m, attn=attn, hidden=h)
        return h, logits, aux


class Critic(nn.Module):
    """Centralized, permutation-invariant (DeepSets) value function V(s)."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, gs):
        H = self.hidden_dim
        a = nn.relu(nn.Dense(H)(gs.agents))
        a = nn.relu(nn.Dense(H)(a)).mean(axis=-2)
        lm = nn.relu(nn.Dense(H)(gs.landmarks))
        lm = nn.relu(nn.Dense(H)(lm)).mean(axis=-2)
        x = jnp.concatenate([a, lm, gs.time], axis=-1)
        x = nn.relu(nn.Dense(H)(x))
        x = nn.relu(nn.Dense(H)(x))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[..., 0]


class GridCritic(nn.Module):
    """Centralized critic for grid tasks (wildfire): a narrow per-cell MLP (cell features include
    the cell position) with mean + max pooling over cells, plus a DeepSets encoder of the agents.
    Works for any grid size (zero-shot transfer across N). Only dense layers, so vmapping over
    independently trained runs gives batched matmuls (a vmapped conv would become a slow grouped
    conv). ~14x cheaper than the 128-wide per-cell DeepSets critic. Never sees messages."""

    hidden_dim: int = 128
    cell_dim: int = 32

    @nn.compact
    def __call__(self, gs):
        H, C = self.hidden_dim, self.cell_dim
        c = nn.relu(nn.Dense(C)(gs.landmarks))
        c = nn.relu(nn.Dense(C)(c))
        grid = jnp.concatenate([c.mean(axis=-2), c.max(axis=-2)], -1)
        a = nn.relu(nn.Dense(H)(gs.agents))
        a = nn.relu(nn.Dense(H)(a)).mean(axis=-2)
        z = jnp.concatenate([grid, a, gs.time], axis=-1)
        z = nn.relu(nn.Dense(H)(z))
        z = nn.relu(nn.Dense(H)(z))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z)[..., 0]
