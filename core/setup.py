"""Builds the environment parameters and networks from a Hydra config."""
from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from types import ModuleType

from envs import make_params
from models.actor_critic import CommActor, Critic, GridCritic


@dataclass(frozen=True)
class Setup:
    env: object  # env params dataclass (SpreadParams | WildfireParams)
    envm: ModuleType  # env module (functional interface, see envs/__init__.py)
    actor: CommActor
    critic: Critic
    gate_mode: str
    k_neighbors: int
    comm_radius: float | None
    hand_rule_d: float
    density: float
    switch_rate: float
    zero_messages: bool
    message_type: str
    vocab_size: int

    @property
    def msg_size(self) -> int:
        return self.vocab_size if self.message_type == "language" else self.actor.msg_dim


def make_setup(cfg: DictConfig, env_overrides: dict | None = None, comm_overrides: dict | None = None) -> Setup:
    """``env_overrides`` / ``comm_overrides`` change settings at evaluation time (e.g. N for
    zero-shot transfer, a control mask for E4) without touching trained parameters."""
    env_cfg = OmegaConf.merge(cfg.env, env_overrides or {})
    comm = OmegaConf.merge(cfg.comm, comm_overrides or {})
    envm, env = make_params(env_cfg)
    hc = comm.hard_concrete
    actor = CommActor(
        n_actions=int(envm.n_actions(env)),
        obs_dim=env.obs_dim,
        hidden_dim=int(cfg.algo.hidden_dim),
        msg_dim=int(comm.msg_dim),
        attn_dim=int(comm.attn_dim),
        gate_hidden=int(comm.gate_hidden),
        gate_mode=str(comm.gate_mode),
        gate_scope=str(comm.gate_scope),
        message_type=str(comm.message_type),
        vocab_size=int(comm.vocab_size),
        hc_beta=float(hc.beta),
        hc_gamma=float(hc.gamma),
        hc_zeta=float(hc.zeta),
        gate_init_logit=float(comm.gate_init_logit),
        gumbel_tau=float(comm.gumbel_tau),
    )
    critic_type = str(cfg.algo.get("critic", "auto"))
    if critic_type == "auto":
        critic_type = "grid" if type(env).__name__ == "WildfireParams" else "deepsets"
    if critic_type == "grid":
        critic = GridCritic(hidden_dim=int(cfg.algo.critic_hidden_dim))
    else:
        critic = Critic(hidden_dim=int(cfg.algo.critic_hidden_dim))
    return Setup(
        env=env,
        envm=envm,
        actor=actor,
        critic=critic,
        gate_mode=str(comm.gate_mode),
        k_neighbors=int(comm.k_neighbors),
        comm_radius=None if comm.comm_radius is None else float(comm.comm_radius),
        hand_rule_d=float(comm.hand_rule_d),
        density=float(comm.density),
        switch_rate=float(comm.switch_rate),
        zero_messages=bool(comm.zero_messages),
        message_type=str(comm.message_type),
        vocab_size=int(comm.vocab_size),
    )
