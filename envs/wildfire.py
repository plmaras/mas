"""Wildfire suppression ("FireWorkers") in pure JAX: a port of free-range-zoo ``wildfire_v0``.

Every function is pure and works on ONE environment; vectorise with ``jax.vmap``. The
fire / suppressant / equipment / capacity transitions, their order inside a step, the
reward and the termination rule follow free-range-zoo ``wildfire_v0`` (OASYS lab,
https://github.com/oasys-mas/free-range-zoo, files ``envs/wildfire/env/v0/wildfire.py`` and
``transitions/*.py``). Differences are switches in ``WildfireParams`` and are listed below.

Grid encoding (identical to v0)
-------------------------------
* ``fires[y, x]``  > 0 : burning, value = fire type = attack power needed to suppress it
                   < 0 : not burning but flammable (``-type``)
                   = 0 : not flammable
* ``intensity``     0 = not lit; ``ignition_temp`` when it ignites; burns out at
                    ``num_fire_states - 1`` (then stays ``fires < 0`` with intensity > 0 and
                    can never re-ignite)
* ``fuel``          number of (re-)ignitions left when ``fire_fuel`` is on.

One step (v0 order): agents act -> suppressant decrease -> equipment -> refill -> tank
capacity -> fire intensity increase -> fire intensity decrease -> fire spread -> rewards ->
termination. Team reward = put-out rewards + burnout penalties (+ bad-action penalties, +
termination reward once). The fire spread probability of an unlit cell is the sum of the
spread weights of its burning 4-neighbours (v0's 3x3 cross filter), plus the random-ignition
probability.

Extensions for the communication study (switches; v0 behaviour = ``mobile=False`` and an
``obs_radius`` at least the grid size):
* ``obs_radius``: agents observe only cells within Chebyshev distance ``obs_radius`` (v0: the
  full grid). Without this, nothing is private and messages carry no information.
* ``mobile``: agents may move one cell per step (v0 firefighters are fixed in place).
* Procedural layouts: each episode samples a fire-type map, initial fires and agent cells
  (v0 takes them from a fixed configuration). Grid side = round(sqrt(N / agent_density)), so
  the number of cells per agent and of initial fires per cell stay constant as N grows.
* Fixed-size action space with masks (v0 builds a variable action list per step):
  ``0`` = noop (refills the tank, as in v0), ``1..4`` = move N/S/W/E, ``5..`` = suppress the
  cell at offset (dy, dx) with ``max(|dy|,|dx|) <= max range``. Invalid actions are masked
  (v0 default ``show_bad_actions=False`` hides them too); a masked action that is still
  taken counts as a bad action (``bad_attack_penalty``) and does nothing.
* Fixed-length episodes for the recurrent MAPPO learner. After v0's termination (all fires
  out, and with ``fire_fuel`` all fuel spent) the state is absorbing (fires cleared, as v0
  does) and rewards are 0.
* Rewards are shared by the team and divided by ``n_agents`` when ``normalize_reward`` (so
  returns are comparable across N); v0's per-cell reward table is given per fire type.

Interface shared with ``envs/spread.py`` (used by the rollout, evaluation and analysis):
positions ``agent_pos[N, 2]`` (row, col as floats), "landmarks" = grid cells
(``landmark_pos[L, 2]``, L = H*W), ``vis[i, l]`` = cell l inside agent i's view,
``covered[l]`` = cell l is NOT burning. So the communication event of the spec, "sender sees an
uncovered landmark the receiver cannot see", reads here "sender sees a burning cell outside
the receiver's view". Success = all fires out at some step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

BIG = 1e9
N_CHANNELS = 6
N_SELF = 9
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], np.int32)  # noop, N, S, W, E as (dy, dx)


def _tup(x):
    """Lists (from YAML) -> nested tuples, so the params dataclass stays hashable."""
    return tuple(_tup(v) for v in x) if isinstance(x, (list, tuple)) else x


@dataclass(frozen=True)
class WildfireParams:
    name: str = "wildfire"
    n_agents: int = 4
    agent_density: float = 0.0625  # agents per cell; grid side = round(sqrt(n_agents / agent_density))
    grid_size: int | None = None  # explicit grid side (overrides agent_density)
    max_steps: int = 50
    obs_radius: int = 1  # Chebyshev radius of the local view (cells)
    mobile: bool = True
    agent_positions: tuple | None = None  # fixed (row, col) per agent; None = random each episode
    # ---- fire layout (procedural)
    fire_type_probs: tuple = (0.1, 0.4, 0.4, 0.1)  # P(cell type = 0 (not flammable), 1, 2, 3)
    init_fire_density: float = 0.05  # initially burning cells per cell (at least 1)
    # ---- fire dynamics (v0 FireConfiguration / StochasticConfiguration)
    num_fire_states: int = 5
    ignition_temp: int = 1
    intensity_increase_probability: float = 0.2
    intensity_decrease_probability: float = 0.8
    extra_power_decrease_bonus: float = 0.1
    burnout_probability: float = 0.2
    base_spread_rate: float = 0.06
    max_spread_rate: float = 67.0
    cell_size: float = 200.0
    wind_direction: float = 0.0
    random_ignition_probability: float = 0.0
    initial_fuel: int = 2
    special_burnout_probability: bool = True
    fire_increase: bool = True
    fire_decrease: bool = True
    fire_spread: bool = True
    realistic_fire_spread: bool = False
    random_fire_ignition: bool = False
    fire_fuel: bool = True
    # ---- agents (v0 AgentConfiguration)
    fire_reduction_power: float = 1.0
    attack_range: int = 1
    suppressant_states: int = 3
    initial_suppressant: float = 3.0
    initial_capacity: float = 3.0
    suppressant_decrease_probability: float = 1.0
    suppressant_refill_probability: float = 0.8
    possible_capacities: tuple = (2.0, 3.0, 4.0)
    capacity_probabilities: tuple = (0.0, 1.0, 0.0)
    tank_switch_probability: float = 1.0
    equipment_states: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # (capacity, power, range) bonus
    initial_equipment_state: int = 2
    repair_probability: float = 1.0
    degrade_probability: float = 1.0
    critical_error_probability: float = 0.0
    equipment_condition_on_actions: bool = False
    suppressant_decrease: bool = False
    suppressant_refill: bool = True
    tank_switch: bool = False
    critical_error: bool = False
    degrade: bool = False
    repair: bool = False
    # ---- rewards (v0 RewardConfiguration; rewards per fire type 0..3)
    fire_rewards_by_type: tuple = (0.0, 1.0, 2.0, 3.0)
    bad_attack_penalty: float = -1.0
    burnout_penalty: float = -1.0
    burnout_penalty_scaled: bool = False
    termination_reward: float = 0.0
    termination_kappa: float = 0.0
    normalize_reward: bool = True
    success_max_burned_frac: float = 0.1  # success = all fires out with at most this share of flammable cells burned
    reward_scale: float | None = None  # read by the learner (overrides algo.reward_scale)

    # ------------------------------------------------------------------ derived sizes
    @property
    def side(self) -> int:
        if self.grid_size is not None:
            return int(self.grid_size)
        return max(3, int(round(math.sqrt(self.n_agents / self.agent_density))))

    @property
    def H(self) -> int:
        return self.side

    @property
    def W(self) -> int:
        return self.side

    @property
    def n_cells(self) -> int:
        return self.H * self.W

    @property
    def n_landmarks(self) -> int:
        return self.n_cells

    @property
    def n_init_fires(self) -> int:
        return max(1, int(round(self.init_fire_density * self.n_cells)))

    @property
    def max_type(self) -> int:
        return len(self.fire_type_probs) - 1

    @property
    def max_range(self) -> int:
        bonus = max(max(e[2] for e in self.equipment_states), 0.0)
        return int(math.floor(self.attack_range + bonus))

    @property
    def max_capacity(self) -> float:
        bonus = max(max(e[0] for e in self.equipment_states), 0.0)
        return float(max(self.possible_capacities + (self.initial_capacity,)) + bonus)

    @property
    def max_power(self) -> float:
        bonus = max(max(e[1] for e in self.equipment_states), 0.0)
        return float(self.fire_reduction_power + bonus)

    @property
    def n_actions(self) -> int:
        return 5 + (2 * self.max_range + 1) ** 2

    @property
    def obs_dim(self) -> int:
        return (2 * self.obs_radius + 1) ** 2 * N_CHANNELS + N_SELF

    @property
    def spread_weights(self):
        """(N, E, S, W) weights of v0's 3x3 cross filter: weight N multiplies the burning
        state of the cell ABOVE (row - 1), etc."""
        if not self.fire_spread:
            return (0.0, 0.0, 0.0, 0.0)
        if self.realistic_fire_spread:
            csf = self.base_spread_rate / self.cell_size
            msf = 1 - self.base_spread_rate / self.max_spread_rate
            cos = [math.cos(0 - self.wind_direction), math.cos(0.5 * math.pi - self.wind_direction),
                   math.cos(math.pi - self.wind_direction), math.cos(1.5 * math.pi - self.wind_direction)]
            return tuple(csf / (1 - c * msf) for c in cos)
        return (self.base_spread_rate,) * 4

    @classmethod
    def from_cfg(cls, cfg) -> "WildfireParams":
        from omegaconf import DictConfig, OmegaConf

        d = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        kw = {k: _tup(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        p = cls(**kw)
        p.validate()
        return p

    def validate(self):
        assert abs(sum(self.fire_type_probs) - 1) < 1e-6, "fire_type_probs must sum to 1"
        assert len(self.fire_rewards_by_type) == len(self.fire_type_probs)
        assert self.num_fire_states >= 3 and 0 < self.ignition_temp < self.num_fire_states - 1
        assert len(self.possible_capacities) == len(self.capacity_probabilities)
        assert 0 <= self.initial_equipment_state < len(self.equipment_states)
        if self.agent_positions is not None:
            assert len(self.agent_positions) == self.n_agents


class EnvState(NamedTuple):
    fires: jax.Array  # [H, W] int32
    intensity: jax.Array  # [H, W] int32
    fuel: jax.Array  # [H, W] int32
    agent_pos: jax.Array  # [N, 2] float32 (row, col)
    suppressants: jax.Array  # [N] float32
    capacity: jax.Array  # [N] float32
    equipment: jax.Array  # [N] int32
    landmark_pos: jax.Array  # [L, 2] float32 cell coordinates (constant)
    t: jax.Array  # int32
    ever_all_covered: jax.Array  # bool: success = all fires out at some step with <= success_max_burned_frac burned
    terminated: jax.Array  # bool: v0 termination happened
    num_burnouts: jax.Array  # int32
    num_putouts: jax.Array  # int32
    init_flammable: jax.Array  # int32: flammable cells at reset (for the burned-area fraction)


class GlobalState(NamedTuple):
    agents: jax.Array  # [N, N_SELF]
    landmarks: jax.Array  # [L, 7]
    time: jax.Array  # [4]


# ---------------------------------------------------------------------------- helpers
def pairwise_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    diff = b[None, :, :] - a[:, None, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


def cell_positions(p: WildfireParams) -> jax.Array:
    yy, xx = jnp.meshgrid(jnp.arange(p.H), jnp.arange(p.W), indexing="ij")
    return jnp.stack([yy.ravel(), xx.ravel()], -1).astype(jnp.float32)


def fight_offsets(p: WildfireParams) -> np.ndarray:
    r = p.max_range
    return np.array([[dy, dx] for dy in range(-r, r + 1) for dx in range(-r, r + 1)], np.int32)


def action_dirs(p: WildfireParams) -> np.ndarray:
    """Movement direction of every action (row, col); suppress / noop actions do not move."""
    d = np.zeros((p.n_actions, 2), np.float32)
    d[1:5] = MOVES[1:5]
    return d


def n_actions(p: WildfireParams) -> int:
    return p.n_actions


def _eq_bonus(p: WildfireParams, equipment, col: int):
    tab = jnp.asarray(p.equipment_states, jnp.float32)
    return tab[equipment, col]


def _effective(p: WildfireParams, s: EnvState):
    power = p.fire_reduction_power + _eq_bonus(p, s.equipment, 1)
    rng = jnp.floor(p.attack_range + _eq_bonus(p, s.equipment, 2)).astype(jnp.int32)
    cap = s.capacity + _eq_bonus(p, s.equipment, 0)
    return power, rng, cap


def burning(s: EnvState) -> jax.Array:
    return (s.fires > 0) & (s.intensity > 0)


def _shift(x, dy: int, dx: int):
    """``out[y, x] = x[y + dy, x + dx]`` with zero padding."""
    H, W = x.shape
    xp = jnp.pad(x, ((1, 1), (1, 1)))
    return jax.lax.dynamic_slice(xp, (1 + dy, 1 + dx), (H, W))


# ---------------------------------------------------------------------------- reset
def reset(key: jax.Array, p: WildfireParams) -> EnvState:
    k_type, k_fire, k_pos = jax.random.split(key, 3)
    H, W, N = p.H, p.W, p.n_agents
    probs = jnp.asarray(p.fire_type_probs, jnp.float32)
    types = jax.random.choice(k_type, probs.shape[0], (H, W), p=probs).astype(jnp.int32)
    # initial fires: n_init_fires random flammable cells (top-k of random scores)
    scores = jax.random.uniform(k_fire, (H * W,)) + jnp.where(types.ravel() > 0, 0.0, -BIG)
    _, idx = jax.lax.top_k(scores, p.n_init_fires)
    lit = jnp.zeros((H * W,), bool).at[idx].set(True).reshape(H, W) & (types > 0)
    fires = jnp.where(lit, types, -types)
    intensity = jnp.where(lit, p.ignition_temp, 0).astype(jnp.int32)
    fuel = jnp.where(fires != 0, p.initial_fuel, 0).astype(jnp.int32)
    if p.agent_positions is not None:
        pos = jnp.asarray(p.agent_positions, jnp.float32)
    else:
        cells = jax.random.randint(k_pos, (N,), 0, H * W)
        pos = jnp.stack([cells // W, cells % W], -1).astype(jnp.float32)
    s = EnvState(
        fires=fires.astype(jnp.int32), intensity=intensity, fuel=fuel, agent_pos=pos,
        suppressants=jnp.full((N,), p.initial_suppressant, jnp.float32),
        capacity=jnp.full((N,), p.initial_capacity, jnp.float32),
        equipment=jnp.full((N,), p.initial_equipment_state, jnp.int32),
        landmark_pos=cell_positions(p), t=jnp.zeros((), jnp.int32),
        ever_all_covered=jnp.zeros((), bool), terminated=jnp.zeros((), bool),
        num_burnouts=jnp.zeros((), jnp.int32), num_putouts=jnp.zeros((), jnp.int32),
        init_flammable=jnp.sum(types > 0).astype(jnp.int32),
    )
    return s._replace(ever_all_covered=~jnp.any(s.fires > 0))


# ---------------------------------------------------------------------------- actions
def _decode(p: WildfireParams, s: EnvState, actions):
    """Per agent: is_noop, move (dy, dx), target cell (y, x), is_fight, valid_fight."""
    offs = jnp.asarray(fight_offsets(p))
    pos = s.agent_pos.astype(jnp.int32)
    a = actions.astype(jnp.int32)
    is_noop = a == 0
    is_move = (a >= 1) & (a <= 4)
    is_fight = a >= 5
    move = jnp.asarray(MOVES)[jnp.clip(a, 0, 4)] * is_move[:, None]
    off = offs[jnp.clip(a - 5, 0, offs.shape[0] - 1)]
    tgt = pos + off
    inb = (tgt[:, 0] >= 0) & (tgt[:, 0] < p.H) & (tgt[:, 1] >= 0) & (tgt[:, 1] < p.W)
    ty, tx = jnp.clip(tgt[:, 0], 0, p.H - 1), jnp.clip(tgt[:, 1], 0, p.W - 1)
    lit_t = burning(s)[ty, tx]
    _, rng, _ = _effective(p, s)
    in_range = jnp.max(jnp.abs(off), axis=-1) <= rng
    valid_fight = is_fight & inb & lit_t & in_range & (s.suppressants > 0)
    return is_noop, is_move, move, ty, tx, is_fight, valid_fight


def action_mask(s: EnvState, p: WildfireParams) -> jax.Array:
    """``[N, A]`` bool: actions an agent may take now (noop is always allowed)."""
    N, A = p.n_agents, p.n_actions
    acts = jnp.broadcast_to(jnp.arange(A), (N, A))
    pos = s.agent_pos.astype(jnp.int32)
    mv = jnp.asarray(MOVES)[1:5]  # [4, 2]
    dest = pos[:, None, :] + mv[None]
    move_ok = (dest[..., 0] >= 0) & (dest[..., 0] < p.H) & (dest[..., 1] >= 0) & (dest[..., 1] < p.W)
    move_ok = move_ok & p.mobile
    fight_ok = jax.vmap(lambda a: _decode(p, s, a)[6], in_axes=1, out_axes=1)(acts[:, 5:])
    return jnp.concatenate([jnp.ones((N, 1), bool), move_ok, fight_ok], axis=1)


# ---------------------------------------------------------------------------- transitions (v0)
def _suppressant_decrease(p, supp, users, u):
    dec = users & (u < p.suppressant_decrease_probability) if p.suppressant_decrease else users
    return jnp.clip(jnp.where(dec, supp - 1, supp), 0, None)


def _equipment(p, eq, u, fought):
    E = len(p.equipment_states)
    pristine = eq == E - 1
    damaged = eq == 0
    inter = ~pristine & ~damaged
    rep_p, deg_p, crit_p = p.repair_probability, p.degrade_probability, p.critical_error_probability
    if p.equipment_condition_on_actions:
        rep_p = rep_p * (damaged & ~fought)
        deg_p = deg_p * ((pristine | inter) & fought)
        crit_p = crit_p * (pristine & fought)
    repairs = (damaged & (u < rep_p)) if p.repair else damaged
    eq = jnp.where(repairs, E - 1, eq)
    criticals = (pristine & (u < crit_p)) if p.critical_error else jnp.zeros_like(pristine)
    eq = jnp.where(criticals, 0, eq)
    degrades = ((pristine | inter) & (u < deg_p)) if p.degrade else (pristine | inter)
    degrades = degrades & ~criticals
    return jnp.where(degrades, eq - 1, eq).astype(jnp.int32)


def _refill(p, supp, cap, eq, refills, u):
    inc = refills & (u < p.suppressant_refill_probability) if p.suppressant_refill else refills
    supp = jnp.where(inc, cap + _eq_bonus(p, eq, 0), supp)
    return supp, inc


def _capacity(p, supp, cap, targets, u0, u1):
    cum = jnp.cumsum(jnp.asarray(p.capacity_probabilities, jnp.float32))
    idx = jnp.clip(jnp.searchsorted(cum, u0, side="left"), 0, len(p.possible_capacities) - 1)
    new_max = jnp.asarray(p.possible_capacities, jnp.float32)[idx]
    switch = targets & (u1 < p.tank_switch_probability) if p.tank_switch else targets
    bonus = supp - cap
    cap = jnp.where(switch, new_max, cap)
    supp = jnp.where(switch, new_max + bonus, supp)
    return supp, cap


def _fire_increase(p, fires, intensity, fuel, attack, u):
    required = jnp.maximum(fires, 0).astype(jnp.float32)
    diff = required - attack
    lit = (fires > 0) & (intensity > 0)
    unmet = (diff > 0) & lit
    almost = unmet & (intensity == p.num_fire_states - 2)
    increasing = unmet & ~almost
    prob = jnp.where(increasing, p.intensity_increase_probability if p.fire_increase else 1.0, 0.0)
    prob = jnp.where(almost, p.burnout_probability if p.special_burnout_probability else p.intensity_increase_probability, prob)
    inc = u < jnp.clip(prob, 0, 1)
    intensity = intensity + inc
    burned = inc & (intensity >= p.num_fire_states - 1)
    fires = jnp.where(burned, -fires, fires)
    fuel = jnp.where(burned, jnp.maximum(fuel - 1, 0), fuel)
    return fires, intensity, fuel, burned


def _fire_decrease(p, fires, intensity, fuel, attack, u):
    required = jnp.maximum(fires, 0).astype(jnp.float32)
    diff = required - attack
    lit = (fires > 0) & (intensity > 0)
    met = (diff <= 0) & lit
    if p.fire_decrease:
        prob = jnp.where(met, p.intensity_decrease_probability - diff * p.extra_power_decrease_bonus, 0.0)
    else:
        prob = jnp.where(met, 1.0, 0.0)
    dec = u < jnp.clip(prob, 0, 1)
    intensity = intensity - dec
    put_out = dec & (intensity <= 0)
    fires = jnp.where(put_out, -fires, fires)
    fuel = jnp.where(put_out, fuel - 1, fuel)
    return fires, intensity, fuel, put_out


def _fire_spread(p, fires, intensity, fuel, u):
    lit = ((fires > 0) & (intensity > 0)).astype(jnp.float32)
    wn, we, ws, ww = p.spread_weights
    prob = wn * _shift(lit, -1, 0) + ws * _shift(lit, 1, 0) + ww * _shift(lit, 0, -1) + we * _shift(lit, 0, 1)
    unlit = (fires < 0) & (intensity == 0)
    if p.fire_fuel:
        unlit = unlit & (fuel > 0)
    rnd = p.random_ignition_probability if p.random_fire_ignition else 0.0
    prob = jnp.where(unlit, prob + rnd, 0.0)
    ignite = u < prob
    fires = jnp.where(ignite, -fires, fires)
    intensity = jnp.where(ignite, p.ignition_temp, intensity)
    return fires, intensity.astype(jnp.int32)


def step(s: EnvState, actions: jax.Array, p: WildfireParams, key: jax.Array):
    """Apply integer actions ``[N]``. Returns ``(new_state, team_reward, done)``."""
    kf, ka = jax.random.split(key)
    uf = jax.random.uniform(kf, (3, p.H, p.W))
    ua = jax.random.uniform(ka, (5, p.n_agents))
    is_noop, is_move, move, ty, tx, is_fight, valid = _decode(p, s, actions)
    power, _, cap_eff = _effective(p, s)
    users = valid
    bad = is_fight & ~valid
    # masked move (e.g. a static agent, or off the grid) does nothing
    mv_ok = is_move & p.mobile
    dest = s.agent_pos.astype(jnp.int32) + move
    inb = (dest[:, 0] >= 0) & (dest[:, 0] < p.H) & (dest[:, 1] >= 0) & (dest[:, 1] < p.W)
    new_pos = jnp.where((mv_ok & inb)[:, None], dest, s.agent_pos.astype(jnp.int32)).astype(jnp.float32)
    attack = jnp.zeros((p.H, p.W), jnp.float32).at[ty, tx].add(jnp.where(users, power, 0.0))

    supp = _suppressant_decrease(p, s.suppressants, users, ua[0])
    eq = _equipment(p, s.equipment, ua[1], users)
    supp, increased = _refill(p, supp, s.capacity, eq, is_noop, ua[2])
    supp, cap = _capacity(p, supp, s.capacity, increased, ua[3], ua[4])
    fires, inten, fuel, burned = _fire_increase(p, s.fires, s.intensity, s.fuel, attack, uf[0])
    fires, inten, fuel, put_out = _fire_decrease(p, fires, inten, fuel, attack, uf[1])
    fires, inten = _fire_spread(p, fires, inten, fuel, uf[2])

    rew_tab = jnp.asarray(p.fire_rewards_by_type, jnp.float32)
    ftype = jnp.clip(jnp.abs(fires), 0, p.max_type)
    r_put = jnp.sum(jnp.where(put_out, rew_tab[ftype], 0.0))
    if p.burnout_penalty_scaled:
        r_burn = jnp.sum(jnp.where(burned, -rew_tab[ftype], 0.0))
    else:
        r_burn = p.burnout_penalty * jnp.sum(burned)
    r_bad = p.bad_attack_penalty * jnp.sum(bad) / p.n_agents  # v0: per-agent penalty; team = mean
    n_burn = s.num_burnouts + jnp.sum(burned)

    fires_out = jnp.max(fires) <= 0
    dead = fires_out & (jnp.sum(fuel) <= 0) if p.fire_fuel else fires_out
    fires = jnp.where(dead, 0, fires)
    newly = dead & ~s.terminated
    term_r = jnp.maximum(p.termination_reward - p.termination_kappa * jnp.log(s.num_burnouts + 1.0), 0.0)
    reward = jnp.where(s.terminated, 0.0, r_put + r_burn + r_bad) + jnp.where(newly, term_r, 0.0)
    if p.normalize_reward:
        reward = reward / p.n_agents
    t = s.t + 1
    new = s._replace(
        fires=fires.astype(jnp.int32), intensity=inten.astype(jnp.int32), fuel=fuel.astype(jnp.int32),
        agent_pos=new_pos, suppressants=supp, capacity=cap, equipment=eq, t=t,
        ever_all_covered=s.ever_all_covered | (~jnp.any(fires > 0) & (n_burn <= p.success_max_burned_frac * s.init_flammable)),
        terminated=s.terminated | dead,
        num_burnouts=n_burn.astype(jnp.int32), num_putouts=(s.num_putouts + jnp.sum(put_out)).astype(jnp.int32),
    )
    return new, reward, t >= p.max_steps


# ---------------------------------------------------------------------------- views
def visibility(s: EnvState, p: WildfireParams) -> jax.Array:
    """``vis[i, l]``: cell l within Chebyshev distance obs_radius of agent i."""
    d = jnp.max(jnp.abs(s.landmark_pos[None, :, :] - s.agent_pos[:, None, :]), axis=-1)
    return d <= p.obs_radius


def coverage(s: EnvState, p: WildfireParams) -> jax.Array:
    """``covered[l]``: cell l is not burning (the analogue of a covered landmark)."""
    return ~burning(s).ravel()


def _box_sum(sat, y0, y1, x0, x1):
    """Sum over rows y0..y1, cols x0..x1 (inclusive, already clipped) from a summed-area table
    ``sat`` of shape [H+1, W+1]; empty boxes give 0."""
    ok = (y1 >= y0) & (x1 >= x0)
    v = sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0]
    return jnp.where(ok, v, 0)


def events(s: EnvState, p: WildfireParams, mask: jax.Array) -> jax.Array:
    """``E[i, j]``: sender i sees a burning cell that is outside receiver j's view (only on
    permitted links). Equals ``core.comm.compute_events(vis, covered, mask)`` but costs
    O(N^2) via a summed-area table instead of O(N^2 L)."""
    lit = burning(s).astype(jnp.int32)
    sat = jnp.pad(jnp.cumsum(jnp.cumsum(lit, 0), 1), ((1, 0), (1, 0)))
    R = p.obs_radius
    pos = s.agent_pos.astype(jnp.int32)
    lo = jnp.maximum(pos - R, 0)
    hi = jnp.minimum(pos + R, jnp.array([p.H - 1, p.W - 1]))
    n_i = _box_sum(sat, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])  # [N]
    y0 = jnp.maximum(lo[:, None, 0], lo[None, :, 0])
    y1 = jnp.minimum(hi[:, None, 0], hi[None, :, 0])
    x0 = jnp.maximum(lo[:, None, 1], lo[None, :, 1])
    x1 = jnp.minimum(hi[:, None, 1], hi[None, :, 1])
    n_ij = _box_sum(sat, y0, y1, x0, x1)
    return ((n_i[:, None] - n_ij) > 0) & mask.astype(bool)


def _self_feats(s: EnvState, p: WildfireParams):
    power, rng, cap = _effective(p, s)
    n = p.n_agents
    return jnp.stack([
        s.agent_pos[:, 0] / max(p.H - 1, 1), s.agent_pos[:, 1] / max(p.W - 1, 1),
        s.suppressants / p.max_capacity, cap / p.max_capacity,
        s.equipment / max(len(p.equipment_states) - 1, 1), power / max(p.max_power, 1e-6),
        jnp.full((n,), s.t / p.max_steps), (s.suppressants > 0).astype(jnp.float32),
        rng / max(p.max_range, 1),
    ], -1).astype(jnp.float32)


def _channels(s: EnvState, p: WildfireParams):
    lit = burning(s)
    S = p.num_fire_states
    burned = (s.fires < 0) & (s.intensity >= S - 1)
    flam = (s.fires < 0) & (s.intensity == 0)
    if p.fire_fuel:
        flam = flam & (s.fuel > 0)
    pos = s.agent_pos.astype(jnp.int32)
    occ = jnp.zeros((p.H, p.W), jnp.float32).at[pos[:, 0], pos[:, 1]].add(1.0)
    return jnp.stack([
        lit * s.fires / max(p.max_type, 1), lit * s.intensity / (S - 1), burned.astype(jnp.float32),
        flam.astype(jnp.float32), occ, jnp.ones((p.H, p.W), jnp.float32),
    ], -1).astype(jnp.float32), occ


def get_obs(s: EnvState, p: WildfireParams) -> jax.Array:
    """Ego-centric (2R+1)^2 window x 6 channels (fire type, intensity, burned out, flammable,
    other agents, inside grid) + 9 own features. Same size for every N (zero-shot transfer)."""
    R = p.obs_radius
    ch, _ = _channels(s, p)
    padded = jnp.pad(ch, ((R, R), (R, R), (0, 0)))
    pos = s.agent_pos.astype(jnp.int32)
    win = jax.vmap(lambda yx: jax.lax.dynamic_slice(padded, (yx[0], yx[1], 0), (2 * R + 1, 2 * R + 1, N_CHANNELS)))(pos)
    win = win.at[:, R, R, 4].add(-1.0)  # do not count yourself as another agent
    win = win.at[..., 4].set(jnp.clip(win[..., 4], 0, 3) / 3)
    return jnp.concatenate([win.reshape(p.n_agents, -1), _self_feats(s, p)], -1)


def global_state(s: EnvState, p: WildfireParams) -> GlobalState:
    ch, _ = _channels(s, p)
    lp = s.landmark_pos
    cells = jnp.concatenate([
        lp / jnp.array([max(p.H - 1, 1), max(p.W - 1, 1)]), ch[..., :4].reshape(p.n_cells, 4),
        (s.fuel / max(p.initial_fuel, 1)).reshape(p.n_cells, 1),
    ], -1).astype(jnp.float32)
    lit = burning(s)
    burned = (s.fires < 0) & (s.intensity >= p.num_fire_states - 1)
    time = jnp.stack([s.t / p.max_steps, lit.mean() * 10, burned.mean(), jnp.mean(s.suppressants > 0)]).astype(jnp.float32)
    return GlobalState(agents=_self_feats(s, p), landmarks=cells, time=time)


def human_label(s: EnvState, p: WildfireParams, n_dirs: int = 8) -> jax.Array:
    """Compass octant (0..7) of the nearest visible burning cell, or 8 ("none")."""
    cand = visibility(s, p) & ~coverage(s, p)[None, :]
    rel = s.landmark_pos[None, :, :] - s.agent_pos[:, None, :]
    d = jnp.where(cand, jnp.linalg.norm(rel, axis=-1), BIG)
    idx = jnp.argmin(d, axis=1)
    r = rel[jnp.arange(p.n_agents), idx]
    ang = jnp.arctan2(r[:, 1], r[:, 0])
    octant = jnp.floor((ang + jnp.pi) / (2 * jnp.pi / n_dirs)).astype(jnp.int32) % n_dirs
    return jnp.where(jnp.any(cand, axis=1), octant, n_dirs)


def final_stats(s: EnvState, p: WildfireParams) -> dict:
    """Episode-level outcome numbers (logged per evaluation episode)."""
    burned = (s.fires < 0) & (s.intensity >= p.num_fire_states - 1)
    return dict(
        final_burned_frac=jnp.sum(burned) / jnp.maximum(s.init_flammable, 1),
        final_burning_cells=jnp.sum(burning(s)).astype(jnp.float32),
        final_putouts=s.num_putouts.astype(jnp.float32),
        final_burnouts=s.num_burnouts.astype(jnp.float32),
    )
