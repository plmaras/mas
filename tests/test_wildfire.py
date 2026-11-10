"""Unit tests for the JAX wildfire env (envs/wildfire.py): v0 transition semantics on
hand-built states, visibility / events / observations, masks, termination, and jit/vmap."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.comm import compute_events
from envs import wildfire as W
from envs.wildfire_baselines import POLICIES, rollout_stats

DET = W.WildfireParams(  # deterministic v0-style configuration on a 5x5 grid, 2 agents
    n_agents=2, grid_size=5, obs_radius=1, max_steps=10, num_fire_states=5, ignition_temp=2,
    fire_increase=False, fire_decrease=False, special_burnout_probability=False, intensity_increase_probability=1.0,
    fire_spread=False, fire_fuel=False, suppressant_refill=False, initial_suppressant=2.0, initial_capacity=2.0,
    possible_capacities=(1.0, 2.0, 3.0), capacity_probabilities=(0.0, 1.0, 0.0), normalize_reward=False,
    fire_rewards_by_type=(0.0, 1.0, 2.0, 3.0), burnout_penalty=-1.0, bad_attack_penalty=-10.0,
    success_max_burned_frac=1.0)


def make_state(p, fires, intensity, agents, fuel=None, supp=None):
    fires = jnp.asarray(fires, jnp.int32)
    intensity = jnp.asarray(intensity, jnp.int32)
    fuel = jnp.where(fires != 0, p.initial_fuel, 0) if fuel is None else jnp.asarray(fuel, jnp.int32)
    n = p.n_agents
    return W.EnvState(
        fires=fires, intensity=intensity, fuel=fuel.astype(jnp.int32), agent_pos=jnp.asarray(agents, jnp.float32),
        suppressants=jnp.full((n,), p.initial_suppressant) if supp is None else jnp.asarray(supp, jnp.float32),
        capacity=jnp.full((n,), p.initial_capacity), equipment=jnp.full((n,), p.initial_equipment_state, jnp.int32),
        landmark_pos=W.cell_positions(p), t=jnp.array(0), ever_all_covered=jnp.array(False),
        terminated=jnp.array(False), num_burnouts=jnp.array(0), num_putouts=jnp.array(0),
        init_flammable=jnp.sum(fires != 0).astype(jnp.int32))


def fight(p, dy, dx):
    """Action index that suppresses the cell at offset (dy, dx)."""
    offs = W.fight_offsets(p)
    return 5 + int(np.flatnonzero((offs[:, 0] == dy) & (offs[:, 1] == dx))[0])


def one_fire(p, typ=2, inten=2, at=(2, 2)):
    f = -np.ones((p.H, p.W), int)
    i = np.zeros((p.H, p.W), int)
    f[at] = typ
    i[at] = inten
    return f, i


def test_reset_encoding_and_counts():
    p = W.WildfireParams()
    s = W.reset(jax.random.PRNGKey(0), p)
    f, i, fu = map(np.asarray, (s.fires, s.intensity, s.fuel))
    assert f.shape == (p.H, p.W) == (8, 8)
    assert (f > 0).sum() == p.n_init_fires == 3
    assert np.all(i[f > 0] == p.ignition_temp) and np.all(i[f <= 0] == 0)
    assert np.all(fu[f != 0] == p.initial_fuel) and np.all(fu[f == 0] == 0)
    assert np.all(np.abs(f) <= p.max_type)
    pos = np.asarray(s.agent_pos)
    assert pos.min() >= 0 and pos[:, 0].max() < p.H and pos[:, 1].max() < p.W


def test_grid_scales_with_n():
    for n, side in ((4, 8), (10, 13), (50, 28), (100, 40)):
        p = W.WildfireParams(n_agents=n)
        assert p.side == side
        assert p.obs_dim == W.WildfireParams(n_agents=4).obs_dim  # fixed obs size -> zero-shot transfer


def test_one_agent_cannot_suppress_type2_fire_but_two_can():
    p = dataclasses.replace(DET, fire_fuel=True)  # fuel left -> no v0 termination (which would clear the grid)
    f, i = one_fire(p, typ=2, inten=2)
    s = make_state(p, f, i, [[2, 1], [2, 3]])
    # one agent attacks: power 1 < type 2 -> intensity increases (deterministic increase)
    s1, r1, _ = W.step(s, jnp.array([fight(p, 0, 1), 0]), p, jax.random.PRNGKey(0))
    assert int(s1.intensity[2, 2]) == 3 and int(s1.fires[2, 2]) == 2
    assert float(s1.suppressants[0]) == 1.0  # suppressant used
    # both attack: needs met -> decrease every step (non-stochastic), put out at intensity 0
    s2, r2, _ = W.step(s1, jnp.array([fight(p, 0, 1), fight(p, 0, -1)]), p, jax.random.PRNGKey(1))
    assert int(s2.intensity[2, 2]) == 2
    s3, _, _ = W.step(s2._replace(suppressants=jnp.full((2,), 2.0)), jnp.array([fight(p, 0, 1), fight(p, 0, -1)]), p, jax.random.PRNGKey(2))
    s4, r4, _ = W.step(s3, jnp.array([fight(p, 0, 1), fight(p, 0, -1)]), p, jax.random.PRNGKey(3))
    assert int(s4.intensity[2, 2]) == 0 and int(s4.fires[2, 2]) == -2  # put out: sign flipped
    assert float(r4) == 2.0  # reward of a type-2 fire
    assert int(s4.fuel[2, 2]) == p.initial_fuel - 1
    assert bool(s4.ever_all_covered)


def test_unattended_fire_burns_out_with_penalty():
    p = dataclasses.replace(DET, fire_fuel=True)
    f, i = one_fire(p, typ=1, inten=2)
    s = make_state(p, f, i, [[0, 0], [4, 4]])
    s1, r1, _ = W.step(s, jnp.array([0, 0]), p, jax.random.PRNGKey(0))  # 2 -> 3 (almost burned out)
    assert int(s1.intensity[2, 2]) == 3 and float(r1) == 0.0
    s2, r2, _ = W.step(s1, jnp.array([0, 0]), p, jax.random.PRNGKey(1))  # 3 -> 4 = burned out
    assert int(s2.intensity[2, 2]) == 4 and int(s2.fires[2, 2]) == -1
    assert float(r2) == -1.0 and int(s2.num_burnouts) == 1
    # burned-out cells never re-ignite (intensity > 0)
    p2 = dataclasses.replace(p, fire_spread=True, base_spread_rate=1.0)
    f2 = np.asarray(s2.fires).copy()
    f2[2, 3] = 1
    i2 = np.asarray(s2.intensity).copy()
    i2[2, 3] = 2
    s3, _, _ = W.step(make_state(p2, f2, i2, [[0, 0], [4, 4]]), jnp.array([0, 0]), p2, jax.random.PRNGKey(2))
    assert int(s3.fires[2, 2]) == -1 and int(s3.intensity[2, 2]) == 4


def test_spread_cross_filter_and_fuel():
    p = dataclasses.replace(DET, fire_spread=True, base_spread_rate=1.0, fire_increase=True,
                            intensity_increase_probability=0.0, fire_fuel=True, special_burnout_probability=True,
                            burnout_probability=0.0)
    f, i = one_fire(p, typ=1, inten=2)
    fuel = np.where(f != 0, 1, 0)
    fuel[1, 2] = 0  # north neighbour has no fuel -> cannot ignite
    s = make_state(p, f, i, [[0, 0], [4, 4]], fuel=fuel)
    s1, _, _ = W.step(s, jnp.array([0, 0]), p, jax.random.PRNGKey(0))
    lit = np.asarray(W.burning(s1))
    expect = np.zeros((5, 5), bool)
    expect[2, 2] = expect[3, 2] = expect[2, 1] = expect[2, 3] = True  # S, W, E ignite; N has no fuel
    np.testing.assert_array_equal(lit, expect)
    assert np.all(np.asarray(s1.intensity)[expect] == p.ignition_temp)


def test_refill_capacity_and_masks():
    p = dataclasses.replace(DET, suppressant_refill=False)
    f, i = one_fire(p)
    s = make_state(p, f, i, [[2, 1], [0, 0]], supp=[0.0, 2.0])
    am = np.asarray(W.action_mask(s, p))
    assert am[:, 0].all()  # noop always allowed
    assert not am[0, 5:].any()  # no suppressant -> cannot fight
    assert am[1, 1:5].tolist() == [False, True, False, True]  # corner (0,0): only S and E
    assert not am[1, 5:].any()  # fire out of range
    s1, _, _ = W.step(s, jnp.array([0, 0]), p, jax.random.PRNGKey(0))
    assert float(s1.suppressants[0]) == 2.0  # noop refills to capacity (deterministic)
    # a masked (bad) fight is penalised and does nothing
    s2, r2, _ = W.step(s, jnp.array([0, fight(p, 1, 1)]), p, jax.random.PRNGKey(0))
    assert float(r2) == -10.0 / 2 and float(s2.suppressants[1]) == 2.0


def test_static_agents_cannot_move():
    p = dataclasses.replace(DET, mobile=False)
    f, i = one_fire(p)
    s = make_state(p, f, i, [[0, 0], [4, 4]])
    assert not np.asarray(W.action_mask(s, p))[:, 1:5].any()
    s1, _, _ = W.step(s, jnp.array([2, 1]), p, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(s1.agent_pos), [[0, 0], [4, 4]])
    pm = dataclasses.replace(DET, mobile=True)
    s2, _, _ = W.step(s, jnp.array([2, 1]), pm, jax.random.PRNGKey(0))  # S from (0,0), N from (4,4)
    np.testing.assert_array_equal(np.asarray(s2.agent_pos), [[1, 0], [3, 4]])


def test_equipment_degrade_repair_cycle():
    p = dataclasses.replace(DET, degrade=False, repair=False)  # deterministic: degrade every step, repair at 0
    eq = jnp.array([2, 1, 0])
    out = W._equipment(p, eq, jnp.zeros(3), jnp.zeros(3, bool))
    np.testing.assert_array_equal(np.asarray(out), [1, 0, 2])


def test_capacity_switch_on_refill():
    p = dataclasses.replace(DET, possible_capacities=(1.0, 2.0, 3.0), capacity_probabilities=(0.0, 0.0, 1.0))
    supp, cap = W._capacity(p, jnp.array([2.0, 2.0]), jnp.array([2.0, 2.0]), jnp.array([True, False]),
                            jnp.array([0.5, 0.5]), jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(cap), [3.0, 2.0])
    np.testing.assert_allclose(np.asarray(supp), [3.0, 2.0])


def test_visibility_events_match_bruteforce():
    p = W.WildfireParams(n_agents=6, grid_size=9, obs_radius=2)
    rng = np.random.default_rng(0)
    for k in range(20):
        s = W.reset(jax.random.PRNGKey(k), p)
        lit = rng.random((9, 9)) < 0.15
        s = s._replace(fires=jnp.where(lit, 1, -1), intensity=jnp.where(lit, 2, 0),
                       agent_pos=jnp.asarray(rng.integers(0, 9, (6, 2)), jnp.float32))
        mask = jnp.asarray(rng.random((6, 6)) < 0.7) & ~jnp.eye(6, dtype=bool)
        vis = np.asarray(W.visibility(s, p))
        pos = np.asarray(s.agent_pos)
        cells = np.asarray(W.cell_positions(p))
        np.testing.assert_array_equal(vis, np.abs(cells[None] - pos[:, None]).max(-1) <= 2)
        ref = np.asarray(compute_events(jnp.asarray(vis), W.coverage(s, p), mask))
        np.testing.assert_array_equal(np.asarray(W.events(s, p, mask)), ref)


def test_observation_window():
    p = dataclasses.replace(DET, obs_radius=1)
    f, i = one_fire(p, typ=2, inten=2, at=(1, 1))
    s = make_state(p, f, i, [[0, 0], [1, 2]])
    o = np.asarray(W.get_obs(s, p))
    win = o[:, : 9 * W.N_CHANNELS].reshape(2, 3, 3, W.N_CHANNELS)
    # agent 0 at (0,0): fire at offset (+1,+1) -> window cell (2,2); out-of-grid cells have in_bounds 0
    assert win[0, 2, 2, 0] == pytest.approx(2 / 3) and win[0, 2, 2, 1] == pytest.approx(2 / 4)
    assert win[0, 0, :, 5].sum() == 0 and win[0, :, 0, 5].sum() == 0 and win[0, 1, 1, 5] == 1
    assert win[0, 1, 1, 4] == 0  # self not counted as another agent
    assert win[1, 1, 1, 4] == 0 and win[1, 1, 0, 0] == pytest.approx(2 / 3)  # agent 1 sees the fire to its west
    assert o.shape == (2, p.obs_dim)


def test_termination_absorbing_and_reward_once():
    p = dataclasses.replace(DET, termination_reward=5.0, fire_fuel=False)
    f, i = one_fire(p, typ=1, inten=1)
    s = make_state(p, f, i, [[2, 1], [0, 0]])
    s1, r1, _ = W.step(s, jnp.array([fight(p, 0, 1), 0]), p, jax.random.PRNGKey(0))
    assert bool(s1.terminated) and float(r1) == 1.0 + 5.0
    assert int(jnp.max(s1.fires)) == 0  # v0 clears the grid of terminated envs
    s2, r2, _ = W.step(s1, jnp.array([0, 0]), p, jax.random.PRNGKey(1))
    assert float(r2) == 0.0 and bool(s2.terminated)


def test_jit_vmap_and_determinism():
    p = W.WildfireParams()
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    states = jax.vmap(W.reset, in_axes=(0, None))(keys, p)
    acts = jnp.zeros((8, p.n_agents), jnp.int32)
    step = jax.jit(jax.vmap(lambda s, a, k: W.step(s, a, p, k)))
    a1 = step(states, acts, keys)
    a2 = step(states, acts, keys)
    np.testing.assert_array_equal(np.asarray(a1[0].fires), np.asarray(a2[0].fires))
    assert a1[1].shape == (8,)


def test_human_label_and_global_state_shapes():
    p = W.WildfireParams()
    s = W.reset(jax.random.PRNGKey(0), p)
    lab = np.asarray(W.human_label(s, p))
    assert lab.shape == (4,) and lab.min() >= 0 and lab.max() <= 8
    gs = W.global_state(s, p)
    assert gs.agents.shape == (4, W.N_SELF) and gs.landmarks.shape == (64, 7) and gs.time.shape == (4,)


def test_calibrated_task_rewards_information():
    """The default task is calibrated so that information matters: a scripted greedy team that
    sees the whole grid clearly beats the same team with only its local view, which beats doing
    nothing (headroom for learned communication)."""
    p = W.WildfireParams()
    st = {k: rollout_stats(POLICIES[k], p, jax.random.PRNGKey(0), 256) for k in ("noop", "greedy_local", "greedy_global")}
    assert st["greedy_global"]["ret"] > st["greedy_local"]["ret"] + 1.0 > st["noop"]["ret"] + 1.0
    assert st["greedy_global"]["success"] < 0.9  # not at ceiling
