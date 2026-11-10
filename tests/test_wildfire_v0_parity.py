"""Differential test: the JAX transitions in envs/wildfire.py against the original PyTorch
transition modules of free-range-zoo wildfire_v0, fed the SAME states and the SAME uniform
random numbers, for every stochastic / deterministic switch combination.

Needs PyTorch and a checkout of https://github.com/oasys-mas/free-range-zoo (only the
``envs/wildfire/env/v0/transitions/*.py`` files are loaded; the package itself is not
installed). Point ``FREE_RANGE_ZOO_SRC`` at the checkout; skipped if either is missing.
"""
import dataclasses
import importlib.util
import itertools
import os
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs import wildfire as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("FREE_RANGE_ZOO_SRC", ROOT.parent / "ext" / "free-range-zoo"))
TR = SRC / "free_range_zoo" / "envs" / "wildfire" / "env" / "v0" / "transitions"
if not TR.exists():
    pytest.skip(f"free-range-zoo source not found at {SRC}", allow_module_level=True)


def _load_v0():
    # the transition files only use WildfireState as a type hint: stub the package chain
    for name in ["free_range_zoo", "free_range_zoo.envs", "free_range_zoo.envs.wildfire",
                 "free_range_zoo.envs.wildfire.env", "free_range_zoo.envs.wildfire.env.v0",
                 "free_range_zoo.envs.wildfire.env.v0.structures"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    st = types.ModuleType("free_range_zoo.envs.wildfire.env.v0.structures.state")
    st.WildfireState = object
    sys.modules["free_range_zoo.envs.wildfire.env.v0.structures.state"] = st
    mods = {}
    for f in ("fire_increase", "fire_decrease", "fire_spreads", "suppressant_decrease", "suppressant_refill",
              "capacity", "equipment"):
        spec = importlib.util.spec_from_file_location(f"v0_{f}", TR / f"{f}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods[f] = m
    return mods


V0 = _load_v0()
B, H, Wd, N, S = 64, 6, 7, 5, 5
RNG = np.random.default_rng(0)


def random_grid():
    types_ = RNG.integers(0, 4, (B, H, Wd))
    lit = RNG.random((B, H, Wd)) < 0.4
    fires = np.where(lit, types_, -types_)
    inten = np.where(fires > 0, RNG.integers(1, S - 1, (B, H, Wd)), 0)
    burned = (fires < 0) & (RNG.random((B, H, Wd)) < 0.2)
    inten = np.where(burned, S - 1, inten)
    fuel = np.where(fires != 0, RNG.integers(0, 3, (B, H, Wd)), 0)
    attack = RNG.integers(0, 4, (B, H, Wd)).astype(np.float32)
    u = RNG.random((B, H, Wd)).astype(np.float32)
    return fires.astype(np.int32), inten.astype(np.int32), fuel.astype(np.int32), attack, u


def tstate(**kw):
    return types.SimpleNamespace(**{k: torch.tensor(np.array(v)) for k, v in kw.items()})


def P(**kw):
    return dataclasses.replace(W.WildfireParams(num_fire_states=S), **kw)


@pytest.mark.parametrize("stoch_inc,special", list(itertools.product([True, False], [True, False])))
def test_fire_increase(stoch_inc, special):
    p = P(fire_increase=stoch_inc, special_burnout_probability=special, intensity_increase_probability=0.4,
          burnout_probability=0.25)
    for _ in range(5):
        f, i, fu, a, u = random_grid()
        ref = V0["fire_increase"].FireIncreaseTransition((B, H, Wd), S, stoch_inc, 0.4, special, 0.25)
        st, burned = ref(tstate(fires=f, intensity=i, fuel=fu), torch.tensor(a), torch.tensor(u), return_burned_out=True)
        out = W._fire_increase(p, jnp.asarray(f), jnp.asarray(i), jnp.asarray(fu), jnp.asarray(a), jnp.asarray(u))
        for got, exp in zip(out, (st.fires, st.intensity, st.fuel, burned)):
            np.testing.assert_array_equal(np.asarray(got), exp.numpy())


@pytest.mark.parametrize("stoch_dec", [True, False])
def test_fire_decrease(stoch_dec):
    p = P(fire_decrease=stoch_dec, intensity_decrease_probability=0.6, extra_power_decrease_bonus=0.15)
    for _ in range(5):
        f, i, fu, a, u = random_grid()
        ref = V0["fire_decrease"].FireDecreaseTransition((B, H, Wd), stoch_dec, 0.6, 0.15)
        st, put = ref(tstate(fires=f, intensity=i, fuel=fu), torch.tensor(a), torch.tensor(u), return_put_out=True)
        out = W._fire_decrease(p, jnp.asarray(f), jnp.asarray(i), jnp.asarray(fu), jnp.asarray(a), jnp.asarray(u))
        for got, exp in zip(out, (st.fires, st.intensity, st.fuel, put)):
            np.testing.assert_array_equal(np.asarray(got), exp.numpy())


@pytest.mark.parametrize("realistic,fuel_on,rnd", [(False, True, False), (True, True, False), (False, False, True),
                                                   (True, False, True)])
def test_fire_spread(realistic, fuel_on, rnd):
    p = P(realistic_fire_spread=realistic, base_spread_rate=13.5 if realistic else 0.3, max_spread_rate=32.0,
          wind_direction=1.1, fire_fuel=fuel_on, random_fire_ignition=rnd, random_ignition_probability=0.05)
    wn, we, ws, ww = p.spread_weights
    filt = torch.tensor([[0.0, wn, 0.0], [ww, 0.0, we], [0.0, ws, 0.0]], dtype=torch.float32)[None, None]
    ign = torch.full((H, Wd), p.ignition_temp, dtype=torch.int32)
    ref = V0["fire_spreads"].FireSpreadTransition(filt, 0.05 if rnd else 0.0, ign, fuel_on)
    for _ in range(5):
        f, i, fu, _, u = random_grid()
        st = ref(tstate(fires=f, intensity=i, fuel=fu), torch.tensor(u))
        fo, io = jax_vmap_spread(p, f, i, fu, u)
        np.testing.assert_array_equal(fo, st.fires.numpy())
        np.testing.assert_array_equal(io, st.intensity.numpy())


def jax_vmap_spread(p, f, i, fu, u):
    import jax

    fo, io = jax.vmap(lambda a, b, c, d: W._fire_spread(p, a, b, c, d))(*map(jnp.asarray, (f, i, fu, u)))
    return np.asarray(fo), np.asarray(io)


def agents():
    supp = RNG.integers(0, 4, (B, N)).astype(np.float32)
    cap = RNG.integers(1, 4, (B, N)).astype(np.float32)
    eq = RNG.integers(0, 3, (B, N)).astype(np.int64)
    mask = RNG.random((B, N)) < 0.5
    u = RNG.random((5, B, N)).astype(np.float32)
    return supp, cap, eq, mask, u


@pytest.mark.parametrize("stoch", [True, False])
def test_suppressant_decrease(stoch):
    p = P(suppressant_decrease=stoch, suppressant_decrease_probability=0.7)
    supp, cap, eq, users, u = agents()
    ref = V0["suppressant_decrease"].SuppressantDecreaseTransition((B, N), stoch, 0.7)
    st = ref(tstate(suppressants=supp), torch.tensor(users), torch.tensor(u[0]))
    got = W._suppressant_decrease(p, jnp.asarray(supp), jnp.asarray(users), jnp.asarray(u[0]))
    np.testing.assert_allclose(np.asarray(got), st.suppressants.numpy())


@pytest.mark.parametrize("stoch", [True, False])
def test_suppressant_refill_and_capacity(stoch):
    eqs = ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.5, 0.5))
    p = P(suppressant_refill=stoch, suppressant_refill_probability=0.6, equipment_states=eqs,
          tank_switch=stoch, tank_switch_probability=0.5, possible_capacities=(1.0, 2.0, 3.0),
          capacity_probabilities=(0.2, 0.5, 0.3))
    supp, cap, eq, refills, u = agents()
    ref = V0["suppressant_refill"].SuppressantRefillTransition((B, N), stoch, 0.6, torch.tensor(eqs)[:, 0])
    st, inc = ref(tstate(suppressants=supp, capacity=cap, equipment=eq), torch.tensor(refills), torch.tensor(u[2]),
                  return_increased=True)
    s_j, inc_j = W._refill(p, jnp.asarray(supp), jnp.asarray(cap), jnp.asarray(eq), jnp.asarray(refills), jnp.asarray(u[2]))
    np.testing.assert_allclose(np.asarray(s_j), st.suppressants.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(inc_j), inc.numpy())
    ref_c = V0["capacity"].CapacityTransition((B, N), stoch, 0.5, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.2, 0.5, 0.3]))
    st2 = ref_c(tstate(suppressants=st.suppressants.numpy(), capacity=cap), inc, torch.tensor(u[3:5]))
    s2, c2 = W._capacity(p, s_j, jnp.asarray(cap), inc_j, jnp.asarray(u[3]), jnp.asarray(u[4]))
    np.testing.assert_allclose(np.asarray(s2), st2.suppressants.numpy(), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(c2), st2.capacity.numpy(), rtol=1e-6)


@pytest.mark.parametrize("repair,degrade,critical,cond", list(itertools.product([True, False], repeat=4)))
def test_equipment(repair, degrade, critical, cond):
    eqs = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    p = P(equipment_states=eqs, initial_equipment_state=3, repair=repair, degrade=degrade, critical_error=critical,
          repair_probability=0.4, degrade_probability=0.3, critical_error_probability=0.2,
          equipment_condition_on_actions=cond)
    _, _, _, fought, u = agents()
    eq = RNG.integers(0, 4, (B, N)).astype(np.int64)
    ref = V0["equipment"].EquipmentTransition(torch.tensor(eqs), repair, 0.4, degrade, 0.3, critical, 0.2, cond)
    st = ref(tstate(equipment=eq.copy()), torch.tensor(u[1]), torch.tensor(fought))
    got = W._equipment(p, jnp.asarray(eq), jnp.asarray(u[1]), jnp.asarray(fought))
    np.testing.assert_array_equal(np.asarray(got), st.equipment.numpy())
