import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.comm import compute_events, knn_mask, markov_rates
from core.mappo import init_train_state
from core.rollout import NO_INTERV, build_comm_in, reset_batch
from core.setup import make_setup
from models import gates as G


def test_knn_mask_correctness():
    pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    m = np.asarray(knn_mask(pos, k=2))
    assert m.shape == (4, 4)
    assert not m.diagonal().any()
    # each receiver (column) has exactly k senders
    assert (m.sum(0) == 2).all()
    # receiver 0's nearest senders are 1 and 2; receiver 3's are 2 and 1
    assert m[:, 0].tolist() == [False, True, True, False]
    assert m[:, 3].tolist() == [False, True, True, False]
    # k larger than N-1 -> fully connected
    assert np.asarray(knn_mask(pos, k=10)).sum() == 12
    # comm radius
    assert np.asarray(knn_mask(pos, k=3, comm_radius=1.5)).sum() == 2


def test_event_definition():
    vis = np.array([[True, False], [False, False], [True, True]])  # 3 agents, 2 landmarks
    covered = np.array([False, False])
    mask = ~np.eye(3, dtype=bool)
    E = compute_events(vis, covered, mask)
    # sender 0 sees l0, receiver 1 does not -> event; receiver 2 sees l0 -> no event
    assert E[0, 1] and not E[0, 2]
    assert not E[1].any()  # sender 1 sees nothing
    assert E[2, 0] and E[2, 1]  # sender 2 sees l1 which 0 cannot see
    E_cov = compute_events(vis, np.array([True, True]), mask)
    assert not E_cov.any()
    assert not compute_events(vis, covered, np.zeros((3, 3), bool)).any()
    # jax and numpy agree
    np.testing.assert_array_equal(np.asarray(compute_events(jnp.asarray(vis), jnp.asarray(covered), jnp.asarray(mask))), E)


def test_hard_concrete_prob_on_matches_samples():
    alpha = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    u = jax.random.uniform(jax.random.PRNGKey(0), (200000, 5), minval=1e-6, maxval=1 - 1e-6)
    g = G.hc_sample(alpha, u)
    emp = np.asarray((g > 0).mean(0))
    np.testing.assert_allclose(emp, np.asarray(G.hc_prob_on(alpha)), atol=5e-3)
    assert float(g.min()) >= 0 and float(g.max()) <= 1
    det = np.asarray(G.hc_deterministic(alpha))
    assert det[0] == 0 and det[-1] > 0.9


def test_penalty_values_toy():
    p = jnp.array([[0.2, 0.8], [0.6, 0.4], [1.0, 0.0]])  # [T=3, links=2]
    perm = jnp.array([[1, 1], [1, 0], [1, 1]], dtype=bool)
    np.testing.assert_allclose(float(G.comm_cost(p, perm)), (0.2 + 0.8 + 0.6 + 1.0 + 0.0) / 5, atol=1e-6)
    # pairs permitted at both steps: (t1,l0),(t2,l0) only
    f1 = 0.6 * 0.8 + 0.2 * 0.4
    f2 = 1.0 * 0.4 + 0.6 * 0.0
    np.testing.assert_allclose(float(G.switching_cost(p, perm)), (f1 + f2) / 2, atol=1e-6)
    # all-off and all-on gates have zero switching cost
    assert float(G.switching_cost(jnp.zeros((4, 3)), jnp.ones((4, 3), bool))) == 0
    assert float(G.switching_cost(jnp.ones((4, 3)), jnp.ones((4, 3), bool))) == 0


def test_lambda_schedule():
    f = lambda x: float(G.lambda_schedule(x, 1.0, 0.2, 0.3))
    assert f(0.0) == 0 and f(0.19) == 0
    assert np.isclose(f(0.35), 0.5) and f(0.5) == 1 and f(0.9) == 1


def test_markov_rates():
    a, b = markov_rates(0.3, 0.1)
    assert np.isclose(a / (a + b), 0.3) and np.isclose(2 * a * b / (a + b), 0.1)


@pytest.mark.parametrize("env", ["wildfire", "spread"])
@pytest.mark.parametrize("comm", ["gated", "nocomm", "fullcomm", "handrule", "gated_reinforce", "ic3net", "language", "aecomm"])
def test_actor_shapes_and_no_nans(make_cfg, comm, env):
    cfg = make_cfg(f"env={env}", f"comm={comm}", "env.n_agents=5", "comm.k_neighbors=2")
    setup = make_setup(cfg)
    ts = init_train_state(setup, cfg, jax.random.PRNGKey(0))
    carry = reset_batch(setup, jax.random.PRNGKey(1), 3)
    c, _, _ = build_comm_in(setup, carry, jax.random.PRNGKey(2), NO_INTERV)
    obs = jax.vmap(setup.envm.get_obs, in_axes=(0, None))(carry.env_state, setup.env)
    f = jax.vmap(lambda h, o, cc: setup.actor.apply(ts.params["actor"], h, o, cc, False, False))
    h, logits, aux = f(carry.h, obs, c)
    N, M = 5, setup.msg_size
    assert logits.shape == (3, N, setup.actor.n_actions) and h.shape == (3, N, cfg.algo.hidden_dim)
    assert aux["g"].shape == (3, N, N) and aux["alpha"].shape == (3, N, N)
    assert aux["msg"].shape == (3, N, M) and aux["attn"].shape == (3, N, N)
    assert np.isfinite(np.asarray(logits)).all()
    g = np.asarray(aux["g"])
    assert (g[~np.asarray(c.mask)] == 0).all()  # links outside the KNN mask are off
    if comm == "nocomm":
        np.testing.assert_allclose(np.asarray(aux["msg"]), 0)


def test_all_gates_closed_no_nan(make_cfg):
    cfg = make_cfg("comm=gated", "env.n_agents=4")
    setup = make_setup(cfg)
    ts = init_train_state(setup, cfg, jax.random.PRNGKey(0))
    h = jnp.zeros((4, cfg.algo.hidden_dim))
    m = jnp.ones((4, setup.msg_size))
    from models.actor_critic import CommActor

    logits, attn = setup.actor.apply(ts.params["actor"], h, m, jnp.zeros((4, 4)), jnp.ones((4, 4)),
                                     method=CommActor.receiver)
    assert np.isfinite(np.asarray(logits)).all()
    np.testing.assert_allclose(np.asarray(attn), 0)


def test_parameter_parity_nocomm_fullcomm(make_cfg):
    sizes = []
    for comm in ("nocomm", "fullcomm", "gated", "handrule"):
        cfg = make_cfg(f"comm={comm}")
        setup = make_setup(cfg)
        ts = init_train_state(setup, cfg, jax.random.PRNGKey(0))
        sizes.append(sum(x.size for x in jax.tree_util.tree_leaves(ts.params)))
    assert len(set(sizes)) == 1


def test_dual_step_and_target_schedule():
    from models.gates import dual_step, target_density_schedule

    assert target_density_schedule(0.1, 0.25, 0.2, 0.3) == 1.0
    assert abs(target_density_schedule(0.35, 0.25, 0.2, 0.3) - 0.625) < 1e-9
    assert target_density_schedule(0.9, 0.25, 0.2, 0.3) == 0.25
    lam = dual_step(0.0, 0.6, 0.25, 0.01, 0.1)
    assert lam > 0  # too dense -> raise the cost
    assert dual_step(0.0, 0.1, 0.25, 0.01, 0.1) < 0  # too sparse -> equality constraint lets it go negative
    assert dual_step(0.099, 1.0, 0.0, 1.0, 0.1) == 0.1  # clipped


def test_pi_step_anti_windup():
    from models.gates import dual_step, pi_step

    i, lam = pi_step(0.0, 0.6, 0.25, 0.01, 0.0, 0.1)
    assert np.isclose(lam, dual_step(0.0, 0.6, 0.25, 0.01, 0.1))  # kp = 0 -> plain dual ascent
    i, lam = pi_step(0.0, 0.6, 0.25, 0.01, 0.1, 0.1)
    assert np.isclose(lam, 0.0035 + 0.035)  # proportional term acts immediately
    i = 0.0
    for _ in range(1000):
        i, lam = pi_step(i, 1.0, 0.0, 0.01, 0.0, 0.1)
    assert i == 0.1  # integral clamped: no wind-up beyond the clip
    i, lam = pi_step(i, 0.0, 0.25, 0.01, 0.1, 0.1)
    assert lam < i  # undershoot pulls lambda down at once
