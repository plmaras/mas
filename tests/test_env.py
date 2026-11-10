import jax
import jax.numpy as jnp
import numpy as np

from envs import spread
from envs.spread import EnvState, SpreadParams

P = SpreadParams(n_agents=3, n_landmarks=3, obs_radius=0.5, cover_threshold=0.1, agent_radius=0.05,
                 collision_penalty=1.0, normalize_reward=False)


def make_state(agents, landmarks, vel=None):
    agents = jnp.asarray(agents, jnp.float32)
    return EnvState(agents, jnp.zeros_like(agents) if vel is None else jnp.asarray(vel, jnp.float32),
                    jnp.asarray(landmarks, jnp.float32), jnp.array(0), jnp.array(False))


def test_visibility_radius():
    s = make_state([[0, 0], [0.9, 0], [-0.9, 0]], [[0.4, 0], [0.6, 0], [0, 0.49]])
    vis = np.asarray(spread.visibility(s, P))
    # agent 0 sees landmarks within 0.5: l0 (0.4) and l2 (0.49) but not l1 (0.6)
    assert vis[0].tolist() == [True, False, True]
    # agent 1 at x=0.9: l0 at distance 0.5 (inclusive), l1 at 0.3
    assert vis[1].tolist() == [True, True, False]
    assert vis[2].tolist() == [False, False, False]


def test_coverage_and_reward():
    s = make_state([[0, 0], [1, 0], [0, 1]], [[0.05, 0], [1.0, 0.5], [0, 1.0]])
    cov = np.asarray(spread.coverage(s, P))
    assert cov.tolist() == [True, False, True]
    r = float(spread.reward_fn(s, P))
    np.testing.assert_allclose(r, -(0.05 + 0.5 + 0.0), atol=1e-6)


def test_collision_penalty():
    s = make_state([[0, 0], [0.05, 0], [1, 1]], [[0, 0], [0, 0], [0, 0]])
    assert int(spread.n_collisions(s, P)) == 1
    s2 = make_state([[0, 0], [0.05, 0], [0.02, 0.02]], [[0, 0], [0, 0], [0, 0]])
    assert int(spread.n_collisions(s2, P)) == 3
    r = float(spread.reward_fn(s, P))
    d = np.array([0.0, 0.0, 0.0])  # all landmarks at origin, agent 0 sits on them
    np.testing.assert_allclose(r, -(d.sum()) - 1.0, atol=1e-6)


def test_reward_normalisation_and_density():
    p4 = SpreadParams(n_agents=4, density=1.0)
    p16 = SpreadParams(n_agents=16, density=1.0)
    assert np.isclose(p4.arena_size**2 / 4, p16.arena_size**2 / 16)
    s = spread.reset(jax.random.PRNGKey(0), p4)
    raw = float(spread.reward_fn(s, SpreadParams(n_agents=4, normalize_reward=False)))
    assert np.isclose(float(spread.reward_fn(s, p4)), raw / 4, atol=1e-6)


def test_obs_masks_invisible_entities():
    s = make_state([[0, 0], [2, 2], [-2, 2]], [[0.3, 0], [3, 3], [-3, -3]])
    p = SpreadParams(n_agents=3, n_landmarks=3, density=0.1, k_obs_landmarks=2, k_obs_agents=2)
    obs = np.asarray(spread.get_obs(s, p))
    assert obs.shape == (3, p.obs_dim)
    lm = obs[0, 5:5 + 6].reshape(2, 3)
    np.testing.assert_allclose(lm[0], [0.3 / 0.5, 0, 1])  # nearest landmark visible
    np.testing.assert_allclose(lm[1], [0, 0, 0])  # second-nearest out of range -> zeros
    ag = obs[0, 11:].reshape(2, 3)
    np.testing.assert_allclose(ag, 0)  # other agents far away


def test_step_dynamics_and_done():
    p = SpreadParams(n_agents=2, n_landmarks=2, max_steps=2)
    s = make_state([[0, 0], [0.5, 0.5]], [[1, 1], [-1, -1]])
    s1, r, d = spread.step(s, jnp.array([1, 0]), p)  # agent 0 accelerates +x
    assert float(s1.agent_pos[0, 0]) > 0 and float(s1.agent_pos[1, 0]) == 0.5
    assert not bool(d)
    s2, _, d2 = spread.step(s1, jnp.array([0, 0]), p)
    assert bool(d2)


def test_human_label():
    s = make_state([[0, 0], [5, 5], [-5, -5]], [[0.3, 0.0], [0.0, 0.3], [9, 9]])
    lab = np.asarray(spread.human_label(s, SpreadParams(n_agents=3, n_landmarks=3, density=0.05)))
    assert lab[1] == 8 and lab[2] == 8  # sees nothing -> "none"
    assert 0 <= lab[0] < 8
