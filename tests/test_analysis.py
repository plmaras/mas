import numpy as np

from analysis import metrics
from analysis.events import auroc_from_scores, counts, event_stats, stats_from_counts
from analysis.null import circular_shift_null, circular_shift_valid, compare_to_null
from analysis.stats import bootstrap_ci, iqm


def _markov(rng, shape, p_on, stay):
    """Persistent binary process along axis 1: keeps its value with prob ``stay``."""
    x = np.zeros(shape, bool)
    x[:, 0] = rng.random(shape[:1] + shape[2:]) < p_on
    for t in range(1, shape[1]):
        resample = rng.random(shape[:1] + shape[2:]) > stay
        x[:, t] = np.where(resample, rng.random(shape[:1] + shape[2:]) < p_on, x[:, t - 1])
    return x


def test_link_metrics_toy():
    # one episode, 6 steps, 2 agents -> link (0,1) only
    g = np.zeros((1, 6, 2, 2))
    g[0, :, 0, 1] = [1, 1, 0, 1, 1, 1]
    mask = np.zeros((1, 6, 2, 2), bool)
    mask[0, :, 0, 1] = True
    assert np.isclose(metrics.active_link_ratio(g, mask), 5 / 6)
    assert np.isclose(metrics.gate_switch_rate(g, mask), 2 / 5)
    assert np.isclose(metrics.mean_run_length(g, mask), 5 / 2)
    assert metrics.mask_switch_rate(mask) == 0
    mask[0, 3, 0, 1] = False
    assert np.isclose(metrics.mask_switch_rate(mask), 2 / 5)


def test_contingency_stats():
    n11, n10, n01, n00 = 30, 10, 20, 40
    s = stats_from_counts(n11, n10, n01, n00)
    expected_phi = (30 * 40 - 10 * 20) / np.sqrt(40 * 60 * 50 * 50)
    assert np.isclose(s["phi"], expected_phi)
    assert np.isclose(s["lift"], (30 / 50) / (10 / 50))
    assert np.isclose(s["auroc"], 0.5 * (1 + 30 / 50 - 10 / 50))
    # binary AUROC matches rank-based AUROC
    g = np.r_[np.ones(30), np.zeros(20), np.ones(10), np.zeros(40)]
    e = np.r_[np.ones(50), np.zeros(50)].astype(bool)
    assert np.isclose(auroc_from_scores(g, e), s["auroc"])
    c = counts(g.astype(bool), e, np.ones(100, bool))
    assert tuple(int(x) for x in c) == (30, 10, 20, 40)


def test_circular_shift_preserves_density_and_validity():
    rng = np.random.default_rng(0)
    x = rng.random((50, 30)) < 0.4
    valid = rng.random((50, 30)) < 0.7
    offs = rng.integers(0, 30, 50)
    y = np.asarray(circular_shift_valid(x, valid, offs))
    np.testing.assert_array_equal((x & valid).sum(1), (y & valid).sum(1))
    np.testing.assert_array_equal(y[~valid], x[~valid])
    # zero offset is the identity
    np.testing.assert_array_equal(np.asarray(circular_shift_valid(x, valid, np.zeros(50, int))), x)


def _synthetic(seed, ep=40, T=50, N=4):
    rng = np.random.default_rng(seed)
    mask = ~np.eye(N, dtype=bool)[None, None].repeat(ep, 0).repeat(T, 1)
    E = _markov(rng, (ep, T, N, N), 0.3, 0.95) & mask
    return rng, mask, E


def test_synthetic_gates_from_events_are_significant():
    rng, mask, E = _synthetic(0)
    noise = rng.random(E.shape) < 0.1
    g = (E ^ noise) & mask  # gates driven by events
    obs = event_stats(g, E, mask)
    assert obs["phi"] > 0.5 and obs["lift"] > 3
    null = circular_shift_null(g, E, mask, n_shifts=300, seed=1)
    c = compare_to_null(obs["phi"], null["null_pooled"]["phi"])
    assert c["p_value"] < 0.01 and c["percentile"] > 0.99


def test_persistent_random_gate_not_significant_under_circular_null():
    """A persistent gate independent of the (persistent) event must not look significant
    under circular shifts, whereas plain timestep shuffling over-rejects."""
    n_rep, sig_circ, sig_shuffle = 12, 0, 0
    for r in range(n_rep):
        rng, mask, E = _synthetic(100 + r, ep=10)
        g = _markov(rng, E.shape, 0.4, 0.97) & mask
        phi = event_stats(g, E, mask)["phi"]
        null = circular_shift_null(g, E, mask, n_shifts=200, seed=r)["null_pooled"]["phi"]
        sig_circ += compare_to_null(phi, null)["p_value"] < 0.05
        # naive null: shuffle timesteps independently per sequence
        shuf = []
        for s in range(200):
            perm = np.argsort(rng.random(g.shape), axis=1)
            gs = np.take_along_axis(g, perm, axis=1)
            shuf.append(event_stats(gs, E, mask)["phi"])
        sig_shuffle += compare_to_null(phi, np.array(shuf))["p_value"] < 0.05
    assert sig_circ <= 3, sig_circ  # ~5% nominal rate (allow small-sample slack)
    assert sig_shuffle > sig_circ


def test_results_invariant_to_agent_relabeling():
    rng, mask, E = _synthetic(3)
    g = (E ^ (rng.random(E.shape) < 0.2)) & mask
    perm = rng.permutation(E.shape[-1])
    relabel = lambda a: a[..., perm, :][..., :, perm]
    a = event_stats(g, E, mask)
    b = event_stats(relabel(g), relabel(E), relabel(mask))
    for k in ("phi", "lift", "auroc"):
        assert np.isclose(a[k], b[k])
    na = circular_shift_null(g, E, mask, n_shifts=100, seed=0)["null_pooled"]["phi"]
    assert np.isfinite(na).all()


def test_stratified_stats_and_bins():
    rng, mask, E = _synthetic(4)
    g = E & mask
    strata = rng.integers(0, 3, E.shape)
    s = event_stats(g, E, mask, strata, 3)
    assert len(s["per_bin"]) == 3 and np.isclose(s["phi_stratified"], 1.0)
    null = circular_shift_null(g, E, mask, strata, 3, n_shifts=50)
    assert null["null_per_bin"]["phi"].shape == (50, 3)


def test_bootstrap_and_iqm():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    m, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo <= m <= hi
    assert iqm(np.arange(10.0)) == 4.5


def test_episode_shuffle_null_removes_shared_time_trend():
    """Gate and event both depend only on time (on early), otherwise independent: the circular
    shift reports a spurious association, the episode-shuffle null and time strata do not."""
    from analysis.events import time_strata
    from analysis.null import episode_shuffle_null

    rng = np.random.default_rng(0)
    ep, T, N = 200, 40, 3
    p = np.linspace(0.9, 0.05, T)[None, :, None, None]
    g = rng.random((ep, T, N, N)) < p
    E = rng.random((ep, T, N, N)) < p
    mask = np.ones((ep, T, N, N), bool)
    obs = event_stats(g, E, mask, time_strata(g.shape), T)
    assert obs["phi"] > 0.2  # strong pooled association created by time alone
    assert abs(obs["phi_stratified"]) < 0.03  # gone within timesteps
    circ = compare_to_null(obs["phi"], circular_shift_null(g, E, mask, n_shifts=200)["null_pooled"]["phi"])
    shuf = compare_to_null(obs["phi"], episode_shuffle_null(g, E, mask, n_perm=200)["null_pooled"]["phi"])
    assert circ["p_value"] < 0.01
    assert shuf["p_value"] > 0.05 and abs(shuf["excess"]) < 0.02


def test_episode_shuffle_null_detects_real_association():
    from analysis.null import episode_shuffle_null

    rng = np.random.default_rng(1)
    ep, T, N = 100, 30, 3
    E = rng.random((ep, T, N, N)) < 0.3
    g = np.where(rng.random(E.shape) < 0.7, E, rng.random(E.shape) < 0.3)  # gate follows the event
    mask = np.ones(E.shape, bool)
    obs = event_stats(g, E, mask)
    res = compare_to_null(obs["phi"], episode_shuffle_null(g, E, mask, n_perm=200)["null_pooled"]["phi"])
    assert res["excess"] > 0.4 and res["p_value"] < 0.01


def test_mantel_haenszel_time_confound_and_real_effect():
    from analysis.events import mantel_haenszel, stratum_counts, time_strata

    rng = np.random.default_rng(2)
    ep, T, N = 200, 40, 3
    p = np.linspace(0.9, 0.05, T)[None, :, None, None]
    mask = np.ones((ep, T, N, N), bool)
    st = time_strata(mask.shape)
    g = rng.random(mask.shape) < p
    E = rng.random(mask.shape) < p
    pooled = mantel_haenszel(stratum_counts(g, E, mask, np.zeros_like(st), 1))
    strat = mantel_haenszel(stratum_counts(g, E, mask, st, T))
    assert pooled["log_or"] > 1.0  # time alone -> large pooled odds ratio
    assert abs(strat["log_or"]) < 3 * strat["se"]  # no association within timesteps
    g2 = np.where(rng.random(E.shape) < 0.5, E, g)  # gate partly follows the event
    assert mantel_haenszel(stratum_counts(g2, E, mask, st, T))["log_or"] > 1.0
    # textbook check: one stratum equals the plain odds ratio
    c = np.array([[30.0], [10.0], [20.0], [40.0]])
    assert np.isclose(mantel_haenszel(c)["odds_ratio"], 30 * 40 / (10 * 20))
