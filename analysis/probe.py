"""Message-content probes (E7) and readability (E8).

Probes are trained on held-out-by-episode splits:
  * regression: message m_i -> offset (in units of R_obs) from sender i to its nearest visible
    uncovered landmark that receiver j cannot see (one sample per permitted link where such a
    landmark exists, i.e. information the receiver lacks);
  * classification: m_i -> "sender sees an uncovered landmark";
  * readability: m_i -> human-readable label (compass octant of the nearest visible
    uncovered landmark, or "none").
Each probe is fit as a linear model and as a small MLP, and compared with a shuffled-label
baseline (same model fit on permuted training targets, tested on true targets).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _split_by_episode(ep_ids, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    test = set(rng.choice(eps, size=max(1, int(round(len(eps) * test_frac))), replace=False).tolist())
    is_test = np.array([e in test for e in ep_ids])
    return ~is_test, is_test


def regression_dataset(logs, obs_radius: float, max_samples: int = 30000, seed: int = 0):
    vis, cov = logs["vis"], logs["covered"]
    pos, lpos, msgs = logs["positions"], logs["landmark_pos"], logs["messages"]
    # E[e,t,i,j] is exactly "permitted link and i sees an uncovered landmark j cannot see"
    idx = np.argwhere(logs["E"])
    rng = np.random.default_rng(seed)
    idx = idx[rng.permutation(len(idx))[:max_samples]]
    X, Y, ep = [], [], []
    for (e, t, i, j) in idx:
        c = vis[e, t, i] & ~cov[e, t] & ~vis[e, t, j]
        d = np.linalg.norm(lpos[e, t] - pos[e, t, i], axis=-1)
        d[~c] = np.inf
        Y.append((lpos[e, t, np.argmin(d)] - pos[e, t, i]) / obs_radius)
        X.append(msgs[e, t, i])
        ep.append(e)
    if not X:
        return np.zeros((0, msgs.shape[-1])), np.zeros((0, 2)), np.zeros(0, int)
    return np.stack(X), np.stack(Y), np.array(ep)


def sender_dataset(logs, label: str, max_samples: int = 30000, seed: int = 0):
    """One sample per (episode, t, sender)."""
    msgs = logs["messages"]
    e_ids = np.broadcast_to(np.arange(msgs.shape[0])[:, None, None], msgs.shape[:3])
    if label == "sees_uncovered":
        y = (logs["vis"] & ~logs["covered"][..., None, :]).any(-1).astype(int)
    elif label == "human_label":
        y = logs["human_label"].astype(int)
    else:
        raise ValueError(label)
    X, y, ep = msgs.reshape(-1, msgs.shape[-1]), y.reshape(-1), e_ids.reshape(-1)
    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(y))[:max_samples]
    return X[sel], y[sel], ep[sel]


def _fit(X, Y, task: str, n_classes: int, hidden: int, steps: int, seed: int, l2: float = 1e-4):
    key = jax.random.PRNGKey(seed)
    d_in = X.shape[1]
    d_out = n_classes if task == "clf" else Y.shape[1]
    k1, k2 = jax.random.split(key)
    if hidden:
        params = dict(W1=jax.random.normal(k1, (d_in, hidden)) / np.sqrt(max(d_in, 1)), b1=jnp.zeros(hidden),
                      W2=jax.random.normal(k2, (hidden, d_out)) / np.sqrt(hidden), b2=jnp.zeros(d_out))
    else:
        params = dict(W2=jnp.zeros((d_in, d_out)), b2=jnp.zeros(d_out))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = jnp.asarray((X - mu) / sd)
    Yj = jnp.asarray(Y)

    def fwd(p, x):
        if "W1" in p:
            x = jax.nn.relu(x @ p["W1"] + p["b1"])
        return x @ p["W2"] + p["b2"]

    def loss(p):
        out = fwd(p, Xn)
        if task == "clf":
            l = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(out), Yj[:, None], 1))
        else:
            l = jnp.mean((out - Yj) ** 2)
        return l + l2 * sum(jnp.sum(v**2) for k, v in p.items() if k.startswith("W"))

    tx = optax.adam(1e-2)
    st = tx.init(params)

    @jax.jit
    def step(p, s):
        g = jax.grad(loss)(p)
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s

    for _ in range(steps):
        params, st = step(params, st)
    return lambda Xt: np.asarray(fwd(params, jnp.asarray((Xt - mu) / sd)))


def _score(task, pred, Y):
    if task == "clf":
        return float((pred.argmax(-1) == Y).mean())
    ss_res = ((pred - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def run_probe(X, Y, ep, task: str, n_classes: int = 0, test_frac: float = 0.2, seed: int = 0,
              mlp_hidden: int = 64, steps: int = 500) -> dict:
    """R^2 (regression) or accuracy (classification) on held-out episodes, for a linear and
    an MLP probe, each with a shuffled-label baseline."""
    if len(Y) < 20 or len(np.unique(ep)) < 2:
        return dict(n=int(len(Y)), note="too few samples")
    tr, te = _split_by_episode(ep, test_frac, seed)
    rng = np.random.default_rng(seed)
    out = dict(n_train=int(tr.sum()), n_test=int(te.sum()))
    if task == "clf":
        majority = np.bincount(Y[tr], minlength=n_classes).argmax()
        out["majority_baseline"] = float((Y[te] == majority).mean())
    for name, hid in (("linear", 0), ("mlp", mlp_hidden)):
        pred = _fit(X[tr], Y[tr], task, n_classes, hid, steps, seed)(X[te])
        Ysh = Y[tr][rng.permutation(tr.sum())]
        pred_sh = _fit(X[tr], Ysh, task, n_classes, hid, steps, seed)(X[te])
        out[name] = _score(task, pred, Y[te])
        out[f"{name}_shuffled"] = _score(task, pred_sh, Y[te])
    return out


def probe_all(logs, obs_radius: float, seed: int = 0, steps: int = 500) -> dict:
    Xr, Yr, er = regression_dataset(logs, obs_radius, seed=seed)
    Xs, ys, es = sender_dataset(logs, "sees_uncovered", seed=seed)
    Xh, yh, eh = sender_dataset(logs, "human_label", seed=seed)
    return dict(
        private_landmark_offset=run_probe(Xr, Yr, er, "reg", seed=seed, steps=steps),
        sees_uncovered=run_probe(Xs, ys, es, "clf", 2, seed=seed, steps=steps),
        human_label=run_probe(Xh, yh, eh, "clf", 9, seed=seed, steps=steps),
    )
