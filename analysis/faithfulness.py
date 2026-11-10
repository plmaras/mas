"""Message-use (faithfulness) tests for E6.

Open-loop (teacher-forced) tests replay the logged receiver hidden states and recompute the
receiver policy with intervened messages / gates at a single step, then measure
KL(pi_clean || pi_intervened) on the receiver's next action:
  F1  zero all messages
  F2  ablate active links at event times vs. the same number of active links at random times
  F3  swap messages with those of another sender (same step) or another episode
  F4  targeted counterfactual: replace sender i's message with a donor message whose sender
      saw an uncovered landmark in a different direction, and test whether receiver j's
      expected movement shifts toward the new target.

F4 pre-registered prediction (defined before running): with p_i, p_j sender/receiver
positions, d_orig the offset from sender i to its nearest visible uncovered landmark that j
cannot see, and d_new the analogous offset of the donor sender,
    u = unit(p_i + d_new - p_j) - unit(p_i + d_orig - p_j)
    dv = sum_a (pi'(a) - pi(a)) * dir(a)       (change in receiver's expected action direction)
    match  <=>  dv . u > 0          (chance level 0.5)
Closed-loop reward drops are measured with ``eval.evaluate(..., interv=Interv(...))``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from core.setup import make_setup
from models.actor_critic import CommActor


def make_receiver_fn(cfg, params, chunk: int = 4096):
    """Returns ``f(hidden[B,N,H], messages[B,N,M], g[B,N,N], mask[B,N,N], amask[B,N,A]) ->
    logits[B,N,A]`` (receiver policy logits, invalid actions masked exactly as in the rollout)."""
    from core.rollout import mask_logits

    setup = make_setup(cfg)
    p_act = params["actor"]

    @jax.jit
    def _f(h, m, g, mk, am):
        lg = jax.vmap(lambda a, b, c, d: setup.actor.apply(p_act, a, b, c, d, method=CommActor.receiver)[0])(h, m, g, mk)
        return mask_logits(lg, am)

    def f(h, m, g, mk, am=None):
        if am is None:
            am = np.ones(h.shape[:2] + (setup.actor.n_actions,), bool)
        outs = [np.asarray(_f(h[s:s + chunk], m[s:s + chunk], g[s:s + chunk], mk[s:s + chunk], am[s:s + chunk]))
                for s in range(0, h.shape[0], chunk)]
        return np.concatenate(outs) if outs else np.zeros((0,))

    f.action_dirs = np.asarray(setup.envm.action_dirs(setup.env), np.float32)
    return f


def _flat(logs, *keys):
    out = []
    for k in keys:
        if k == "action_mask" and k not in logs:  # logs from envs without masks: all allowed
            v = np.ones(logs["action_logits"].shape, bool)
        else:
            v = logs[k]
        out.append(v.reshape((-1,) + v.shape[2:]))
    return out


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl_categorical(logits_p, logits_q):
    lp, lq = log_softmax(logits_p), log_softmax(logits_q)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def _kl_summary(kl, affected, n_ablated=None):
    out = dict(kl_mean_all=float(kl.mean()),
               kl_mean_affected=float(kl[affected].mean()) if affected.any() else float("nan"),
               frac_affected=float(affected.mean()))
    if n_ablated is not None:
        out["kl_per_ablated_link"] = float(kl.sum() / max(n_ablated, 1))
        out["n_ablated"] = int(n_ablated)
    return out


def consistency_check(f, logs, n: int = 256) -> float:
    """Max abs difference between recomputed and logged logits (should be ~0)."""
    h, m, g, mk, lg, am = _flat(logs, "hidden", "messages", "g", "mask", "action_logits", "action_mask")
    lg_new = f(h[:n], m[:n], g[:n], mk[:n], am[:n])
    return float(np.abs(np.where(am[:n], lg_new - lg[:n], 0.0)).max())


def f1_zero_messages(f, logs):
    h, m, g, mk, lg, am = _flat(logs, "hidden", "messages", "g", "mask", "action_logits", "action_mask")
    kl = kl_categorical(lg, f(h, np.zeros_like(m), g, mk, am))
    affected = ((g > 0) & mk).any(1)  # receiver has >= 1 active incoming link
    return _kl_summary(kl, affected)


def f2_ablation(f, logs, seed: int = 0):
    """Event-time ablation vs. random ablation of the same number of active links per episode."""
    rng = np.random.default_rng(seed)
    g, E, mk = logs["g"], logs["E"], logs["mask"]
    active = (g > 0) & mk
    ab_event = active & E
    ab_rand = np.zeros_like(ab_event)
    for e in range(g.shape[0]):
        idx = np.flatnonzero(active[e])
        k = int(ab_event[e].sum())
        if k and idx.size:
            ab_rand[e].flat[rng.choice(idx, size=min(k, idx.size), replace=False)] = True
    h, m, gf, mkf, lg, am = _flat(logs, "hidden", "messages", "g", "mask", "action_logits", "action_mask")
    res = {}
    for name, ab in (("event", ab_event), ("random", ab_rand)):
        abf = ab.reshape(gf.shape)
        kl = kl_categorical(lg, f(h, m, gf * (~abf), mkf, am))
        res[name] = _kl_summary(kl, abf.any(1), abf.sum())
    res["ratio_event_over_random"] = res["event"]["kl_per_ablated_link"] / max(res["random"]["kl_per_ablated_link"], 1e-12)
    return res


def f3_swap(f, logs, seed: int = 0):
    rng = np.random.default_rng(seed)
    h, m, g, mk, lg, am = _flat(logs, "hidden", "messages", "g", "mask", "action_logits", "action_mask")
    affected = ((g > 0) & mk).any(1)
    n = m.shape[1]
    perm = np.stack([np.roll(np.arange(n), rng.integers(1, n)) for _ in range(m.shape[0])])  # derangement
    m_sender = np.take_along_axis(m, perm[..., None], axis=1)
    m_episode = np.roll(logs["messages"], 1, axis=0).reshape(m.shape)
    return dict(other_sender=_kl_summary(kl_categorical(lg, f(h, m_sender, g, mk, am)), affected),
                other_episode=_kl_summary(kl_categorical(lg, f(h, m_episode, g, mk, am)), affected))


def _private_target(pos, lpos, vis, cov, i, j):
    """Offset from sender i to its nearest visible uncovered landmark that j cannot see."""
    cand = vis[i] & ~cov & ~vis[j]
    if not cand.any():
        return None
    d = np.linalg.norm(lpos - pos[i], axis=-1)
    d[~cand] = np.inf
    return lpos[np.argmin(d)] - pos[i]


def f4_counterfactual(f, logs, n_cases: int = 3000, min_angle_deg: float = 90.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    g, E, mk = logs["g"], logs["E"], logs["mask"]
    cases = np.argwhere((g > 0) & mk & E)  # (e, t, i, j)
    if len(cases) == 0:
        return dict(n=0, match_rate=float("nan"))
    cases = cases[rng.permutation(len(cases))[: n_cases * 3]]
    pos, lpos, vis, cov = logs["positions"], logs["landmark_pos"], logs["vis"], logs["covered"]
    donors = np.argwhere((vis & ~cov[..., None, :]).any(-1))  # (e, t, i) senders seeing an uncovered landmark
    amask = logs["action_mask"] if "action_mask" in logs else np.ones(logs["action_logits"].shape, bool)
    H, M, G, MK, U, J, LG, AM = [], [], [], [], [], [], [], []
    cos_thr = np.cos(np.deg2rad(min_angle_deg))
    for (e, t, i, j) in cases:
        d_orig = _private_target(pos[e, t], lpos[e, t], vis[e, t], cov[e, t], i, j)
        if d_orig is None:
            continue
        for _ in range(10):
            de, dt, di = donors[rng.integers(len(donors))]
            cand = vis[de, dt, di] & ~cov[de, dt]
            dd = np.linalg.norm(lpos[de, dt] - pos[de, dt, di], axis=-1)
            dd[~cand] = np.inf
            d_new = lpos[de, dt, np.argmin(dd)] - pos[de, dt, di]
            c = d_new @ d_orig / (np.linalg.norm(d_new) * np.linalg.norm(d_orig) + 1e-9)
            if c <= cos_thr:
                break
        else:
            continue
        unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
        u = unit(pos[e, t, i] + d_new - pos[e, t, j]) - unit(pos[e, t, i] + d_orig - pos[e, t, j])
        m = logs["messages"][e, t].copy()
        m[i] = logs["messages"][de, dt, di]
        H.append(logs["hidden"][e, t]); M.append(m); G.append(g[e, t]); MK.append(mk[e, t])
        U.append(u); J.append(j); LG.append(logs["action_logits"][e, t, j]); AM.append(amask[e, t])
        if len(H) >= n_cases:
            break
    if not H:
        return dict(n=0, match_rate=float("nan"))
    logits_new = f(np.stack(H), np.stack(M), np.stack(G), np.stack(MK), np.stack(AM))[np.arange(len(J)), np.array(J)]
    dirs = f.action_dirs
    p_old = np.exp(log_softmax(np.stack(LG)))
    p_new = np.exp(log_softmax(logits_new))
    dv = (p_new - p_old) @ dirs
    U = np.stack(U)
    proj = (dv * U).sum(-1)
    moved = np.linalg.norm(dv, axis=-1) > 1e-6
    cos = proj / (np.linalg.norm(dv, axis=-1) * np.linalg.norm(U, axis=-1) + 1e-12)
    return dict(n=int(len(J)), n_moved=int(moved.sum()),
                match_rate=float((proj[moved] > 0).mean()) if moved.any() else float("nan"),
                mean_cosine=float(cos[moved].mean()) if moved.any() else float("nan"),
                mean_dv_norm=float(np.linalg.norm(dv, axis=-1).mean()),
                kl_mean=float(kl_categorical(np.stack(LG), logits_new).mean()))


def faithfulness_scores(cfg, params, logs, seed: int = 0) -> dict:
    f = make_receiver_fn(cfg, params)
    return dict(consistency_max_abs=consistency_check(f, logs), F1=f1_zero_messages(f, logs),
                F2=f2_ablation(f, logs, seed), F3=f3_swap(f, logs, seed), F4=f4_counterfactual(f, logs, seed=seed))
