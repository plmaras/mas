"""Train one MAPPO run.

CLI (Hydra):   python train.py comm=gated penalties.lambda_c=0.05 seed=1
Programmatic:  from train import run_training; run_training(cfg)
Batched:       from train import run_training_batch; run_training_batch([cfg_a, cfg_b, ...])
               (runs that share every static setting are vmapped into one jitted update)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from core.mappo import init_train_state, make_update
from core.setup import make_setup
from core.utils import git_hash, peak_memory_mb, save_checkpoint, seed_everything
from models.gates import lambda_schedule, pi_step, target_density_schedule


def _maybe_wandb(cfg: DictConfig):
    if not cfg.logging.wandb:
        return None
    try:
        import wandb

        run = wandb.init(
            project=cfg.logging.wandb_project,
            mode=cfg.logging.wandb_mode,
            name=f"{cfg.exp_name}/{cfg.run_name}/seed_{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(Path(cfg.out_dir)),
        )
        run.config.update({"git_hash": git_hash()}, allow_val_change=True)
        return run
    except Exception as e:  # never let logging kill a training run
        print(f"[train] wandb disabled: {e}")
        return None


def _lambdas(P: DictConfig, progress: float, lam_dual: float):
    """``(lambda_c, lambda_s, density target)`` for one run at ``progress``. Without
    ``target_density`` both lambdas follow the warm-up/ramp schedule and the target is None;
    with it, lambda_c is the dual variable ``lam_dual``."""
    ls_w = P.warmup_frac if P.get("ls_warmup_frac") is None else P.ls_warmup_frac
    ls_a = P.anneal_frac if P.get("ls_anneal_frac") is None else P.ls_anneal_frac
    lam_s = float(lambda_schedule(progress, P.lambda_s, ls_w, ls_a))
    if P.get("target_density") is None:
        return float(lambda_schedule(progress, P.lambda_c, P.warmup_frac, P.anneal_frac)), lam_s, None
    target = target_density_schedule(progress, float(P.target_density), P.warmup_frac, P.anneal_frac)
    return float(lam_dual), lam_s, target


def _dual_active(P: DictConfig, progress: float) -> bool:
    return P.get("target_density") is not None and progress >= P.warmup_frac


def run_training(cfg: DictConfig, verbose: bool = True) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    key = seed_everything(int(cfg.seed))
    setup = make_setup(cfg)
    k_init, key = jax.random.split(key)
    ts = init_train_state(setup, cfg, k_init)
    update = jax.jit(make_update(setup, cfg))
    wb = _maybe_wandb(cfg)

    total = int(cfg.algo.total_updates)
    P = cfg.penalties
    steps_per_update = int(cfg.algo.num_envs) * setup.env.max_steps
    rows = []
    t_start = time.time()
    t_compiled = None
    lam_dual, integ = 0.0, 0.0
    for u in range(total):
        progress = u / max(total, 1)
        lam_c, lam_s, target = _lambdas(P, progress, lam_dual)
        key, k = jax.random.split(key)
        ts, metrics = update(ts, k, lam_c, lam_s)
        if u == 0:
            jax.block_until_ready(metrics)
            t_compiled = time.time()
        if _dual_active(P, progress):
            integ, lam_dual = pi_step(integ, float(metrics["p_on_mean"]), target, P.dual_lr, P.get("dual_kp", 0.0),
                                      P.dual_clip)
            integ, lam_dual = float(integ), float(lam_dual)
        if u % int(cfg.logging.log_every) == 0 or u == total - 1:
            m = {k_: float(v) for k_, v in jax.device_get(metrics).items()}
            elapsed = time.time() - (t_compiled or t_start)
            m.update(update=u, env_steps=(u + 1) * steps_per_update,
                     fps=(u * steps_per_update / elapsed) if u > 0 and elapsed > 0 else float("nan"))
            if target is not None:
                m["target_density"] = target
            rows.append(m)
            if verbose:
                print(f"[{cfg.run_name} s{cfg.seed}] upd {u:5d} ret {m['episode_return']:8.3f} "
                      f"succ {m['success_rate']:.2f} links {m['active_link_ratio']:.2f} "
                      f"sw {m['gate_switch_rate']:.3f} lc {lam_c:.3g} ls {lam_s:.3g}", flush=True)
            if wb is not None:
                wb.log(m, step=u)

    wall = time.time() - t_start
    run_time = time.time() - (t_compiled or t_start)
    fps = (total - 1) * steps_per_update / run_time if total > 1 and run_time > 0 else float("nan")
    extra = dict(wall_time_s=wall, fps_env_steps=fps, fps_agent_steps=fps * setup.env.n_agents,
                 peak_memory_mb=peak_memory_mb(), final_metrics=rows[-1] if rows else {})
    save_checkpoint(out_dir, ts.params, cfg, extra)
    if rows:
        with open(out_dir / "train_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if wb is not None:
        wb.summary.update({k_: v for k_, v in extra.items() if isinstance(v, (int, float))})
        wb.finish()
    if verbose:
        print(f"[train] saved {out_dir}  ({wall:.0f}s, {fps:.0f} env-steps/s)\n{json.dumps(extra['final_metrics'], default=float)}")
    return out_dir


_PER_RUN_KEYS = ("seed", "run_name", "out_dir", "exp_name", "penalties")


def _static_part(cfg: DictConfig) -> dict:
    c = OmegaConf.to_container(cfg, resolve=True)
    return {k: v for k, v in c.items() if k not in _PER_RUN_KEYS}


class DeadlineReached(RuntimeError):
    """Raised when ``WF_DEADLINE`` (unix time) is close: progress was checkpointed and the same
    command resumes it. Used to stop cleanly before a Kaggle session limit."""


def _deadline_close(margin_s: float = 120.0) -> bool:
    d = os.environ.get("WF_DEADLINE")
    return bool(d) and time.time() > float(d) - margin_s


def _resume_path(cfgs) -> Path:
    h = hashlib.sha1(json.dumps([str(c.out_dir) for c in cfgs] + [_static_part(cfgs[0])], sort_keys=True,
                                default=str).encode()).hexdigest()[:16]
    return Path(cfgs[0].out_dir).parents[1] / "_resume" / f"batch_{h}.pkl"


def _save_resume(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: a session killed mid-write keeps the previous checkpoint


def n_devices(max_devices: int | None = None) -> int:
    env = os.environ.get("WF_MAX_DEVICES")
    m = max_devices if max_devices is not None else (int(env) if env else None)
    n = jax.local_device_count()
    return max(1, min(n, m) if m else n)


def run_training_batch(cfgs: list, verbose: bool = True, max_devices: int | None = None) -> list:
    """Train several runs at once: the update is vmapped over runs, and the run axis is split
    across all local accelerators with ``pmap`` (e.g. both T4s of a Kaggle "GPU T4 x2"
    session; runs are independent, so there is no cross-device communication). The runs must
    share every setting except seed, run name, output directory and penalties (lambda values,
    density targets). Each run is seeded and checkpointed exactly like ``run_training``.

    Fault tolerance: the full training state (params, optimizer, PRNG keys, PI-controller
    state, metric rows) is saved atomically to ``runs/<exp>/_resume/`` every
    ``logging.ckpt_every`` updates; calling again with the same runs resumes from there. If
    the environment variable ``WF_DEADLINE`` (unix time) is about to pass, the state is saved
    and ``DeadlineReached`` is raised."""
    base = cfgs[0]
    ref = _static_part(base)
    for c in cfgs[1:]:
        if _static_part(c) != ref:
            raise ValueError(f"run {c.run_name} differs from {base.run_name} in a static setting")
    S = len(cfgs)
    D = min(n_devices(max_devices), S)
    pad = (-S) % D
    Sa = S + pad  # padded with copies of the last run (trained, never saved)
    all_cfgs = list(cfgs) + [cfgs[-1]] * pad
    for c in cfgs:
        Path(c.out_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(c, Path(c.out_dir) / "config.yaml", resolve=True)
    setup = make_setup(base)
    keys, k_inits = [], []
    for c in all_cfgs:
        k_init, k = jax.random.split(seed_everything(int(c.seed)))
        keys.append(k)
        k_inits.append(k_init)
    keys, k_inits = jax.numpy.stack(keys), jax.numpy.stack(k_inits)

    to_dev = (lambda x: x.reshape((D, Sa // D) + x.shape[1:])) if D > 1 else (lambda x: x)
    from_dev = (lambda x: np.asarray(x).reshape((Sa,) + np.shape(x)[2:])) if D > 1 else np.asarray
    init_fn = jax.vmap(lambda k: init_train_state(setup, base, k))
    upd = jax.vmap(make_update(setup, base))
    if D > 1:
        devs = jax.local_devices()[:D]
        init_fn = jax.pmap(init_fn, devices=devs)
        update = jax.pmap(upd, devices=devs)
    else:
        init_fn = jax.jit(init_fn)
        update = jax.jit(upd)
    split = jax.jit(jax.vmap(lambda k: jax.random.split(k)))

    total = int(base.algo.total_updates)
    ckpt_every = int(base.logging.get("ckpt_every", 200))
    steps_per_update = int(base.algo.num_envs) * setup.env.max_steps
    Ps = [c.penalties for c in all_cfgs]
    lam_dual, integ = np.zeros(Sa), np.zeros(Sa)
    rows = [[] for _ in range(Sa)]
    u0, wall_before = 0, 0.0
    rpath = _resume_path(cfgs)
    if rpath.exists():
        with open(rpath, "rb") as f:
            st = pickle.load(f)
        ts = jax.tree_util.tree_map(lambda x: to_dev(np.asarray(x)), st["ts"])
        keys = jax.numpy.asarray(st["keys"])
        lam_dual, integ, rows, u0, wall_before = st["lam_dual"], st["integ"], st["rows"], st["u"], st["wall"]
        if verbose:
            print(f"[train-batch] resumed {S} runs at update {u0}/{total} from {rpath}", flush=True)
    else:
        ts = init_fn(to_dev(k_inits))

    def checkpoint(u_next):
        _save_resume(rpath, dict(ts=jax.tree_util.tree_map(from_dev, jax.device_get(ts)), keys=np.asarray(keys),
                                 lam_dual=lam_dual, integ=integ, rows=rows, u=u_next,
                                 wall=wall_before + time.time() - t_start))

    t_start = time.time()
    t_compiled = None
    u_compiled = u0
    for u in range(u0, total):
        progress = u / max(total, 1)
        lams = [_lambdas(P, progress, lam_dual[i]) for i, P in enumerate(Ps)]
        lam_c = np.asarray([x[0] for x in lams], np.float32)
        lam_s = np.asarray([x[1] for x in lams], np.float32)
        kk = split(keys)
        keys, k = kk[:, 0], kk[:, 1]
        ts, metrics = update(ts, to_dev(k), to_dev(lam_c), to_dev(lam_s))
        if t_compiled is None:
            jax.block_until_ready(metrics)
            t_compiled, u_compiled = time.time(), u + 1
        dual_idx = [i for i, P in enumerate(Ps) if _dual_active(P, progress)]
        if dual_idx:
            dens = from_dev(jax.device_get(metrics["p_on_mean"]))
            for i in dual_idx:
                integ[i], lam_dual[i] = pi_step(integ[i], dens[i], lams[i][2], Ps[i].dual_lr,
                                                Ps[i].get("dual_kp", 0.0), Ps[i].dual_clip)
        if u % int(base.logging.log_every) == 0 or u == total - 1:
            mm = {k_: from_dev(v) for k_, v in jax.device_get(metrics).items()}
            elapsed = time.time() - t_compiled
            fps = ((u + 1 - u_compiled) * steps_per_update * Sa / elapsed) if u + 1 > u_compiled and elapsed > 0 else float("nan")
            for i in range(Sa):
                m = {k_: float(v[i]) for k_, v in mm.items()}
                m.update(update=u, env_steps=(u + 1) * steps_per_update, fps=fps / Sa)
                if lams[i][2] is not None:
                    m["target_density"] = lams[i][2]
                rows[i].append(m)
            if verbose:
                r, d = mm["episode_return"][:S], mm["p_on_mean"][:S]
                print(f"[batch {S} runs/{D} dev] upd {u:5d} ret mean {r.mean():8.3f} [{r.min():.2f}, {r.max():.2f}] "
                      f"P(on) [{d.min():.2f}, {d.max():.2f}] lc [{float(lam_c[:S].min()):.4f}, {float(lam_c[:S].max()):.4f}] "
                      f"{fps:.0f} env-steps/s", flush=True)
        if (u + 1) % ckpt_every == 0 and u + 1 < total:
            checkpoint(u + 1)
        if _deadline_close() and u + 1 < total:
            checkpoint(u + 1)
            raise DeadlineReached(f"deadline: {S} runs checkpointed at update {u + 1}/{total} ({rpath})")

    wall = wall_before + time.time() - t_start
    run_time = time.time() - (t_compiled or t_start)
    n_timed = total - u_compiled
    fps = n_timed * steps_per_update * Sa / run_time if n_timed > 0 and run_time > 0 else float("nan")
    params = jax.tree_util.tree_map(from_dev, jax.device_get(ts.params))
    out = []
    for i, c in enumerate(cfgs):
        extra = dict(wall_time_s=wall, fps_env_steps=fps / Sa, fps_env_steps_batch=fps, batch_size_runs=S,
                     n_devices=D, fps_agent_steps=fps / Sa * setup.env.n_agents, peak_memory_mb=peak_memory_mb(),
                     final_metrics=rows[i][-1] if rows[i] else {})
        save_checkpoint(c.out_dir, jax.tree_util.tree_map(lambda x: x[i], params), c, extra)
        if rows[i]:
            with open(Path(c.out_dir) / "train_metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[i][0].keys()))
                w.writeheader()
                w.writerows(rows[i])
        out.append(Path(c.out_dir))
    if rpath.exists():
        rpath.unlink()
    if verbose:
        print(f"[train-batch] saved {S} runs on {D} device(s) ({wall:.0f}s, {fps:.0f} env-steps/s in total)", flush=True)
    return out


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    np.set_printoptions(precision=4)
    run_training(cfg)


if __name__ == "__main__":
    main()
