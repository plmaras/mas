"""Measure training throughput on this machine (run first on Kaggle) and project the plan's cost.

    python scripts/benchmark.py                    # batch sizes 1, 8, 24 at N=4, plus N=50
    python scripts/benchmark.py --batches 8 16 24 32 --updates 5

Prints env-steps/s for one vmapped program split across all local GPUs (as the experiment
scripts train), the speed-up from batching, and the projected GPU hours of the full plan
(``scripts/run_plan.py --estimate``) at the best measured throughput.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bench(n_runs, overrides, updates):
    import jax
    import numpy as np
    from hydra import compose, initialize_config_dir

    from core.mappo import init_train_state, make_update
    from core.setup import make_setup
    from train import n_devices

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose("config", overrides=["comm=gated", "penalties=density_target", *overrides])
    setup = make_setup(cfg)
    D = min(n_devices(), n_runs)
    S = n_runs + (-n_runs) % D
    init = jax.vmap(lambda k: init_train_state(setup, cfg, k))
    upd = jax.vmap(make_update(setup, cfg))
    keys = jax.random.split(jax.random.PRNGKey(0), S)
    lam = np.zeros(S, np.float32)
    rs = (lambda x: x.reshape((D, S // D) + x.shape[1:])) if D > 1 else (lambda x: x)
    if D > 1:
        init, upd = jax.pmap(init), jax.pmap(upd)
    else:
        init, upd = jax.jit(init), jax.jit(upd)
    ts = init(rs(keys))
    t0 = time.time()
    ts, m = upd(ts, rs(keys), rs(lam), rs(lam))
    jax.block_until_ready(m)
    compile_s = time.time() - t0
    t0 = time.time()
    for _ in range(updates):
        ts, m = upd(ts, rs(keys), rs(lam), rs(lam))
    jax.block_until_ready(m)
    dt = (time.time() - t0) / updates
    steps = S * int(cfg.algo.num_envs) * int(cfg.env.max_steps)
    return dict(runs=n_runs, devices=D, compile_s=compile_s, s_per_update=dt, env_steps_per_s=steps / dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 24])
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--skip-n50", action="store_true")
    a = ap.parse_args()
    import jax

    print(f"devices: {jax.devices()}", flush=True)
    best = 0.0
    for b in a.batches:
        r = bench(b, [], a.updates)
        best = max(best, r["env_steps_per_s"])
        print(f"N=4  batch {b:3d} runs on {r['devices']} dev: {r['env_steps_per_s']:9.0f} env-steps/s "
              f"({r['s_per_update']:.2f} s/update, compile {r['compile_s']:.0f} s)", flush=True)
    if not a.skip_n50:
        r = bench(3, ["env.n_agents=50", "algo.num_envs=16", "algo.num_minibatches=4"], max(1, a.updates // 2))
        print(f"N=50 batch   3 runs on {r['devices']} dev: {r['env_steps_per_s']:9.0f} env-steps/s "
              f"({r['s_per_update']:.2f} s/update, compile {r['compile_s']:.0f} s)", flush=True)
    print(f"\nprojected plan at {best:.0f} env-steps/s:", flush=True)
    subprocess.call([sys.executable, str(ROOT / "scripts" / "run_plan.py"), "--estimate", "--env-steps-per-s", str(best)])


if __name__ == "__main__":
    main()
