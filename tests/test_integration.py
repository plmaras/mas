"""End-to-end smoke tests: a few training updates, evaluation logging, replay consistency."""
import jax
import numpy as np
import pytest

from analysis.faithfulness import consistency_check, f1_zero_messages, make_receiver_fn
from core.rollout import Interv
from eval import evaluate, load_logs, save_logs
from train import run_training, run_training_batch
from core.utils import load_checkpoint

TINY = ["algo.total_updates=2", "algo.num_envs=4", "algo.num_minibatches=2", "algo.update_epochs=1",
        "env.max_steps=10", "eval.episodes=4", "logging.log_every=1"]

LOG_KEYS = ["positions", "actions", "action_logits", "alpha", "g", "g_sampled", "mask", "messages", "attn",
            "vis", "covered", "E", "reward", "hidden", "landmark_pos", "success"]


@pytest.mark.parametrize("comm,env", [("gated", "wildfire"), ("gated_reinforce", "wildfire"), ("language", "wildfire"),
                                      ("gated", "spread"), ("aecomm", "wildfire_r2")])
def test_train_eval_roundtrip(make_cfg, tmp_path, comm, env):
    cfg = make_cfg(f"env={env}", f"comm={comm}", "penalties=persist", "penalties.warmup_frac=0.0", f"out_dir={tmp_path.as_posix()}/run", *TINY)
    out = run_training(cfg, verbose=False)
    cfg2, params = load_checkpoint(out)
    logs, summary = evaluate(cfg2, params, n_episodes=4)
    for k in LOG_KEYS:
        assert k in logs, k
    T, N = cfg.env.max_steps, cfg.env.n_agents
    L = logs["covered"].shape[-1]  # landmarks (spread) or grid cells (wildfire)
    assert logs["g"].shape == (4, T, N, N) and logs["vis"].shape == (4, T, N, L)
    if env.startswith("wildfire"):
        assert "action_mask" in logs and "final_burned_frac" in summary
        # sampled actions are always allowed by the env's action mask
        assert np.take_along_axis(logs["action_mask"], logs["actions"][..., None], -1).all()
    assert np.isfinite(logs["action_logits"]).all()
    assert 0 <= summary["active_link_ratio"] <= 1
    # teacher-forced receiver reproduces logged logits exactly
    f = make_receiver_fn(cfg2, params)
    assert consistency_check(f, logs) < 1e-4
    assert f1_zero_messages(f, logs)["kl_mean_all"] >= 0
    save_logs(tmp_path / "e.npz", logs, summary)
    logs2, meta = load_logs(tmp_path / "e.npz")
    assert meta["n_episodes"] == 4 and np.array_equal(logs2["g"], logs["g"])


def test_eval_interventions_and_controls(make_cfg, tmp_path):
    cfg = make_cfg("comm=fullcomm", f"out_dir={tmp_path.as_posix()}/run", *TINY)
    out = run_training(cfg, verbose=False)
    cfg2, params = load_checkpoint(out)
    logs, s = evaluate(cfg2, params, n_episodes=4, interv=Interv(zero_messages=True))
    assert np.allclose(logs["messages"], 0)
    logs, s = evaluate(cfg2, params, n_episodes=4, comm_overrides={"gate_mode": "random_fixed", "density": 0.0})
    assert s["active_link_ratio"] == 0
    logs, s = evaluate(cfg2, params, n_episodes=4, interv=Interv(ablate="event"))
    assert not ((logs["g"] > 0) & logs["E"]).any()
    # zero-shot transfer to a larger N with the same weights
    logs, s = evaluate(cfg2, params, n_episodes=2, env_overrides={"n_agents": 8})
    assert logs["g"].shape[-1] == 8
    # cross-play
    logs, s = evaluate(cfg2, params, n_episodes=2, sender_params=params["actor"])
    assert np.isfinite(logs["action_logits"]).all()


def test_batched_training_matches_single(make_cfg, tmp_path):
    """vmapped batch training reproduces independent single runs (same seeds, same lambdas)."""
    ov = ["comm=gated", "penalties=persist", "penalties.warmup_frac=0.0", *TINY]
    singles, batch = [], []
    for s in (0, 1):
        singles.append(make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/single_{s}"))
        batch.append(make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/batch_{s}"))
    for c in singles:
        run_training(c, verbose=False)
    run_training_batch(batch, verbose=False)
    for s in (0, 1):
        _, p1 = load_checkpoint(tmp_path / f"single_{s}")
        _, p2 = load_checkpoint(tmp_path / f"batch_{s}")
        diff = jax.tree_util.tree_map(lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), p1, p2)
        assert max(jax.tree_util.tree_leaves(diff)) < 1e-3


def test_batched_training_rejects_mixed_static_settings(make_cfg, tmp_path):
    a = make_cfg("comm=gated", f"out_dir={tmp_path.as_posix()}/a", *TINY)
    b = make_cfg("comm=fullcomm", f"out_dir={tmp_path.as_posix()}/b", *TINY)
    with pytest.raises(ValueError):
        run_training_batch([a, b], verbose=False)


def test_density_target_logs_dual_lambda(make_cfg, tmp_path):
    cfg = make_cfg("comm=gated", "penalties.target_density=0.3", "penalties.warmup_frac=0.0",
                   "penalties.anneal_frac=0.5", "penalties.dual_lr=0.01", f"out_dir={tmp_path.as_posix()}/run",
                   *TINY[:-1], "algo.total_updates=6", "logging.log_every=1")
    out = run_training(cfg, verbose=False)
    import pandas as pd

    m = pd.read_csv(out / "train_metrics.csv")
    assert np.isclose(m["target_density"].iloc[-1], 0.3)
    # gates start near-on (P ~ 0.96) and the target falls below that, so lambda_c must rise
    assert m["lambda_c"].iloc[-1] > 0


def _max_param_diff(a, b):
    _, p1 = load_checkpoint(a)
    _, p2 = load_checkpoint(b)
    d = jax.tree_util.tree_map(lambda x, y: float(np.max(np.abs(np.asarray(x) - np.asarray(y)))), p1, p2)
    return max(jax.tree_util.tree_leaves(d))


def test_deadline_checkpoint_and_resume_reproduce_uninterrupted_run(make_cfg, tmp_path, monkeypatch):
    """A batch stopped by WF_DEADLINE and resumed later ends with the same weights as an
    uninterrupted batch (Kaggle sessions end after ~12 h; training continues next session)."""
    import time

    from train import DeadlineReached

    ov = ["comm=gated", "penalties.target_density=0.5", "penalties.warmup_frac=0.0", *TINY[:-1],
          "algo.total_updates=4", "logging.log_every=1", "logging.ckpt_every=100"]
    ref = [make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/ref/r/seed_{s}") for s in (0, 1)]
    run_training_batch(ref, verbose=False)
    cut = [make_cfg(*ov, f"seed={s}", f"out_dir={tmp_path.as_posix()}/cut/r/seed_{s}") for s in (0, 1)]
    monkeypatch.setenv("WF_DEADLINE", str(time.time() + 60))  # already within the 120 s margin
    with pytest.raises(DeadlineReached):
        run_training_batch(cut, verbose=False)
    assert list((tmp_path / "cut" / "_resume").glob("batch_*.pkl"))
    monkeypatch.delenv("WF_DEADLINE")
    run_training_batch(cut, verbose=False)
    assert not list((tmp_path / "cut" / "_resume").glob("batch_*.pkl"))  # cleaned up when done
    for s in (0, 1):
        assert _max_param_diff(tmp_path / f"ref/r/seed_{s}", tmp_path / f"cut/r/seed_{s}") < 1e-6


MULTI_DEVICE_SCRIPT = r"""
import os, sys
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
sys.path.insert(0, {root!r})
import jax
from hydra import compose, initialize_config_dir
from train import run_training_batch
assert jax.local_device_count() == 2
TINY = {tiny!r}
def cfg(s, d):
    with initialize_config_dir(config_dir={root!r} + "/configs", version_base="1.3"):
        return compose("config", overrides=["comm=gated", "penalties.target_density=0.5", *TINY, f"seed={{s}}",
                                            f"out_dir={{d}}/seed_{{s}}"])
run_training_batch([cfg(s, {out!r} + "/multi/r") for s in (0, 1, 2)], verbose=False)              # 3 runs on 2 devices (padded)
run_training_batch([cfg(s, {out!r} + "/single/r") for s in (0, 1, 2)], verbose=False, max_devices=1)
print("OK")
"""


def test_multi_device_batch_matches_single_device(tmp_path):
    """pmap over 2 (simulated) devices x vmap over runs gives the same weights as 1 device."""
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    code = MULTI_DEVICE_SCRIPT.format(root=root, tiny=TINY, out=tmp_path.as_posix())
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-3000:]
    for s in (0, 1, 2):
        assert _max_param_diff(tmp_path / f"multi/r/seed_{s}", tmp_path / f"single/r/seed_{s}") < 1e-4
