"""Seeding, reproducibility metadata, checkpoints and resource measurement."""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import jax
import numpy as np
from flax import serialization
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> jax.Array:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return jax.random.PRNGKey(seed)


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        h = out.stdout.strip() or "unknown"
        return h + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # git missing
        return "unknown"


def save_checkpoint(out_dir: str | Path, params, cfg: DictConfig, extra: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(serialization.msgpack_serialize(jax.device_get(serialization.to_state_dict(params))))
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    meta = {"git_hash": git_hash(), **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


def load_checkpoint(run_dir: str | Path):
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    params = serialization.msgpack_restore((run / "params.msgpack").read_bytes())
    params = jax.tree_util.tree_map(jax.numpy.asarray, params)
    return cfg, params


def has_checkpoint(run_dir: str | Path) -> bool:
    return (Path(run_dir) / "params.msgpack").exists()


def peak_memory_mb() -> float:
    """Peak device memory if the backend reports it (GPU), else peak process memory."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats["peak_bytes_in_use"] / 2**20
    except Exception:
        pass
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or mi.rss  # peak_wset exists on Windows
        if peak == mi.rss:
            try:
                import resource  # Linux/macOS: ru_maxrss in KiB (Linux)

                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except Exception:
                pass
        return peak / 2**20
    except Exception:
        return float("nan")
