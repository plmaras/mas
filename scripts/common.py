"""Shared helpers for the experiment scripts (E1-E9) on the wildfire ("FireWorkers") task.

Runs live in ``runs/<exp_name>/<run_name>/seed_<s>/`` and are reused across experiments
(E1-E3 train the core models, E4-E9 reuse them). Every pending (condition, seed) is grouped
with the others that share the same static settings and trained as ONE vmapped program,
split across all local GPUs (``train.run_training_batch``); ``--max-batch`` caps the number
of runs per program. ``--quick`` uses a tiny budget and separate ``runs/quick_*`` /
``results/quick_*`` namespaces to check that a script runs end to end.

v2 design choices applied everywhere (see README, "Design"):
  * learned gates are trained to a TARGET link density rho (PI-controlled lambda_c), and the
    switching penalty lambda_s is switched on only after the target is reached;
  * sampled gates (as in training) are the primary evaluation (``--tag stoch``); deterministic
    gates are a robustness check (``--tag det``);
  * one statistic per seed, 95 % t-intervals across seeds, Holm correction within a family.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from analysis.stats import summarize_seeds  # noqa: E402

QUICK = ["algo.total_updates=6", "algo.num_envs=8", "algo.num_minibatches=2", "algo.update_epochs=1",
         "eval.episodes=16", "eval.batch_size=16", "logging.log_every=2"]
DEADLINE_EXIT = 3  # exit code when WF_DEADLINE stopped training (re-run the same command to resume)


def parser(desc: str, default_seeds=(0, 1, 2, 3, 4)) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(default_seeds))
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke run (2 seeds, few updates)")
    ap.add_argument("--exp-name", default="core", help="namespace of the shared training runs")
    ap.add_argument("--set", nargs="*", default=[], help="extra Hydra overrides applied to every run")
    ap.add_argument("--updates", type=int, default=None, help="training updates per run (default: config)")
    ap.add_argument("--tag", default="stoch", choices=["stoch", "det"],
                    help="gate evaluation: stoch = sampled gates as in training (primary), det = deterministic")
    ap.add_argument("--max-batch", type=int, default=int(os.environ.get("WF_MAX_BATCH", 24)),
                    help="max runs trained together in one vmapped program (memory cap)")
    ap.add_argument("--train-only", action="store_true", help="train missing runs, then stop (no evaluation)")
    ap.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    return ap


def finalize_args(a):
    if a.quick:
        a.seeds = a.seeds[:2]
        a.exp_name = "quick_" + a.exp_name
    return a


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def base_overrides(a):
    ov = [f"exp_name={a.exp_name}"] + (QUICK if a.quick else [])
    if a.updates and not a.quick:
        ov.append(f"algo.total_updates={a.updates}")
    return ov + list(a.set)


def make_run_cfg(name, ov, seed, a):
    cfg = compose_cfg(base_overrides(a) + [f"run_name={name}"] + list(ov) + [f"seed={seed}"])
    cfg.out_dir = str(ROOT / "runs" / a.exp_name / name / f"seed_{seed}")  # absolute, cwd-independent
    return cfg


MATCH_KEYS = ("env", "comm", "algo", "penalties")


def _settings(cfg) -> str:
    from omegaconf import OmegaConf

    c = OmegaConf.to_container(cfg, resolve=True)
    # `name` fields are labels (env=wildfire is an alias of a variant): compare settings only
    return json.dumps({k: {kk: vv for kk, vv in (c.get(k) or {}).items() if kk != "name"} for k in MATCH_KEYS},
                      sort_keys=True, default=str)


def check_cached(cfg, d: Path) -> None:
    """A cached run must have been trained with the settings now requested (env, comm, algo,
    penalties); otherwise experiments would silently mix tasks or budgets."""
    from omegaconf import OmegaConf

    old = OmegaConf.load(d / "config.yaml")
    if _settings(old) != _settings(cfg):
        a, b = json.loads(_settings(old)), json.loads(_settings(cfg))
        diff = [f"{k}.{kk}: {a[k].get(kk)!r} -> {b[k].get(kk)!r}" for k in MATCH_KEYS
                for kk in sorted(set(a[k] or {}) | set(b[k] or {})) if (a[k] or {}).get(kk) != (b[k] or {}).get(kk)]
        raise SystemExit(f"cached run {d} was trained with different settings:\n  " + "\n  ".join(diff[:12]) +
                         "\nUse another --exp-name, delete the run directory, or pass --retrain.")


def train_runs(conditions: dict, a, verbose: bool = True) -> dict:
    """``conditions[name] = list of Hydra overrides``. Trains every (condition, seed) without a
    checkpoint (batched by static settings, multi-GPU) and returns ``{name: [run_dir per seed]}``.
    Cached runs are reused only if they were trained with the same env/comm/algo/penalty settings."""
    from core.utils import has_checkpoint
    from train import _static_part, run_training_batch

    out, pending = {}, []
    for name, ov in conditions.items():
        dirs = []
        for s in a.seeds:
            cfg = make_run_cfg(name, ov, s, a)
            d = Path(cfg.out_dir)
            if a.retrain or not has_checkpoint(d):
                pending.append(cfg)
            else:
                check_cached(cfg, d)
            dirs.append(d)
        out[name] = dirs
    groups: dict[str, list] = {}
    for cfg in pending:
        groups.setdefault(json.dumps(_static_part(cfg), sort_keys=True, default=str), []).append(cfg)
    for cfgs in groups.values():
        for i in range(0, len(cfgs), a.max_batch):
            chunk = cfgs[i:i + a.max_batch]
            print(f"=== train {len(chunk)} runs: " + ", ".join(f"{c.run_name}/s{c.seed}" for c in chunk), flush=True)
            run_training_batch(chunk, verbose=verbose)
    if a.train_only:
        print("--train-only: all runs trained; stopping before evaluation", flush=True)
        sys.exit(0)
    return out


def eval_runs(runs: dict, tag: str = "stoch", **kw) -> dict:
    """Evaluate (cached ``eval_<tag>``) every run. Returns ``{name: [summary per seed]}``."""
    from eval import evaluate_run

    return {name: [evaluate_run(d, tag=tag, deterministic=(tag == "det"), **kw) for d in dirs]
            for name, dirs in runs.items()}


def per_seed(summaries: dict, key: str) -> dict:
    return {name: [s[key] for s in ss] for name, ss in summaries.items()}


def table(summaries: dict, keys) -> dict:
    return {name: {k: summarize_seeds([s.get(k, np.nan) for s in ss]) for k in keys} for name, ss in summaries.items()}


def results_dir(a, exp: str) -> Path:
    d = ROOT / "results" / (("quick_" if a.quick else "") + exp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_np_default))


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_table_csv(path, tab: dict, keys):
    """Rows = conditions; columns = mean / CI / IQM of each key."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + [f"{k}_{s}" for k in keys for s in ("mean", "ci_lo", "ci_hi", "iqm", "n")])
        for name, row in tab.items():
            w.writerow([name] + [row[k][s] for k in keys for s in ("mean", "ci_lo", "ci_hi", "iqm", "n")])


def fmt_ci(s: dict) -> str:
    return f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] (IQM {s['iqm']:.3f}, n={s['n']})"


# ----------------------------------------------------------------------------- conditions
def gname(rho, ls, prefix="gated"):
    """Run name of a learned-gate condition at target density ``rho`` (1 = no density target)."""
    return f"{prefix}_rho{rho:g}_ls{ls:g}"


def gated_overrides(rho, ls, comm="gated"):
    """Learned gates trained to target link density ``rho`` with switching penalty ``ls``
    (switched on after sparsification). ``rho >= 1`` -> unpenalised gates (lambda_c = 0)."""
    if rho is None or rho >= 1:
        return [f"comm={comm}", "penalties=density_target", "penalties.target_density=null", f"penalties.lambda_s={ls}"]
    return [f"comm={comm}", "penalties=density_target", f"penalties.target_density={rho}", f"penalties.lambda_s={ls}"]


def baseline_conditions():
    return {"nocomm": ["comm=nocomm"], "fullcomm": ["comm=fullcomm"]}


# ----------------------------------------------------------------------------- statistics
def t_ci(x, ci: float = 0.95) -> dict:
    """Mean and t-interval over seeds (one number per seed)."""
    from scipy import stats as sps

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), sd=float("nan"), n=0)
    m = float(x.mean())
    if n == 1:
        return dict(mean=m, lo=float("nan"), hi=float("nan"), sd=float("nan"), n=1)
    sd = float(x.std(ddof=1))
    h = float(sps.t.ppf((1 + ci) / 2, n - 1) * sd / np.sqrt(n))
    return dict(mean=m, lo=m - h, hi=m + h, sd=sd, n=n)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values within one family of tests."""
    keys = [k for k in pvals if np.isfinite(pvals[k])]
    order = sorted(keys, key=lambda k: pvals[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    for k in pvals:
        out.setdefault(k, float("nan"))
    return out


def ttest_1samp(x) -> float:
    from scipy import stats as sps

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(sps.ttest_1samp(x, 0).pvalue) if x.size > 1 and np.std(x) > 0 else float("nan")


def welch(x, y) -> float:
    from scipy import stats as sps

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    return float(sps.ttest_ind(x, y, equal_var=False).pvalue)


def ratio_ci(full, nocomm, gated, n_boot: int = 20000, seed: int = 0) -> dict:
    """Bootstrap CI of the loss fraction (FullComm - X) / (FullComm - NoComm), seeds resampled
    independently per group."""
    rng = np.random.default_rng(seed)
    f, n, g = (np.asarray(x, float) for x in (full, nocomm, gated))
    if min(f.size, n.size, g.size) == 0:
        return dict(point=float("nan"), lo=float("nan"), hi=float("nan"))
    bf = f[rng.integers(0, f.size, (n_boot, f.size))].mean(1)
    bn = n[rng.integers(0, n.size, (n_boot, n.size))].mean(1)
    bg = g[rng.integers(0, g.size, (n_boot, g.size))].mean(1)
    r = (bf - bg) / np.where(np.abs(bf - bn) > 1e-12, bf - bn, np.nan)
    point = (f.mean() - g.mean()) / (f.mean() - n.mean()) if abs(f.mean() - n.mean()) > 1e-12 else float("nan")
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi))


def association(run_dir, a, seed: int = 0, n_perm: int | None = None) -> dict:
    from analysis.association import association_run

    n = n_perm if n_perm is not None else (50 if a.quick else 1000)
    return association_run(run_dir, tag=a.tag, n_perm=n, seed=seed)


def run_main(main):
    """Run a script's ``main``; a WF_DEADLINE stop exits with code 3 (progress is saved)."""
    from train import DeadlineReached

    try:
        main()
    except DeadlineReached as e:
        print(f"\n*** {e}\n*** re-run the same command to resume", flush=True)
        sys.exit(DEADLINE_EXIT)
