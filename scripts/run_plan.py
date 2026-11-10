"""Run the whole E1-E9 plan in order, resumable across Kaggle sessions.

    python scripts/run_plan.py --budget-hours 11.3                 # everything, as far as the budget allows
    python scripts/run_plan.py --phases e1 e2_pilot e2 --budget-hours 6
    python scripts/run_plan.py --status                            # what is done / pending
    python scripts/run_plan.py --estimate --env-steps-per-s 60000  # projected GPU hours per phase

How it survives session limits:
  * ``WF_DEADLINE`` is set to now + budget; training checkpoints and stops cleanly before it
    (exit code 3) and the next call resumes the interrupted batch from its last checkpoint;
  * finished phases are recorded in ``results/plan_status.json`` and skipped next time;
  * trained runs are cached on disk (``runs/``), so a re-run phase only trains what is missing.
Carry ``runs/`` and ``results/`` from one session to the next (see kaggle/README_KAGGLE.md).

Phase choices that depend on earlier phases are read from their outputs:
  * the task variant from ``results/e1_calibrate/task_choice.json`` (passed as ``--set env=<chosen>``
    to every later phase; default ``env=wildfire`` if calibration was skipped);
  * the E3 lambda_s values from ``results/e3_pilot/e3_pilot_decisions.json`` (default 0.01 / 0.03).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results" / "plan_status.json"
DEADLINE_EXIT = 3
QUICK = False  # --quick: tiny budgets, results/quick_* (for testing the plan end to end)


def _res(name):
    return ROOT / "results" / (("quick_" if QUICK else "") + name)


def task_choice():
    p = _res("e1_calibrate") / "task_choice.json"
    return json.loads(p.read_text())["chosen"] if p.exists() else None


def ls_choice():
    p = _res("e3_pilot") / "e3_pilot_decisions.json"
    mid, high = 0.01, 0.03
    if p.exists():
        d = json.loads(p.read_text())
        mid = d.get("ls_mid") or mid
        high = d.get("ls_high") or high
    return mid, high


def phases(a):
    """``[(name, argv, n_runs_new, cost_factor)]``. ``n_runs_new`` = runs this phase trains (the rest
    are cached from earlier phases); ``cost_factor`` scales one N=4 run (pilots are shorter, N=50
    runs are larger). Used only for the time estimate."""
    seeds = ["--seeds", *map(str, a.seeds)]
    s9 = ["--seeds", *map(str, a.seeds[:3])]
    common = (["--updates", str(a.updates)] if a.updates else []) + ["--max-batch", str(a.max_batch)]
    common += ["--quick"] if a.quick else []
    task = task_choice()
    task_set = ["--set", f"env={task}"] if task else []
    mid, high = ls_choice()
    ls = ["--ls", "0", f"{mid:g}", f"{high:g}"]
    rho = ["--rho", "0.5", "0.25", "0.1"]
    k = len(a.seeds)
    s3 = ["--seeds", *map(str, a.seeds[:3])]
    plan = [
        ("e1_calibrate", ["e1_task_validation.py", "--calibrate", *s3], 18, a.calib_factor),
        ("e1", ["e1_task_validation.py", *seeds], 5 * k, 1.0),
        ("e2_pilot", ["e2_sparsity_sweep.py", "--pilot", *seeds], 6, 1 / 3),
        ("e2", ["e2_sparsity_sweep.py", *seeds], 11 * k, 1.0),
        ("e3_pilot", ["e3_persistence_grid.py", "--pilot", *seeds], 5, 1.0),
        ("e3", ["e3_persistence_grid.py", *rho, *ls, *seeds], 6 * k, 1.0),
        ("e4", ["e4_matched_controls.py", "--rho", "0.25", "--ls", "0", *seeds], 4 * k, 1.0),
        ("e5", ["e5_event_association.py", *rho, *ls, *seeds], 0, 1.0),
        ("e6", ["e6_message_use.py", "--rho", "1", "0.25", "0.1", "--ls", "0", f"{high:g}", *seeds], 1 * k, 1.0),
        ("e7", ["e7_message_probe.py", *seeds], 0, 1.0),
        ("e8", ["e8_method_comparison.py", "--rho", "0.25", "--with-ic3net", *seeds], 3 * k, 1.0),
        ("e9", ["e9_transfer_scale.py", "--rho", "0.25", "--ls", f"{high:g}", *s9], 3 * 20, 1.0 + 12 / 60 * a.n50_factor),
    ]
    # every phase after calibration runs on the chosen task variant
    plan = [(n, argv + (task_set if n != "e1_calibrate" else []), r, f) for n, argv, r, f in plan]
    return plan, common


def load_status():
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def save_status(st):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(st, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", nargs="*", default=None, help="subset of phases (default: all, in order)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=None, help="training updates per run (default: configs/algo)")
    ap.add_argument("--max-batch", type=int, default=24, help="runs per vmapped program")
    ap.add_argument("--budget-hours", type=float, default=11.3, help="stop cleanly before this much wall time")
    ap.add_argument("--force", action="store_true", help="re-run phases already marked done")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny budgets in runs/quick_* (tests the plan end to end)")
    ap.add_argument("--estimate", action="store_true", help="print projected hours and exit")
    ap.add_argument("--env-steps-per-s", type=float, default=None, help="measured total throughput (scripts/benchmark.py)")
    ap.add_argument("--n50-factor", type=float, default=8.0, help="cost of one N=50 run relative to one N=4 run")
    ap.add_argument("--calib-factor", type=float, default=1.5, help="cost of a calibration run (larger grids) vs a default run")
    a = ap.parse_args()
    global QUICK, STATUS
    if a.quick:
        QUICK, STATUS = True, ROOT / "results" / "quick_plan_status.json"
    plan, common = phases(a)
    names = [p[0] for p in plan]
    sel = a.phases or names
    bad = [x for x in sel if x not in names]
    if bad:
        raise SystemExit(f"unknown phases {bad}; choose from {names}")
    st = load_status()

    if a.status or a.estimate:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose("config")
        upd = a.updates or int(cfg.algo.total_updates)
        steps = upd * int(cfg.algo.num_envs) * int(cfg.env.max_steps)
        tot = 0.0
        for name, argv, n_new, f in plan:
            if name not in sel:
                continue
            h = n_new * f * steps / a.env_steps_per_s / 3600 if a.env_steps_per_s else float("nan")
            tot += 0 if st.get(name, {}).get("done") else (h if h == h else 0)
            print(f"{name:9s} {'done' if st.get(name, {}).get('done') else 'pending':8s} new runs ~{n_new:3d} "
                  f"x{f:.2f}  est {h:6.2f} h   {' '.join(argv)}")
        if a.env_steps_per_s:
            print(f"remaining (pending phases): ~{tot:.1f} GPU hours at {a.env_steps_per_s:.0f} env-steps/s, {upd} updates/run")
        return

    deadline = time.time() + a.budget_hours * 3600
    env = dict(os.environ, WF_DEADLINE=str(deadline), PYTHONUNBUFFERED="1")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    for name in names:
        if name not in sel or (st.get(name, {}).get("done") and not a.force):
            continue
        # rebuild the phase list now: earlier phases decide the task variant and lambda_s values
        plan, common = phases(a)
        argv = dict((p[0], p[1]) for p in plan)[name]
        if time.time() > deadline - 300:
            print(f"budget nearly used: stopping before {name}", flush=True)
            break
        cmd = [sys.executable, str(ROOT / "scripts" / argv[0]), *argv[1:], *common]
        print(f"\n######## {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        st = load_status()
        st[name] = dict(done=rc == 0, last_exit=rc, seconds=round(time.time() - t0), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_status(st)
        if rc == DEADLINE_EXIT:
            print(f"\n*** {name}: session budget reached, progress saved. Run the same command next session.", flush=True)
            sys.exit(DEADLINE_EXIT)
        if rc != 0:
            print(f"\n*** {name} failed (exit {rc}); fix and re-run (finished phases are skipped).", flush=True)
            sys.exit(rc)
    print("\nplan finished for:", [n for n in sel if load_status().get(n, {}).get("done")], flush=True)


if __name__ == "__main__":
    main()
