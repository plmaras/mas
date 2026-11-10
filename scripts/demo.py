"""Demonstration figures for one evaluation episode (same layout for every model):

  * an animated GIF: burning cells, firefighters, open links (arrows), side by side per model;
  * a link raster: every directed link over time (open = dark), event times marked.

    python scripts/demo.py --runs runs/core/nocomm/seed_0 runs/core/fullcomm/seed_0 \
        runs/core/gated_rho0.25_ls0/seed_0 runs/core/gated_rho0.25_ls0.03/seed_0 --episode 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from eval import load_logs  # noqa: E402


def _load(run, tag, ep):
    logs, _ = load_logs(Path(run) / f"eval_{tag}.npz")
    cfg = OmegaConf.load(Path(run) / "config.yaml")
    return {k: v[ep] for k, v in logs.items() if np.ndim(v) >= 1 and v.shape[0] == logs["positions"].shape[0]}, cfg


def link_raster(runs, tag, ep, path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 1.6 + 1.4 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        L, cfg = _load(run, tag, ep)
        g, m, E = L["g"] > 0, L["mask"], L["E"]
        N = g.shape[1]
        links = [(i, j) for i in range(N) for j in range(N) if i != j]
        img = np.array([[1.0 if g[t, i, j] and m[t, i, j] else 0.0 for t in range(g.shape[0])] for i, j in links])
        ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest")
        ev = np.array([[E[t, i, j] for t in range(g.shape[0])] for i, j in links])
        yy, xx = np.nonzero(ev)
        ax.scatter(xx, yy, s=6, color="#d55e00", marker="|", label="event (sender sees a fire receiver can't)")
        ax.set_yticks(range(len(links)))
        ax.set_yticklabels([f"{i}→{j}" for i, j in links], fontsize=6)
        ax.set_title(f"{cfg.run_name}  (open links dark)", loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("time step")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(runs, tag, ep, path, fps=3):
    data = [_load(r, tag, ep) for r in runs]
    T = data[0][0]["positions"].shape[0]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.4), squeeze=False)
    axes = axes[0]

    def draw(t):
        for ax, (L, cfg) in zip(axes, data):
            ax.clear()
            side = int(round(L["covered"].shape[-1] ** 0.5))
            burning = (~L["covered"][t]).reshape(side, side)
            ax.imshow(burning, cmap="OrRd", vmin=0, vmax=1.3, origin="upper")
            pos = L["positions"][t]
            g, m = L["g"][t] > 0, L["mask"][t]
            for i in range(pos.shape[0]):
                for j in range(pos.shape[0]):
                    if i != j and g[i, j] and m[i, j]:
                        ax.annotate("", xy=(pos[j, 1], pos[j, 0]), xytext=(pos[i, 1], pos[i, 0]),
                                    arrowprops=dict(arrowstyle="->", color="#0072b2", lw=1, alpha=0.7))
            ax.scatter(pos[:, 1], pos[:, 0], s=60, color="#009e73", edgecolor="white", zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{cfg.run_name}\nt={t}", fontsize=8)

    anim = FuncAnimation(fig, draw, frames=T)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="stoch")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "demo"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gated = [r for r in a.runs if "nocomm" not in r]
    link_raster(gated or a.runs, a.tag, a.episode, out / "demo_link_raster.png")
    animate(a.runs, a.tag, a.episode, out / "demo_episode.gif")
    print(f"wrote {out}/demo_link_raster.png and demo_episode.gif")


if __name__ == "__main__":
    main()
