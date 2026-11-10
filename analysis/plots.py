"""Static matplotlib figures for the experiment reports.

Conventions: categorical hues in a fixed order (never cycled), sequential = single blue hue
light->dark, recessive grid/axes, per-seed raw points always shown next to CIs, one y-axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
    "legend.frameon": False,
})


def color(i: int) -> str:
    if i >= len(CATEGORICAL):
        raise ValueError("more than 8 series: fold into 'Other' or use small multiples")
    return CATEGORICAL[i]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def points_with_ci(groups: dict, ylabel: str, path, title: str = "", seed=0):
    """One column per condition: raw per-seed points (jittered) + mean with bootstrap 95% CI."""
    from analysis.stats import bootstrap_ci

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(groups) + 1.5), 3.6))
    rng = np.random.default_rng(seed)
    for k, (name, vals) in enumerate(groups.items()):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        c = CATEGORICAL[k % len(CATEGORICAL)] if len(groups) <= 8 else CATEGORICAL[0]
        ax.scatter(k + rng.uniform(-0.12, 0.12, v.size), v, s=22, color=c, alpha=0.55, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        m, lo, hi = bootstrap_ci(v)
        if np.isfinite(lo):
            ax.errorbar(k + 0.28, m, yerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=6, capsize=3, lw=1.5, zorder=4)
        else:
            ax.plot(k + 0.28, m, "o", color=INK, ms=6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    ax.text(1.0, 1.02, "dots: seeds   black: mean ± 95% bootstrap CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    return _save(fig, path)


def xy_by_group(series: dict, xlabel: str, ylabel: str, path, title: str = "", connect: bool = True,
                xlim=None, ylim=None):
    """``series[name] = list of (x_values_per_seed, y_values_per_seed)`` points per setting.
    Draws raw per-seed points and the per-setting mean with 95% CI in both x and y."""
    from analysis.stats import bootstrap_ci

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for k, (name, pts) in enumerate(series.items()):
        c = color(k)
        mx, my = [], []
        for xs, ys in pts:
            xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            ax.scatter(xs, ys, s=16, color=c, alpha=0.35, edgecolor="none")
            x0, xlo, xhi = bootstrap_ci(xs)
            y0, ylo, yhi = bootstrap_ci(ys)
            ax.errorbar(x0, y0, xerr=None if not np.isfinite(xlo) else [[x0 - xlo], [xhi - x0]],
                        yerr=None if not np.isfinite(ylo) else [[y0 - ylo], [yhi - y0]],
                        fmt="o", color=c, ms=8, mec="#fcfcfb", mew=1.5, capsize=2, lw=1.2)
            mx.append(x0)
            my.append(y0)
        if connect and len(mx) > 1:
            o = np.argsort(mx)
            ax.plot(np.array(mx)[o], np.array(my)[o], color=c, lw=2, alpha=0.8, label=name)
        else:
            ax.plot([], [], "o", color=c, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    if len(series) >= 2:
        ax.legend(loc="best", fontsize=8)
    return _save(fig, path)


def heatmaps(mats: dict, row_labels, col_labels, row_name: str, col_name: str, path, fmt: str = "{:.2f}"):
    """Small multiples: one sequential heatmap per metric, value printed in each cell."""
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.3))
    axes = np.atleast_1d(axes)
    for ax, (name, M) in zip(axes, mats.items()):
        M = np.asarray(M, float)
        im = ax.imshow(M, cmap=SEQ_BLUE, origin="lower", aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            if np.isfinite(v):
                frac = (v - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-12)
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="#ffffff" if frac > 0.55 else INK)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def scatter_colored(x, y, c, xlabel, ylabel, clabel, path, title: str = "", labels=None):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    sc = ax.scatter(x, y, c=c, cmap=SEQ_BLUE, s=36, edgecolor=AXIS, linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(clabel)
    if labels is not None:
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(li, (xi, yi), fontsize=6, color=MUTED, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, path)


def null_histogram(null, observed, xlabel, path, title: str = "", null_label: str = "null"):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(null, bins=40, color=CATEGORICAL[0], alpha=0.85, edgecolor="#fcfcfb", linewidth=0.5,
            label=f"{null_label} (n={null.size})")
    ax.axvline(observed, color=INK, lw=2, label=f"observed = {observed:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, path)


def lines_by_group(x, ys: dict, xlabel, ylabel, path, title: str = "", xticklabels=None):
    """``ys[name] = [n_seeds, len(x)]``; draws per-seed thin lines + mean line."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for k, (name, Y) in enumerate(ys.items()):
        Y = np.atleast_2d(np.asarray(Y, float))
        for row in Y:
            ax.plot(x, row, color=color(k), lw=0.8, alpha=0.3)
        ax.plot(x, np.nanmean(Y, 0), color=color(k), lw=2, marker="o", ms=5, label=name)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(ys) >= 2:
        ax.legend(fontsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return _save(fig, path)
