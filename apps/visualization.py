import math
import os
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Jupyter is migrating its paths.*",
    category=DeprecationWarning,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import apply_paper_theme, style_axis


def _ensure_parent(save_path):
    parent = Path(save_path).expanduser().resolve().parent
    os.makedirs(parent, exist_ok=True)


def plot_confusion_matrix(cm, save_path, title, labels=("Control", "Positive"), dpi=300):
    _ensure_parent(save_path)
    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#4f7ea8", as_cmap=True),
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.8,
        linecolor="white",
        square=True,
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_interaction_heatmap(matrix_df, save_path, title="Gene Interaction Matrix", dpi=300):
    _ensure_parent(save_path)
    apply_paper_theme()
    fig_size = max(7.0, min(13.0, 0.34 * len(matrix_df) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    values = matrix_df.to_numpy(dtype=float)
    center = 0.0 if np.nanmin(values) < 0 < np.nanmax(values) else float(np.nanmean(values))
    sns.heatmap(
        matrix_df,
        cmap="RdBu_r",
        center=center,
        square=True,
        linewidths=0.15,
        linecolor="#f4f7fb",
        cbar_kws={"shrink": 0.72, "label": "Interaction strength"},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_top_interactions(matrix_df, top_n=30, min_abs_weight=0.0):
    matrix_df = pd.DataFrame(matrix_df).copy()
    genes = matrix_df.index.astype(str).tolist()
    edges = []
    for i, source in enumerate(genes):
        for j in range(i + 1, len(genes)):
            target = genes[j]
            weight = float(matrix_df.iloc[i, j])
            if not np.isfinite(weight):
                continue
            abs_weight = abs(weight)
            if abs_weight <= min_abs_weight:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "abs_weight": abs_weight,
                }
            )

    edges.sort(key=lambda item: item["abs_weight"], reverse=True)
    return edges[:top_n]


def _circle_layout(nodes):
    total = max(len(nodes), 1)
    return {
        node: (
            math.cos(2.0 * math.pi * idx / total),
            math.sin(2.0 * math.pi * idx / total),
        )
        for idx, node in enumerate(nodes)
    }


def plot_gene_interaction_network(
    matrix_df,
    save_path,
    *,
    top_n=30,
    min_abs_weight=0.0,
    title="Top Gene Interaction Network",
    dpi=300,
):
    _ensure_parent(save_path)
    edges = select_top_interactions(matrix_df, top_n=top_n, min_abs_weight=min_abs_weight)
    if not edges:
        return None

    nodes = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    positions = _circle_layout(nodes)
    max_abs_weight = max(edge["abs_weight"] for edge in edges)
    node_strength = {node: 0.0 for node in nodes}
    for edge in edges:
        node_strength[edge["source"]] += edge["abs_weight"]
        node_strength[edge["target"]] += edge["abs_weight"]

    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        width = 0.8 + 4.2 * edge["abs_weight"] / max_abs_weight
        color = "#3d74a6" if edge["weight"] >= 0 else "#c75b5b"
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width,
            alpha=0.32,
            solid_capstyle="round",
            zorder=1,
        )

    max_strength = max(node_strength.values()) if node_strength else 1.0
    for node in nodes:
        x, y = positions[node]
        size = 320 + 780 * node_strength[node] / max(max_strength, 1e-12)
        ax.scatter(
            [x],
            [y],
            s=size,
            color="#eef5fb",
            edgecolors="#4f7ea8",
            linewidths=1.3,
            zorder=2,
        )
        label_radius = 1.16
        ax.text(
            x * label_radius,
            y * label_radius,
            node,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1f3344",
            zorder=3,
        )

    ax.set_title(title, pad=16)
    ax.text(
        0.02,
        0.02,
        "Blue: positive interaction | Red: negative interaction | Width: absolute strength",
        transform=ax.transAxes,
        fontsize=8,
        color="#52677a",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return edges


def plot_ranked_importance(
    data,
    *,
    score_col,
    label_col,
    save_path,
    title,
    xlabel,
    ylabel="Gene",
    top_n=15,
    dpi=300,
):
    from plot_style import ranked_barplot

    plot_df = data.sort_values(score_col, ascending=False).head(top_n)
    ranked_barplot(
        plot_df,
        x=score_col,
        y=label_col,
        save_path=save_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=(10, 6),
        dpi=dpi,
    )


def plot_metric_barplot(
    data,
    *,
    x,
    y,
    save_path,
    title,
    xlabel,
    ylabel,
    ylim=None,
    dpi=300,
):
    _ensure_parent(save_path)
    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=data,
        x=x,
        y=y,
        color="#7fa8c6",
        edgecolor="#d6e1ec",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelrotation=30)
    style_axis(ax, x_grid=False, y_grid=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_expression_heatmap(
    data,
    save_path,
    *,
    title,
    center=None,
    dpi=300,
):
    _ensure_parent(save_path)
    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(12, 9))
    values = data.to_numpy(dtype=float)
    if center is None:
        center = float(np.nanmean(values))
    sns.heatmap(
        data,
        cmap="RdBu_r",
        center=center,
        cbar_kws={"shrink": 0.75, "label": "Expression"},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_grouped_metric_barplot(
    data,
    *,
    x,
    y,
    hue,
    save_path,
    title,
    xlabel,
    ylabel,
    dpi=300,
):
    _ensure_parent(save_path)
    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=sns.blend_palette(["#4f7ea8", "#d88c60", "#78a878", "#a87ca8", "#c6a45f"], n_colors=max(data[hue].nunique(), 1)),
        edgecolor="#e6edf4",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(title=hue, frameon=True, loc="best")
    style_axis(ax, x_grid=False, y_grid=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
