import matplotlib.pyplot as plt
import seaborn as sns


SOFT_BLUE_ANCHORS = [
    "#4f7ea8",
    "#5f8db5",
    "#729fc0",
    "#88b1cc",
    "#a2c3db",
    "#c3d8e9",
    "#e8f0f7",
]


def soft_blue_palette(n_colors):
    return sns.blend_palette(SOFT_BLUE_ANCHORS, n_colors=n_colors)


def apply_paper_theme():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "white",
            "grid.color": "#d9e6f2",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.edgecolor": "#9fb3c8",
            "axes.linewidth": 0.8,
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
            "axes.unicode_minus": False,
        },
    )


def style_axis(ax, *, x_grid=True, y_grid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if x_grid else "both", alpha=0.65)
    if not y_grid:
        ax.grid(False, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9fb3c8")
    ax.spines["bottom"].set_color("#9fb3c8")


def ranked_barplot(
    data,
    *,
    x,
    y,
    save_path,
    title,
    xlabel,
    ylabel,
    figsize=(10, 6),
    xlim=None,
    dpi=300,
):
    apply_paper_theme()
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = data.reset_index(drop=True).copy()
    order = plot_df[y].tolist()
    sns.barplot(
        data=plot_df,
        x=x,
        y=y,
        order=order,
        color="#7fa8c6",
        ax=ax,
        edgecolor="#d6e1ec",
        linewidth=0.6,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
