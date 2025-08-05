# src/core/plotter.py
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def plot_error_heatmap(
        errors_matrix: np.ndarray,
        title: str = "Relative Error Heatmap",
        save_path: str = None,
        bins: int = 128,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
):
    """
    绘制 step-wise 误差热力图。

    参数:
        errors_matrix (ndarray): (n_trials, n_steps) 误差矩阵。
        title (str): 图标题。
        save_path (str): 若提供路径则保存图像。
        bins (int): bin 数量。
    """
    n_trials, n_steps = errors_matrix.shape
    flat_errors = errors_matrix.flatten()
    step_indices = np.tile(np.arange(n_steps), n_trials)

    fig, ax = plt.subplots(figsize=(8, 5))

    hb = ax.hist2d(step_indices, flat_errors, bins=(bins, bins), cmap='coolwarm', cmin=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative Error")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add log2 reference lines
    for n in range(5, int(np.log2(n_steps)) + 1):
        ax.axvline(2 ** n, color='red', linestyle='dotted', linewidth=1)

    fig.colorbar(hb[3], ax=ax, label="Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_error_distribution(
        errors_matrix: np.ndarray,
        title: str = "Relative Error Distribution",
        save_path: str = None,
        bins: int = 80,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
):
    """
    绘制误差分布图（带 KDE 的直方图）。

    参数:
        errors_matrix (ndarray): (n_trials, n_steps) 误差矩阵。
        title (str): 图标题。
        save_path (str): 若提供路径则保存图像。
        bins (int): bin 数量。
    """
    flat_errors = errors_matrix.flatten()

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.histplot(y=flat_errors, bins=bins, kde=True, ax=ax, color='skyblue', orientation="horizontal")

    ax.set_ylabel("Relative Error")
    ax.set_xlabel("Density")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_error_vs_size(
        sizes,
        error_matrix,
        dtype,
        label=None,
        color="blue",
        show_fit=True,
        ax=None
):
    """
    Plot relative errors over problem sizes, with log-log scale.

    Parameters
    ----------
    sizes : list of int
        Problem sizes (e.g., vector lengths).
    error_matrix : ndarray (shape: [repeats, len(sizes)])
        Relative errors from experiments.
    dtype : np.dtype
        Precision type (e.g., np.float32).
    label : str
        Optional label for legend.
    color : str
        Plotting color.
    show_fit : bool
        Whether to plot fitted upper bound.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Creates new if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    repeats, n_sizes = error_matrix.shape
    flat_errors = error_matrix.flatten()
    sizes_repeated = np.tile(sizes, repeats)

    # Scatter plot
    ax.scatter(
        sizes_repeated,
        flat_errors,
        alpha=0.25,
        s=18,
        color=color,
        label=f"{label or dtype.__name__} samples"
    )

    # Max error line fit
    max_errors = np.max(error_matrix, axis=0)
    ax.scatter(sizes, max_errors, color='black')
    slope, intercept = fit_log_error_bound(sizes, max_errors)
    if show_fit:
        fit_y = np.exp(slope * np.log(sizes) + intercept)
        ax.plot(
            sizes,
            fit_y,
            linestyle='--',
            color=color,
            label=f"{label or dtype.__name__} fit (k={slope:.2f})"
        )

    # Epsilon line
    eps_val = np.finfo(dtype).eps
    ax.axhline(eps_val, color=color, linestyle=':', linewidth=1,
               label=f"{label or dtype.__name__} ε ≈ {eps_val:.1e}")

    # Log scale formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vector Length (log scale)")
    ax.set_ylabel("Relative Error (log scale)")
    ax.grid(True, which="both", linestyle="dotted", alpha=0.6)
    ax.legend()


def fit_log_error_bound(sizes, errors):
    """
    Fit a line: log(error) ≈ k * log(size) + c

    Parameters
    ----------
    sizes : list[int]
        Sizes of input (e.g., vector length)
    errors : list[float]
        Maximum or summary errors per size

    Returns
    -------
    slope : float
    intercept : float
    """
    log_n = np.log(sizes)
    log_e = np.log(errors)
    slope, intercept, *_ = linregress(log_n, log_e)
    return slope, intercept
