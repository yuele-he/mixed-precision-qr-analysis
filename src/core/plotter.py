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


def plot_qr_metrics_bar(results, title="QR Decomposition Errors"):
    """
    Plot residual and orthogonality errors of QR decomposition methods.

    Parameters
    ----------
    results : dict
        Dictionary in format:
        {
            "method1": {"residual": val1, "orthogonality": val2},
            "method2": {"residual": val3, "orthogonality": val4},
            ...
        }
    title : str
        Title of the bar chart.
    """
    methods = list(results.keys())
    residuals = [results[m]["residual"] for m in methods]
    orthos = [results[m]["orthogonality"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, residuals, width, label="Residual Norm")
    plt.bar(x + width / 2, orthos, width, label="Orthogonality Norm")
    plt.xticks(x, methods)
    plt.yscale("log")  # QR误差通常较小，建议对数坐标
    plt.ylabel("Error (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_qr_metrics_line(data, title="QR Errors vs Size"):
    """
    Optional: If QRExperiment is run for multiple sizes, plot line chart.

    Parameters
    ----------
    data : dict
        {
            "method1": {"sizes": [...], "residuals": [...], "orthogonality": [...]},
            ...
        }
    title : str
        Title of the line chart.
    """
    plt.figure(figsize=(8, 6))
    for method, vals in data.items():
        plt.plot(vals["sizes"], vals["residuals"], label=f"{method} residual", linestyle='-')
        plt.plot(vals["sizes"], vals["orthogonality"], label=f"{method} orthogonality", linestyle='--')
    plt.yscale("log")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Error (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_qr_metric_for_ppt(
    df,
    metric,
    precision="float64",
    title=None,
    ylabel=None,
    save_path=None,
    show_samples=True,
    show_fit=False,
):
    """
    Plot one QR error metric for PPT.

    Recommended metrics:
        "reconstruction"
        "orthogonality"
        "least_squares"
    """
    plt.style.use("default")

    plot_df = df[df["precision"] == precision].copy()

    if plot_df.empty:
        raise ValueError(f"No data found for precision={precision}")

    if title is None:
        title = f"QR {metric.replace('_', ' ').title()} Error ({precision})"

    if ylabel is None:
        ylabel = f"Relative {metric.replace('_', ' ')} error"

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)

    markers = {
        "Householder": "o",
        "WY": "s",
        "Block QR": "^",
    }

    method_order = ["Householder", "WY", "Block QR"]

    for method in method_order:
        sub = plot_df[plot_df["method"] == method].copy()

        if sub.empty:
            continue

        summary = (
            sub.groupby("n")[metric]
            .agg(["mean", "max", "std"])
            .reset_index()
            .sort_values("n")
        )

        if show_samples:
            ax.scatter(
                sub["n"],
                sub[metric],
                alpha=0.18,
                s=16,
            )

        ax.plot(
            summary["n"],
            summary["mean"],
            marker=markers.get(method, "o"),
            linewidth=2,
            markersize=5,
            label=method,
        )

        if show_fit:
            x = summary["n"].to_numpy(dtype=float)
            y = summary["max"].to_numpy(dtype=float)

            k, c, *_ = linregress(np.log(x), np.log(y + 1e-300))
            fit_y = np.exp(k * np.log(x) + c)

            ax.plot(
                x,
                fit_y,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"{method} fit, k={k:.2f}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Matrix width n  (m = 2n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_qr_grid(
    df,
    save_path=None,
    show_fit=True,
):
    """
    Plot a 3 x 2 grid similar to the dissertation figure.

    Rows:
        reconstruction, orthogonality, least_squares

    Columns:
        float32, float64
    """
    plt.style.use("default")

    metrics = ["reconstruction", "orthogonality", "least_squares"]
    metric_labels = {
        "reconstruction": "Reconstruction Error",
        "orthogonality": "Orthogonality Error",
        "least_squares": "Least-Squares Error",
    }

    precisions = ["float32", "float64"]
    method_order = ["Householder", "WY", "Block QR"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), dpi=160)

    for i, metric in enumerate(metrics):
        for j, precision in enumerate(precisions):
            ax = axes[i, j]

            plot_df = df[df["precision"] == precision].copy()

            for method in method_order:
                sub = plot_df[plot_df["method"] == method].copy()

                if sub.empty:
                    continue

                ax.scatter(
                    sub["n"],
                    sub[metric],
                    label=method,
                    alpha=0.22,
                    s=14,
                )

                if show_fit:
                    summary = (
                        sub.groupby("n")[metric]
                        .max()
                        .reset_index()
                        .sort_values("n")
                    )

                    x = summary["n"].to_numpy(dtype=float)
                    y = summary[metric].to_numpy(dtype=float)

                    k, c, *_ = linregress(np.log(x), np.log(y + 1e-300))
                    fit_y = np.exp(k * np.log(x) + c)

                    ax.plot(
                        x,
                        fit_y,
                        linestyle="--",
                        linewidth=1.2,
                        label=f"{method} fit, k={k:.2f}",
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")

            if i == 0:
                ax.set_title(f"{precision} QR")

            if j == 0:
                ax.set_ylabel(metric_labels[metric])

            if i == 2:
                ax.set_xlabel("Matrix width n")

            ax.grid(True, which="both", linestyle=":", alpha=0.45)

            if i == 0 and j == 1:
                ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()