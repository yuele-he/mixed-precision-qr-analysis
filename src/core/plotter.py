"""Plotting utilities for floating-point and QR error experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


DEFAULT_METHOD_ORDER = ("Householder", "WY", "Block QR")
DEFAULT_PRECISION_ORDER = ("float64", "float32", "mixed_fma32_64", "mixed_fma32_64_store32", "mixed32_64", "mixed32_64_store32")

METRIC_LABELS = {
    "reconstruction": "Reconstruction error",
    "orthogonality": "Orthogonality error",
    "least_squares": "Least-squares error",
}

PRECISION_LABELS = {
    "float64": "Float64",
    "float32": "Float32",
    "mixed32_64": "FP32 input + FP64 QR",
    "mixed32_64_store32": "FP32 input + FP64 QR, store FP32",
    "mixed_fma32_64": "Mixed-FMA, FP64 acc.",
    "mixed_fma32_64_store32": "Mixed-FMA, FP32 storage",
}

MARKERS = {
    "Householder": "o",
    "WY": "s",
    "Block QR": "^",
    "float64": "o",
    "float32": "s",
    "mixed_fma32_64": "^",
    "mixed_fma32_64_store32": "D",
    "mixed32_64": "P",
    "mixed32_64_store32": "X",
    "Mixed-FMA Householder QR": "P",
}


def _finish_figure(fig: plt.Figure, save_path: Optional[str | Path] = None) -> None:
    """Tighten, save if requested, and show the figure."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def fit_log_error_bound(sizes: Iterable[float], errors: Iterable[float]) -> tuple[float, float]:
    """Fit ``log(error) ≈ slope * log(size) + intercept``."""
    x = np.asarray(list(sizes), dtype=float)
    y = np.asarray(list(errors), dtype=float)
    slope, intercept, *_ = linregress(np.log(x), np.log(y + 1e-300))
    return float(slope), float(intercept)


def plot_error_heatmap(
    errors_matrix: np.ndarray,
    title: str = "Relative error heatmap",
    save_path: Optional[str | Path] = None,
    bins: int = 128,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Plot a heatmap for step-wise error distributions."""
    n_trials, n_steps = errors_matrix.shape
    flat_errors = errors_matrix.ravel()
    step_indices = np.tile(np.arange(n_steps), n_trials)

    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap = ax.hist2d(step_indices, flat_errors, bins=(bins, bins), cmin=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Relative error")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for exponent in range(5, int(np.log2(max(n_steps, 1))) + 1):
        ax.axvline(2 ** exponent, linestyle="dotted", linewidth=1)

    fig.colorbar(heatmap[3], ax=ax, label="Count")
    _finish_figure(fig, save_path)


def plot_error_distribution(
    errors_matrix: np.ndarray,
    title: str = "Relative error distribution",
    save_path: Optional[str | Path] = None,
    bins: int = 80,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Plot a simple horizontal histogram of error values."""
    flat_errors = errors_matrix.ravel()

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.hist(flat_errors, bins=bins, orientation="horizontal", alpha=0.75)

    ax.set_ylabel("Relative error")
    ax.set_xlabel("Count")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    _finish_figure(fig, save_path)


def plot_error_vs_size(
    sizes: Iterable[int],
    error_matrix: np.ndarray,
    dtype: type,
    label: Optional[str] = None,
    show_fit: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot error samples and fitted growth over problem size."""
    sizes = np.asarray(list(sizes), dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    repeats, _n_sizes = error_matrix.shape
    flat_errors = error_matrix.ravel()
    repeated_sizes = np.tile(sizes, repeats)

    label = label or np.dtype(dtype).name

    ax.scatter(repeated_sizes, flat_errors, alpha=0.25, s=18, label=f"{label} samples")

    max_errors = np.max(error_matrix, axis=0)
    ax.scatter(sizes, max_errors, s=24)

    if show_fit:
        slope, intercept = fit_log_error_bound(sizes, max_errors)
        fit_y = np.exp(slope * np.log(sizes) + intercept)
        ax.plot(sizes, fit_y, linestyle="--", label=f"{label} fit, k={slope:.2f}")

    eps_val = np.finfo(dtype).eps
    ax.axhline(eps_val, linestyle=":", linewidth=1, label=f"{label} eps ≈ {eps_val:.1e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vector length")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", linestyle="dotted", alpha=0.6)
    ax.legend(fontsize=8)

    return ax


def plot_qr_metrics_bar(results: dict[str, dict[str, float]], title: str = "QR factorization errors") -> None:
    """Plot reconstruction and orthogonality errors for a single QR test."""
    methods = list(results.keys())
    reconstruction = [results[m].get("reconstruction", results[m].get("residual")) for m in methods]
    orthogonality = [results[m]["orthogonality"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, reconstruction, width, label="Reconstruction")
    ax.bar(x + width / 2, orthogonality, width, label="Orthogonality")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_yscale("log")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()

    _finish_figure(fig)


def plot_qr_metrics_line(data: dict[str, dict[str, list[float]]], title: str = "QR errors vs size") -> None:
    """Legacy line plot for ``QRExperiment.run_sweep`` output."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, values in data.items():
        sizes = values["sizes"]
        ax.plot(sizes, values.get("reconstruction", values.get("residuals")), marker="o", label=f"{method} reconstruction")
        ax.plot(sizes, values["orthogonality"], marker="s", linestyle="--", label=f"{method} orthogonality")

    ax.set_yscale("log")
    ax.set_xlabel("Matrix width n")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    _finish_figure(fig)


def plot_qr_metric_for_ppt(
    df: pd.DataFrame,
    metric: str,
    precision: str = "float64",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show_samples: bool = False,
    show_fit: bool = False,
    method_order: Iterable[str] = DEFAULT_METHOD_ORDER,
) -> None:
    """Plot one QR metric for method comparison in a presentation."""
    if "precision" not in df.columns:
        raise ValueError("DataFrame must contain a 'precision' column. Use run_qr_method_experiments for this plot.")

    plot_df = df[df["precision"] == precision].copy()
    if plot_df.empty:
        raise ValueError(f"No data found for precision={precision!r}.")

    title = title or f"QR {METRIC_LABELS.get(metric, metric)} ({precision})"
    ylabel = ylabel or f"Relative {METRIC_LABELS.get(metric, metric).lower()}"

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)

    for method in method_order:
        sub = plot_df[plot_df["method"] == method].copy()
        if sub.empty:
            continue

        summary = sub.groupby("n")[metric].agg(["mean", "max"]).reset_index().sort_values("n")

        if show_samples:
            ax.scatter(sub["n"], sub[metric], alpha=0.15, s=14)

        ax.plot(
            summary["n"],
            summary["mean"],
            marker=MARKERS.get(method, "o"),
            linewidth=2,
            markersize=5,
            label=method,
        )

        if show_fit:
            slope, intercept = fit_log_error_bound(summary["n"], summary["max"])
            fit_y = np.exp(slope * np.log(summary["n"].to_numpy(dtype=float)) + intercept)
            ax.plot(summary["n"], fit_y, linestyle="--", linewidth=1.2, label=f"{method} fit, k={slope:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix width n  (m = 2n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    _finish_figure(fig, save_path)


def plot_precision_comparison_for_ppt(
    df: pd.DataFrame,
    metric: str,
    method: str = "Householder",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show_samples: bool = False,
    precision_order: Iterable[str] = DEFAULT_PRECISION_ORDER,
) -> None:
    """Plot how precision affects one QR method and one error metric."""
    required = {"method", "precision_mode", "n", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

    plot_df = df[df["method"] == method].copy()
    if plot_df.empty:
        raise ValueError(f"No data found for method={method!r}.")

    title = title or f"Effect of precision on {method} QR"
    ylabel = ylabel or f"Relative {METRIC_LABELS.get(metric, metric).lower()}"

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)

    for precision_mode in precision_order:
        sub = plot_df[plot_df["precision_mode"] == precision_mode].copy()
        if sub.empty:
            continue

        summary = sub.groupby("n")[metric].agg(["mean", "max"]).reset_index().sort_values("n")

        if show_samples:
            ax.scatter(sub["n"], sub[metric], alpha=0.15, s=14)

        ax.plot(
            summary["n"],
            summary["mean"],
            marker=MARKERS.get(precision_mode, "o"),
            linewidth=2,
            markersize=5,
            label=PRECISION_LABELS.get(precision_mode, precision_mode),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix width n  (m = 2n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    _finish_figure(fig, save_path)


def plot_qr_grid(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    show_fit: bool = True,
    method_order: Iterable[str] = DEFAULT_METHOD_ORDER,
) -> None:
    """Plot a compact 3 x 2 QR error grid for checking results."""
    if "precision" not in df.columns:
        raise ValueError("DataFrame must contain a 'precision' column.")

    metrics = ("reconstruction", "orthogonality", "least_squares")
    precisions = ("float32", "float64")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), dpi=160)

    for i, metric in enumerate(metrics):
        for j, precision in enumerate(precisions):
            ax = axes[i, j]
            plot_df = df[df["precision"] == precision]

            for method in method_order:
                sub = plot_df[plot_df["method"] == method]
                if sub.empty:
                    continue

                ax.scatter(sub["n"], sub[metric], alpha=0.20, s=14, label=method)

                if show_fit:
                    summary = sub.groupby("n")[metric].max().reset_index().sort_values("n")
                    slope, intercept = fit_log_error_bound(summary["n"], summary[metric])
                    fit_y = np.exp(slope * np.log(summary["n"].to_numpy(dtype=float)) + intercept)
                    ax.plot(summary["n"], fit_y, linestyle="--", linewidth=1.2, label=f"{method} fit, k={slope:.2f}")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)

            if i == 0:
                ax.set_title(f"{precision} QR")
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric].title())
            if i == 2:
                ax.set_xlabel("Matrix width n")
            if i == 0 and j == 1:
                ax.legend(fontsize=8)

    _finish_figure(fig, save_path)
