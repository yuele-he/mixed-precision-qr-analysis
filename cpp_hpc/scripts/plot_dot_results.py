from __future__ import annotations

import html as html_lib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("CPP_HPC_CONFIG", str(PROJECT_ROOT / "cpp_hpc" / "config.json")))

PRECISION_TITLES = {
    "fp64": "FP64",
    "mixed_fp32_fp64": "Mixed FP32-product / FP64-accumulation",
}

DISTRIBUTION_TITLES = {
    "normal": "normal",
    "positive_uniform": "positive uniform",
    "alternating_sign": "alternating sign",
    "ill_scaled": "ill-scaled",
    "nearly_canceling": "nearly canceling",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def project_path(relative_path: str | Path) -> Path:
    return PROJECT_ROOT / relative_path


def get_results_dir(config: dict) -> Path:
    return project_path(config.get("results_dir", "cpp_hpc/results"))


def result_file(config: dict, key: str) -> Path:
    return get_results_dir(config) / config["files"][key]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def figure_file(config: dict, key: str) -> Path:
    """Return figure path. When `_figure_suffix` is set, insert it before .png."""
    filename = config["figures"][key]
    path = get_results_dir(config) / filename
    suffix = config.get("_figure_suffix", "")
    if suffix:
        path = path.with_name(f"{path.stem}_{safe_name(suffix)}{path.suffix}")
    return path


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def prepare_numeric_columns(runtime_df: pd.DataFrame, error_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime_df = runtime_df.copy()
    error_df = error_df.copy()

    if "threads" in runtime_df:
        runtime_df["threads"] = pd.to_numeric(runtime_df["threads"], errors="coerce")
    if "threads" in error_df:
        error_df["threads"] = pd.to_numeric(error_df["threads"], errors="coerce")

    for col in ["n", "trial", "repeat", "runtime", "result"]:
        if col in runtime_df:
            runtime_df[col] = pd.to_numeric(runtime_df[col], errors="coerce")

    for col in [
        "n", "trial", "result", "reference_fp64", "reference_same_precision",
        "abs_reference_fp64", "abs_reference_same_precision",
        "sum_abs_products", "dot_condition_number_fp64", "dot_condition_number_same_precision",
        "abs_error_vs_serial_fp64", "scaled_error_vs_sum_abs_products",
        "scaled_parallel_error_vs_sum_abs_products",
        "error_vs_serial_fp64", "mixed_error_vs_fp64", "parallel_error_vs_serial_same_precision",
    ]:
        if col in error_df:
            error_df[col] = pd.to_numeric(error_df[col], errors="coerce")

    return runtime_df, error_df


def get_plot_distributions(config: dict, runtime_df: pd.DataFrame, error_df: pd.DataFrame) -> list[str]:
    """Resolve which input distributions to plot."""
    if "plot_input_distributions" in config:
        requested = config["plot_input_distributions"]
    elif "plot_input_distribution" in config:
        requested = [config["plot_input_distribution"]]
    else:
        requested = sorted(set(runtime_df.get("input_distribution", pd.Series(dtype=str))).union(
            set(error_df.get("input_distribution", pd.Series(dtype=str)))
        ))

    available = sorted(set(runtime_df.get("input_distribution", pd.Series(dtype=str))).union(
        set(error_df.get("input_distribution", pd.Series(dtype=str)))
    ))
    if requested == "all":
        return available

    result = []
    for item in requested:
        if item in available and item not in result:
            result.append(item)
    if not result and available:
        print(f"Requested plot distributions {requested} not found; falling back to {available[:1]}.")
        result = available[:1]
    return result


def filter_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    distribution = config.get("plot_input_distribution")
    if distribution and "input_distribution" in df.columns:
        return df[df["input_distribution"] == distribution].copy()
    return df.copy()


def sequential_colors(precision: str, thread_values: list[int]) -> dict[int, tuple]:
    cmap = plt.cm.Blues if precision == "fp64" else plt.cm.Reds
    if len(thread_values) == 1:
        positions = [0.70]
    else:
        positions = np.linspace(0.35, 0.90, len(thread_values))
    return {thread: cmap(pos) for thread, pos in zip(thread_values, positions)}


def positive_finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[arr > 0]


def log_y_limits(values: Iterable[float], config: dict) -> tuple[float, float] | None:
    arr = positive_finite(values)
    if len(arr) == 0:
        return None
    ycfg = config.get("y_axis", {})
    pad = float(ycfg.get("pad_decades", 0.25))
    min_decades = float(ycfg.get("min_decades", 1.5))
    lower_percentile = float(ycfg.get("lower_percentile", 0.0))
    upper_percentile = float(ycfg.get("upper_percentile", 100.0))
    round_to_decades = bool(ycfg.get("round_to_decades", True))

    lo = float(np.percentile(arr, lower_percentile))
    hi = float(np.percentile(arr, upper_percentile))
    if lo <= 0 or not np.isfinite(lo):
        lo = float(np.min(arr))
    if hi <= 0 or not np.isfinite(hi):
        hi = float(np.max(arr))
    if lo == hi:
        lo /= 10.0
        hi *= 10.0

    log_lo = math.log10(lo) - pad
    log_hi = math.log10(hi) + pad
    if log_hi - log_lo < min_decades:
        center = 0.5 * (log_lo + log_hi)
        log_lo = center - 0.5 * min_decades
        log_hi = center + 0.5 * min_decades

    if round_to_decades:
        log_lo = math.floor(log_lo)
        log_hi = math.ceil(log_hi)

    return 10.0 ** log_lo, 10.0 ** log_hi


def set_log_y_axis(ax, values: Iterable[float], config: dict) -> None:
    ax.set_yscale("log")
    limits = log_y_limits(values, config)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.yaxis.set_minor_formatter(NullFormatter())


def apply_log_limits_to_axes(axes: Iterable, values: Iterable[float], config: dict) -> None:
    limits = log_y_limits(values, config)
    for ax in axes:
        ax.set_yscale("log")
        if limits is not None:
            ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
        ax.yaxis.set_minor_formatter(NullFormatter())


def pick_representative_n(values: Iterable[int], max_count: int = 4) -> list[int]:
    unique = sorted(pd.unique(pd.Series(values).dropna()).tolist())
    if len(unique) <= max_count:
        return unique
    idx = np.linspace(0, len(unique) - 1, max_count).round().astype(int)
    return [unique[i] for i in idx]


def quantile_summary(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(
            n_obs="size",
            min="min",
            q05=lambda s: s.quantile(0.05),
            q25=lambda s: s.quantile(0.25),
            median="median",
            q75=lambda s: s.quantile(0.75),
            q95=lambda s: s.quantile(0.95),
            max="max",
            mean="mean",
            std="std",
        )
        .reset_index()
        .sort_values(group_cols)
    )


def make_runtime_summary(runtime_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(runtime_df, ["input_distribution", "method", "precision", "threads", "n", "runtime"], "runtime_df")
    return (
        runtime_df.groupby(["input_distribution", "method", "precision", "threads", "n"], dropna=False)["runtime"]
        .agg(
            n_obs="size",
            best="min",
            median="median",
            mean="mean",
            std="std",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            q05=lambda s: s.quantile(0.05),
            q95=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values(["input_distribution", "precision", "method", "threads", "n"])
    )


def make_speedup_summary(runtime_summary: pd.DataFrame) -> pd.DataFrame:
    serial = (
        runtime_summary[runtime_summary["method"] == "serial"]
        [["input_distribution", "precision", "n", "best", "median"]]
        .rename(columns={"best": "serial_best", "median": "serial_median"})
        .drop_duplicates(subset=["input_distribution", "precision", "n"])
    )
    openmp = runtime_summary[runtime_summary["method"] == "openmp"].copy()
    speedup = openmp.merge(serial, on=["input_distribution", "precision", "n"], how="left")
    speedup["speedup_best"] = speedup["serial_best"] / speedup["best"]
    speedup["speedup_median"] = speedup["serial_median"] / speedup["median"]
    return speedup.sort_values(["input_distribution", "precision", "threads", "n"])


def make_error_summaries(error_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        error_df,
        ["input_distribution", "method", "precision", "threads", "n", "mixed_error_vs_fp64",
         "parallel_error_vs_serial_same_precision"],
        "error_df",
    )

    mixed_serial = error_df[
        (error_df["method"] == "serial") &
        (error_df["precision"] == "mixed_fp32_fp64")
        ].copy()
    mixed_summary = quantile_summary(mixed_serial, ["input_distribution", "n"], "mixed_error_vs_fp64")

    parallel = error_df[error_df["method"] == "openmp"].copy()
    parallel_summary = quantile_summary(
        parallel,
        ["input_distribution", "precision", "threads", "n"],
        "parallel_error_vs_serial_same_precision",
    )

    mixed_summary.insert(0, "metric", "mixed_error_vs_fp64")
    parallel_summary.insert(0, "metric", "parallel_error_vs_serial_same_precision")
    return mixed_summary, parallel_summary


def make_condition_summary(error_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise dot-product conditioning using one row per trial.

    The condition number is
        kappa_dot = sum_i |x_i y_i| / |x^T y|.

    We use serial FP64 rows because the condition number is a property of the
    input vectors and the FP64 reference, not of method or thread count.
    """
    require_columns(
        error_df,
        ["input_distribution", "method", "precision", "n", "trial",
         "sum_abs_products", "dot_condition_number_fp64"],
        "error_df",
    )
    condition_rows = error_df[
        (error_df["method"] == "serial") &
        (error_df["precision"] == "fp64")
        ].copy()
    condition_rows = condition_rows[
        condition_rows["dot_condition_number_fp64"].notna() &
        np.isfinite(condition_rows["dot_condition_number_fp64"]) &
        (condition_rows["dot_condition_number_fp64"] > 0.0)
        ].copy()
    summary = quantile_summary(
        condition_rows,
        ["input_distribution", "n"],
        "dot_condition_number_fp64",
    )
    summary.insert(0, "metric", "dot_condition_number_fp64")
    return summary


def distribution_title(config: dict) -> str:
    dist = config.get("plot_input_distribution", "")
    return DISTRIBUTION_TITLES.get(dist, dist)


def plot_runtime_benchmark(runtime_df: pd.DataFrame, runtime_summary: pd.DataFrame, config: dict,
                           summary_stat: str = "median") -> None:
    runtime_df = filter_distribution(runtime_df, config)
    runtime_summary = filter_distribution(runtime_summary, config)
    precisions = [p for p in ["fp64", "mixed_fp32_fp64"] if p in set(runtime_df["precision"])]

    fig, axes = plt.subplots(1, len(precisions), figsize=(7.0 * len(precisions), 5.0), sharey=True,
                             constrained_layout=True)
    if len(precisions) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    global_y_values = runtime_df["runtime"].dropna().tolist()

    for ax, precision in zip(axes, precisions):
        raw = runtime_df[runtime_df["precision"] == precision].copy()
        summary = runtime_summary[runtime_summary["precision"] == precision].copy()
        y_values = []

        serial_raw = raw[raw["method"] == "serial"].copy()
        serial_sum = summary[summary["method"] == "serial"].sort_values("n").copy()
        if not serial_raw.empty:
            jitter = np.exp(rng.normal(0.0, 0.02, len(serial_raw)))
            ax.scatter(serial_raw["n"] * jitter, serial_raw["runtime"], alpha=0.13, s=22, marker="s", color="0.45",
                       label="serial repeats")
            y_values.extend(serial_raw["runtime"].tolist())
        if not serial_sum.empty:
            ax.plot(serial_sum["n"], serial_sum[summary_stat], marker="s", linewidth=2.3, color="0.12",
                    label=f"serial {summary_stat}")
            y_values.extend(serial_sum[summary_stat].tolist())

        openmp_raw = raw[raw["method"] == "openmp"].copy()
        openmp_sum = summary[summary["method"] == "openmp"].copy()
        thread_values = sorted(openmp_raw["threads"].dropna().astype(int).unique())
        colors = sequential_colors(precision, thread_values)
        for thread in thread_values:
            raw_t = openmp_raw[openmp_raw["threads"].astype(int) == thread].copy()
            sum_t = openmp_sum[openmp_sum["threads"].astype(int) == thread].sort_values("n").copy()
            if not raw_t.empty:
                jitter = np.exp(rng.normal(0.0, 0.02, len(raw_t)))
                ax.scatter(raw_t["n"] * jitter, raw_t["runtime"], alpha=0.12, s=17, color=colors[thread])
                y_values.extend(raw_t["runtime"].tolist())
            if not sum_t.empty:
                ax.plot(sum_t["n"], sum_t[summary_stat], marker="o", linewidth=2.0, color=colors[thread],
                        label=f"openmp t{thread}")
                y_values.extend(sum_t[summary_stat].tolist())

        ax.set_xscale("log")
        ax.set_xlabel("Vector size n")
        ax.set_title(f"{PRECISION_TITLES.get(precision, precision)} runtime")
        ax.grid(True, which="both", alpha=0.25)

    apply_log_limits_to_axes(axes, global_y_values, config)
    axes[0].set_ylabel("Runtime (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Series", loc="center left", bbox_to_anchor=(1.01, 0.5))
    dist = distribution_title(config)
    fig.suptitle(f"Runtime benchmark ({dist})", fontsize=18)
    savefig(figure_file(config, "runtime_benchmark"))
    plt.close(fig)


def plot_speedup(speedup_df: pd.DataFrame, config: dict, stat: str = "speedup_median") -> None:
    speedup_df = filter_distribution(speedup_df, config)
    precisions = [p for p in ["fp64", "mixed_fp32_fp64"] if p in set(speedup_df["precision"])]
    fig, axes = plt.subplots(1, len(precisions), figsize=(6.6 * len(precisions), 4.6), sharey=True,
                             constrained_layout=True)
    if len(precisions) == 1:
        axes = [axes]

    all_values = []
    for ax, precision in zip(axes, precisions):
        sub = speedup_df[speedup_df["precision"] == precision].copy()
        thread_values = sorted(sub["threads"].dropna().astype(int).unique())
        colors = sequential_colors(precision, thread_values)
        for thread in thread_values:
            st = sub[sub["threads"].astype(int) == thread].sort_values("n")
            ax.plot(st["n"], st[stat], marker="o", linewidth=2.0, color=colors[thread], label=f"t{thread}")
            all_values.extend(st[stat].dropna().tolist())
        ax.axhline(1.0, linestyle="--", linewidth=1.1, color="0.45")
        ax.set_xscale("log")
        ax.set_xlabel("Vector size n")
        ax.set_title(f"{PRECISION_TITLES.get(precision, precision)} speedup")
        ax.grid(True, which="both", alpha=0.25)
    if all_values:
        hi = max(max(all_values) * 1.12, 1.25)
        lo = min(0.0, min(all_values) * 0.95)
        for ax in axes:
            ax.set_ylim(lo, hi)
    axes[0].set_ylabel("Speedup vs serial same precision")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Threads", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"OpenMP speedup ({distribution_title(config)})", fontsize=18)
    savefig(figure_file(config, "speedup"))
    plt.close(fig)


def plot_parallel_error_median(error_df: pd.DataFrame, config: dict) -> None:
    metric = "parallel_error_vs_serial_same_precision"
    require_columns(error_df, ["method", "precision", "threads", "n", "input_distribution", metric], "error_df")
    df = filter_distribution(error_df, config)
    df = df[(df["method"] == "openmp") & df[metric].notna()].copy()
    summary = quantile_summary(df, ["input_distribution", "precision", "threads", "n"], metric)
    summary.insert(0, "metric", metric)

    precisions = [p for p in ["fp64", "mixed_fp32_fp64"] if p in set(summary["precision"])]
    fig, axes = plt.subplots(1, len(precisions), figsize=(6.6 * len(precisions), 4.8), sharey=True,
                             constrained_layout=True)
    if len(precisions) == 1:
        axes = [axes]

    positive_medians = positive_finite(summary["median"].dropna())
    zero_floor = float(positive_medians.min() / 10.0) if len(positive_medians) else 1e-18
    summary["median_plot"] = summary["median"].where(summary["median"] > 0, zero_floor)
    global_y_values = summary["median_plot"].dropna().tolist()

    for ax, precision in zip(axes, precisions):
        sub = summary[summary["precision"] == precision].copy()
        thread_values = sorted(sub["threads"].dropna().astype(int).unique())
        colors = sequential_colors(precision, thread_values)
        y_values = []
        for thread in thread_values:
            st = sub[sub["threads"].astype(int) == thread].sort_values("n")
            ax.plot(st["n"], st["median_plot"], marker="o", linewidth=2.1, color=colors[thread], label=f"t{thread}")
            y_values.extend(st["median_plot"].tolist())
        zero_medians = int((sub["median"] == 0).sum())
        total_medians = len(sub)
        if zero_medians > 0:
            ax.text(
                0.03, 0.05,
                f"zero medians: {zero_medians}/{total_medians}",
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )
        ax.set_xscale("log")
        ax.set_xlabel("Vector size n")
        ax.set_title(f"{PRECISION_TITLES.get(precision, precision)}")
        ax.grid(True, which="both", alpha=0.25)

    apply_log_limits_to_axes(axes, global_y_values, config)
    axes[0].set_ylabel("Median relative difference vs serial same precision")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Threads", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"Parallel reduction difference: median trend ({distribution_title(config)})", fontsize=18)
    savefig(figure_file(config, "parallel_error_median"))
    plt.close(fig)


def plot_mixed_error_trend(mixed_summary: pd.DataFrame, config: dict) -> None:
    summary = filter_distribution(mixed_summary, config).sort_values("n")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if summary.empty:
        print("No mixed error summary to plot.")
        return
    x = summary["n"].to_numpy(dtype=float)
    median = summary["median"].to_numpy(dtype=float)
    q25 = summary["q25"].to_numpy(dtype=float)
    q75 = summary["q75"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color="0.80", alpha=0.5, label="q25-q75 across trials")
    ax.plot(x, median, marker="o", linewidth=2.4, color="0.10", label="median across trials")
    ax.set_xscale("log")
    set_log_y_axis(ax, list(q25) + list(q75) + list(median), config)
    ax.set_xlabel("Vector size n")
    ax.set_ylabel("Relative error vs serial FP64")
    ax.set_title(f"Mixed-precision arithmetic error: quantile trend ({distribution_title(config)})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    savefig(figure_file(config, "mixed_error_trend"))
    plt.close(fig)


def box_scatter(ax, groups: list[np.ndarray], labels: list[str], colors: list[tuple], yscale_log: bool = True) -> list[
    float]:
    clean_groups: list[np.ndarray] = []
    clean_labels: list[str] = []
    clean_colors: list[tuple] = []
    all_values: list[float] = []
    finite_total = 0
    zero_total = 0

    for g, lab, col in zip(groups, labels, colors):
        arr = np.asarray(g, dtype=float)
        arr = arr[np.isfinite(arr)]
        finite_total += len(arr)
        zero_total += int((arr == 0).sum())

        if yscale_log:
            arr = arr[arr > 0]

        if len(arr) > 0:
            clean_groups.append(arr)
            clean_labels.append(lab)
            clean_colors.append(col)
            all_values.extend(arr.tolist())

    if not clean_groups:
        if finite_total > 0 and zero_total == finite_total:
            message = "All zero"
        elif finite_total > 0:
            message = "No positive values"
        else:
            message = "No data"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        return all_values

    positions = np.arange(1, len(clean_groups) + 1)
    bp = ax.boxplot(
        clean_groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for pos, arr, col in zip(positions, clean_groups, clean_colors):
        jitter = rng.normal(0.0, 0.075, len(arr))
        ax.scatter(np.full(len(arr), pos) + jitter, arr, alpha=0.36, s=24, color=col, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(clean_labels)
    if yscale_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.22)
    return all_values


def plot_parallel_error_distribution(error_df: pd.DataFrame, config: dict) -> None:
    df = filter_distribution(error_df, config)
    df = df[df["method"] == "openmp"].copy()
    n_values = pick_representative_n(df["n"], max_count=4)
    precisions = [p for p in ["fp64", "mixed_fp32_fp64"] if p in set(df["precision"])]
    fig, axes = plt.subplots(len(precisions), len(n_values), figsize=(4.1 * len(n_values), 3.7 * len(precisions)),
                             sharey=True, constrained_layout=True)
    axes = np.array(axes, dtype=object).reshape(len(precisions), len(n_values))

    global_values = []
    all_axes = []
    for i, precision in enumerate(precisions):
        sub_precision = df[df["precision"] == precision].copy()
        thread_values = sorted(sub_precision["threads"].dropna().astype(int).unique())
        colors = sequential_colors(precision, thread_values)
        for j, n_val in enumerate(n_values):
            ax = axes[i, j]
            all_axes.append(ax)
            sub_n = sub_precision[sub_precision["n"] == n_val].copy()
            groups = [sub_n[sub_n["threads"].astype(int) == t]["parallel_error_vs_serial_same_precision"].to_numpy() for
                      t in thread_values]
            labels = [str(t) for t in thread_values]
            values = box_scatter(ax, groups, labels, [colors[t] for t in thread_values], yscale_log=True)
            global_values.extend(values)
            ax.set_title(f"{PRECISION_TITLES.get(precision, precision)}\nn={int(n_val):,}")
            ax.set_xlabel("Threads")
            if j == 0:
                ax.set_ylabel("Relative difference")
    apply_log_limits_to_axes(all_axes, global_values, config)
    fig.suptitle(f"Parallel reduction difference: distribution at representative n ({distribution_title(config)})",
                 fontsize=16)
    savefig(figure_file(config, "parallel_error_distribution"))
    plt.close(fig)


def plot_mixed_error_distribution(error_df: pd.DataFrame, config: dict) -> None:
    """Single-panel distribution plot for mixed arithmetic error.

    x-axis: representative vector sizes n
    y-axis: mixed_error_vs_fp64
    Each n gets one boxplot plus jittered trial points.
    """
    metric = "mixed_error_vs_fp64"
    df = filter_distribution(error_df, config)
    df = df[(df["method"] == "serial") & (df["precision"] == "mixed_fp32_fp64")].copy()
    df = df[df[metric].notna() & (df[metric] > 0.0)].copy()
    if df.empty:
        print("No mixed-error data to plot.")
        return

    n_values = pick_representative_n(df["n"], max_count=4)
    n_values = sorted(n_values)
    groups = [df[df["n"] == n][metric].to_numpy() for n in n_values]
    labels = [f"{int(n):,}" for n in n_values]
    colors = [plt.cm.Greys(0.62)] * len(groups)

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    all_values = box_scatter(ax, groups, labels, colors, yscale_log=True)
    apply_log_limits_to_axes([ax], all_values, config)

    ax.set_xlabel("Vector size n")
    ax.set_ylabel("Relative error vs serial FP64")
    ax.set_title(f"Mixed-precision arithmetic error: distribution by n ({distribution_title(config)})")
    ax.tick_params(axis="x", rotation=0)

    savefig(figure_file(config, "mixed_error_distribution"))
    plt.close(fig)


def plot_condition_number_trend(condition_summary: pd.DataFrame, config: dict) -> None:
    """Plot median dot-product condition number across vector sizes."""
    if condition_summary.empty:
        print("No condition-number summary to plot.")
        return
    summary = filter_distribution(condition_summary, config).sort_values("n")
    if summary.empty:
        print("No condition-number data for this input distribution.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    x = summary["n"].to_numpy(dtype=float)
    median = summary["median"].to_numpy(dtype=float)
    q25 = summary["q25"].to_numpy(dtype=float)
    q75 = summary["q75"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color="0.80", alpha=0.50, label="q25-q75 across trials")
    ax.plot(x, median, marker="o", linewidth=2.4, color="0.10", label="median across trials")
    ax.set_xscale("log")
    set_log_y_axis(ax, list(q25) + list(q75) + list(median), config)
    ax.set_xlabel("Vector size n")
    ax.set_ylabel(r"Dot-product condition number $\kappa_{dot}$")
    ax.set_title(f"Dot-product condition number: quantile trend ({distribution_title(config)})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    savefig(figure_file(config, "condition_number_trend"))
    plt.close(fig)


def plot_condition_number_distribution(error_df: pd.DataFrame, config: dict) -> None:
    """Single-panel boxplot of kappa_dot by representative n."""
    metric = "dot_condition_number_fp64"
    df = filter_distribution(error_df, config)
    df = df[(df["method"] == "serial") & (df["precision"] == "fp64")].copy()
    df = df[df[metric].notna() & np.isfinite(df[metric]) & (df[metric] > 0.0)].copy()
    if df.empty:
        print("No condition-number data to plot.")
        return

    n_values = pick_representative_n(df["n"], max_count=4)
    n_values = sorted(n_values)
    groups = [df[df["n"] == n][metric].to_numpy() for n in n_values]
    labels = [f"{int(n):,}" for n in n_values]
    colors = [plt.cm.Greys(0.62)] * len(groups)

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    all_values = box_scatter(ax, groups, labels, colors, yscale_log=True)
    apply_log_limits_to_axes([ax], all_values, config)

    ax.set_xlabel("Vector size n")
    ax.set_ylabel(r"Dot-product condition number $\kappa_{dot}$")
    ax.set_title(f"Dot-product condition number: distribution by n ({distribution_title(config)})")
    ax.tick_params(axis="x", rotation=0)

    savefig(figure_file(config, "condition_number_distribution"))
    plt.close(fig)


def plot_runtime_distribution(runtime_df: pd.DataFrame, config: dict) -> None:
    df = filter_distribution(runtime_df, config)
    n_values = pick_representative_n(df["n"], max_count=3)
    precisions = [p for p in ["fp64", "mixed_fp32_fp64"] if p in set(df["precision"])]
    fig, axes = plt.subplots(len(precisions), len(n_values), figsize=(4.7 * len(n_values), 3.8 * len(precisions)),
                             sharey=True, constrained_layout=True)
    axes = np.array(axes, dtype=object).reshape(len(precisions), len(n_values))
    global_values = []
    all_axes = []
    for i, precision in enumerate(precisions):
        sub_precision = df[df["precision"] == precision].copy()
        thread_values = sorted(
            sub_precision[sub_precision["method"] == "openmp"]["threads"].dropna().astype(int).unique())
        colors = sequential_colors(precision, thread_values)
        for j, n_val in enumerate(n_values):
            ax = axes[i, j]
            all_axes.append(ax)
            sub_n = sub_precision[sub_precision["n"] == n_val].copy()
            groups = []
            labels = []
            group_colors = []
            serial = sub_n[sub_n["method"] == "serial"]["runtime"].to_numpy()
            if len(serial):
                groups.append(serial)
                labels.append("serial")
                group_colors.append((0.3, 0.3, 0.3, 1.0))
            openmp_n = sub_n[sub_n["method"] == "openmp"].copy()
            for t in thread_values:
                groups.append(openmp_n[openmp_n["threads"].astype(int) == t]["runtime"].to_numpy())
                labels.append(f"t{t}")
                group_colors.append(colors[t])
            values = box_scatter(ax, groups, labels, group_colors, yscale_log=True)
            global_values.extend(values)
            ax.set_title(f"{PRECISION_TITLES.get(precision, precision)}\nn={int(n_val):,}")
            ax.set_xlabel("Series")
            ax.tick_params(axis="x", rotation=45)
            if j == 0:
                ax.set_ylabel("Runtime (s)")
    apply_log_limits_to_axes(all_axes, global_values, config)
    fig.suptitle(f"Runtime distribution at representative n ({distribution_title(config)})", fontsize=16)
    savefig(figure_file(config, "runtime_distribution"))
    plt.close(fig)


def print_sanity_report(runtime_df: pd.DataFrame, error_df: pd.DataFrame) -> None:
    print("\n=== Plot-data sanity check ===")
    if "input_distribution" in error_df.columns:
        print("Error input distributions:", sorted(error_df["input_distribution"].dropna().unique().tolist()))
    if "input_distribution" in runtime_df.columns:
        print("Runtime input distributions:", sorted(runtime_df["input_distribution"].dropna().unique().tolist()))
    if "parallel_error_vs_serial_same_precision" in error_df.columns:
        metric = "parallel_error_vs_serial_same_precision"
        openmp = error_df[error_df["method"] == "openmp"].copy()
        print("Parallel error median by input_distribution and precision:")
        print(openmp.groupby(["input_distribution", "precision"])[metric].median())
    if "mixed_error_vs_fp64" in error_df.columns:
        mixed = error_df[(error_df["method"] == "serial") & (error_df["precision"] == "mixed_fp32_fp64")].copy()
        print("Mixed arithmetic error median by input_distribution:")
        print(mixed.groupby("input_distribution")["mixed_error_vs_fp64"].median())
    if "dot_condition_number_fp64" in error_df.columns:
        cond = error_df[(error_df["method"] == "serial") & (error_df["precision"] == "fp64")].copy()
        cond = cond[np.isfinite(cond["dot_condition_number_fp64"])]
        print("Dot-condition-number median by input_distribution:")
        print(cond.groupby("input_distribution")["dot_condition_number_fp64"].median())
    if "runtime" in runtime_df.columns:
        print("Runtime min/max:", runtime_df["runtime"].min(), runtime_df["runtime"].max())
    print("=== End sanity check ===\n")


def main() -> None:
    config = load_config()
    results_dir = get_results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_df = pd.read_csv(result_file(config, "runtime_results"))
    error_df = pd.read_csv(result_file(config, "error_results"))
    runtime_df, error_df = prepare_numeric_columns(runtime_df, error_df)
    print_sanity_report(runtime_df, error_df)

    runtime_summary = make_runtime_summary(runtime_df)
    speedup_summary = make_speedup_summary(runtime_summary)
    mixed_summary, parallel_summary = make_error_summaries(error_df)
    condition_summary = make_condition_summary(error_df)

    runtime_summary.to_csv(result_file(config, "runtime_summary"), index=False)
    speedup_summary.to_csv(result_file(config, "speedup_summary"), index=False)
    mixed_summary.to_csv(result_file(config, "mixed_error_summary"), index=False)
    parallel_summary.to_csv(result_file(config, "parallel_error_summary"), index=False)
    condition_summary.to_csv(result_file(config, "condition_number_summary"), index=False)
    print(f"Saved summary CSVs to {results_dir}")

    plot_distributions = get_plot_distributions(config, runtime_df, error_df)
    use_suffix = len(plot_distributions) > 1
    print(f"Plotting input distributions: {plot_distributions}")
    for distribution in plot_distributions:
        dist_config = dict(config)
        dist_config["plot_input_distribution"] = distribution
        if use_suffix:
            dist_config["_figure_suffix"] = distribution
        print(f"\n--- Plotting distribution: {distribution} ---")
        plot_runtime_benchmark(runtime_df, runtime_summary, dist_config,
                               summary_stat=config.get("runtime_summary_stat", "median"))
        plot_speedup(speedup_summary, dist_config, stat=config.get("speedup_stat", "speedup_median"))
        plot_parallel_error_median(error_df, dist_config)
        plot_parallel_error_distribution(error_df, dist_config)
        plot_mixed_error_trend(mixed_summary, dist_config)
        plot_mixed_error_distribution(error_df, dist_config)
        plot_condition_number_trend(condition_summary, dist_config)
        plot_condition_number_distribution(error_df, dist_config)
        plot_runtime_distribution(runtime_df, dist_config)


# === scaled-error plots injected ===
def _scaled_project_path(path_str):
    from pathlib import Path

    project_root = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def _scaled_load_config():
    import json
    import os
    from pathlib import Path

    project_root = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
    config_path = globals().get(
        "CONFIG_PATH",
        Path(os.environ.get("CPP_HPC_CONFIG", str(project_root / "cpp_hpc" / "config.json"))),
    )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scaled_results_dir(config):
    if "results_dir" in config:
        return _scaled_project_path(config["results_dir"])

    # fallback used by this project
    return _scaled_project_path("cpp_hpc/results")


def _scaled_input_distributions(config, error_df):
    if "plot_input_distributions" in config:
        return config["plot_input_distributions"]

    if "input_distributions" in config:
        return config["input_distributions"]

    if "experiments" in config and isinstance(config["experiments"], dict):
        if "input_distributions" in config["experiments"]:
            return config["experiments"]["input_distributions"]

    return sorted(error_df["input_distribution"].dropna().unique())


def _scaled_pick_representative_n(values, k=4):
    import numpy as np
    import pandas as pd

    values = sorted(int(v) for v in pd.unique(values))
    if len(values) <= k:
        return values

    idx = np.linspace(0, len(values) - 1, k).round().astype(int)
    return [values[i] for i in idx]


def _scaled_positive_log_ylim(values, pad_decades=0.3):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) == 0:
        return 1e-18, 1e-12

    log_min = np.floor(np.log10(values.min()) - pad_decades)
    log_max = np.ceil(np.log10(values.max()) + pad_decades)

    if log_min == log_max:
        log_min -= 1
        log_max += 1

    return 10.0 ** log_min, 10.0 ** log_max


def _scaled_add_plot_value_with_zero_floor(df, metric):
    """
    log-scale cannot plot zero.
    Keep original metric, and create metric_plot:
    - positive values are unchanged
    - zero values are placed on a small floor
    """
    import numpy as np

    out = df.copy()
    out = out[np.isfinite(out[metric]) & (out[metric] >= 0)].copy()

    positive = out.loc[out[metric] > 0, metric]
    if len(positive) > 0:
        floor = positive.min() / 10.0
    else:
        floor = 1e-18

    out[metric + "_plot"] = out[metric].where(out[metric] > 0, floor)
    out["is_zero_metric"] = out[metric] == 0
    return out


def _scaled_quantile_summary(df, group_cols, metric):
    return (
        df.groupby(group_cols, as_index=False)[metric]
        .agg(
            n_obs="size",
            q25=lambda s: s.quantile(0.25),
            median="median",
            q75=lambda s: s.quantile(0.75),
            q05=lambda s: s.quantile(0.05),
            q95=lambda s: s.quantile(0.95),
            min="min",
            max="max",
        )
        .sort_values(group_cols)
    )


def _scaled_savefig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_scaled_mixed_error_quantile(error_df, input_distribution, results_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    metric = "scaled_error_vs_sum_abs_products"

    df = error_df[
        (error_df["input_distribution"] == input_distribution)
        & (error_df["method"] == "serial")
        & (error_df["precision"] == "mixed_fp32_fp64")
        & np.isfinite(error_df[metric])
        & (error_df[metric] >= 0)
        ].copy()

    if df.empty:
        print(f"No scaled mixed error data for {input_distribution}")
        return

    dfp = _scaled_add_plot_value_with_zero_floor(df, metric)
    plot_metric = metric + "_plot"

    raw_summary = _scaled_quantile_summary(df, ["n"], metric)
    raw_summary.to_csv(
        results_dir / f"scaled_mixed_error_summary_{input_distribution}.csv",
        index=False,
    )
    summary = _scaled_quantile_summary(dfp, ["n"], plot_metric)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    ax.fill_between(
        summary["n"],
        summary["q25"],
        summary["q75"],
        alpha=0.22,
        color="0.65",
        label="q25-q75 across trials",
    )

    ax.plot(
        summary["n"],
        summary["median"],
        marker="o",
        linewidth=2.5,
        color="0.10",
        label="median across trials",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*_scaled_positive_log_ylim(dfp[plot_metric]))

    ax.set_xlabel("Vector size n")
    ax.set_ylabel(r"Scaled mixed error: $|s_{mixed}-s_{FP64}| / \sum_i |x_i y_i|$")
    ax.set_title(f"Scaled mixed-precision arithmetic error ({input_distribution})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    _scaled_savefig(
        fig,
        results_dir / f"scaled_mixed_error_quantile_trend_{input_distribution}.png",
    )


def _plot_scaled_mixed_error_distribution(error_df, input_distribution, results_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    metric = "scaled_error_vs_sum_abs_products"

    df = error_df[
        (error_df["input_distribution"] == input_distribution)
        & (error_df["method"] == "serial")
        & (error_df["precision"] == "mixed_fp32_fp64")
        & np.isfinite(error_df[metric])
        & (error_df[metric] >= 0)
        ].copy()

    if df.empty:
        print(f"No scaled mixed error distribution data for {input_distribution}")
        return

    n_values = _scaled_pick_representative_n(df["n"].unique(), k=4)
    df = df[df["n"].isin(n_values)].copy()

    dfp = _scaled_add_plot_value_with_zero_floor(df, metric)
    plot_metric = metric + "_plot"

    n_values = sorted(n_values)
    data_by_n = [
        dfp.loc[dfp["n"] == n, plot_metric].to_numpy()
        for n in n_values
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    positions = np.arange(len(n_values))

    ax.boxplot(
        data_by_n,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "0.92", "edgecolor": "0.2"},
        medianprops={"color": "0.05", "linewidth": 1.6},
        whiskerprops={"color": "0.25"},
        capprops={"color": "0.25"},
    )

    rng = np.random.default_rng(0)
    ymin, ymax = _scaled_positive_log_ylim(dfp[plot_metric])

    for i, n in enumerate(n_values):
        sub = dfp[dfp["n"] == n]
        values = sub[plot_metric].to_numpy()
        x = i + rng.normal(0.0, 0.06, size=len(values))

        ax.scatter(
            x,
            values,
            s=28,
            alpha=0.35,
            color="0.45",
            edgecolors="none",
        )

        zero_count = int((sub[metric] == 0).sum())
        total = len(sub)

        if zero_count > 0:
            ax.text(
                i,
                ymin * 1.4,
                f"zero {zero_count}/{total}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n:,}" for n in n_values])

    ax.set_xlabel("Vector size n")
    ax.set_ylabel(r"Scaled mixed error: $|s_{mixed}-s_{FP64}| / \sum_i |x_i y_i|$")
    ax.set_title(f"Scaled mixed-precision arithmetic error: distribution by n ({input_distribution})")
    ax.grid(True, which="both", axis="y", alpha=0.25)

    _scaled_savefig(
        fig,
        results_dir / f"scaled_mixed_error_distribution_{input_distribution}.png",
    )


def _plot_scaled_parallel_error_median(error_df, input_distribution, results_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    metric = "scaled_parallel_error_vs_sum_abs_products"

    df = error_df[
        (error_df["input_distribution"] == input_distribution)
        & (error_df["method"] == "openmp")
        & np.isfinite(error_df[metric])
        & (error_df[metric] >= 0)
        ].copy()

    if df.empty:
        print(f"No scaled parallel error data for {input_distribution}")
        return

    dfp = _scaled_add_plot_value_with_zero_floor(df, metric)
    plot_metric = metric + "_plot"

    raw_summary = _scaled_quantile_summary(
        df,
        ["precision", "threads", "n"],
        metric,
    )
    raw_summary.to_csv(
        results_dir / f"scaled_parallel_error_summary_{input_distribution}.csv",
        index=False,
    )

    summary = _scaled_quantile_summary(
        dfp,
        ["precision", "threads", "n"],
        plot_metric,
    )

    precisions = ["fp64", "mixed_fp32_fp64"]
    titles = {
        "fp64": "FP64",
        "mixed_fp32_fp64": "Mixed FP32-product / FP64-accumulation",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        sharey=True,
        constrained_layout=True,
    )

    global_ylim = _scaled_positive_log_ylim(dfp[plot_metric])

    for ax, precision in zip(axes, precisions):
        sub = summary[summary["precision"] == precision].copy()
        raw_sub = dfp[dfp["precision"] == precision].copy()

        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(titles[precision])
            continue

        thread_values = sorted(sub["threads"].dropna().astype(int).unique())
        cmap = plt.cm.Blues if precision == "fp64" else plt.cm.Reds
        colors = np.linspace(0.35, 0.90, len(thread_values))

        for t, c in zip(thread_values, colors):
            st = sub[sub["threads"].astype(int) == t].sort_values("n")

            ax.plot(
                st["n"],
                st["median"],
                marker="o",
                linewidth=2.0,
                color=cmap(c),
                label=f"t{t}",
            )

        zero_count = int((raw_sub[metric] == 0).sum())
        total = len(raw_sub)

        if zero_count > 0:
            ax.text(
                0.03,
                0.05,
                f"zero values: {zero_count}/{total}",
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*global_ylim)
        ax.set_xlabel("Vector size n")
        ax.set_title(titles[precision])
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"Scaled parallel error: $|s_{omp}-s_{serial}| / \sum_i |x_i y_i|$")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Threads",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    fig.suptitle(f"Scaled parallel reduction difference: median trend ({input_distribution})")

    _scaled_savefig(
        fig,
        results_dir / f"scaled_parallel_error_median_{input_distribution}.png",
    )


def _run_scaled_error_plots_after_main():
    import pandas as pd

    config = _scaled_load_config()
    results_dir = _scaled_results_dir(config)
    error_csv = results_dir / "dot_error_results.csv"

    if not error_csv.exists():
        print(f"Skip scaled error plots: cannot find {error_csv}")
        return

    error_df = pd.read_csv(error_csv)

    required = {
        "input_distribution",
        "method",
        "precision",
        "threads",
        "n",
        "scaled_error_vs_sum_abs_products",
        "scaled_parallel_error_vs_sum_abs_products",
    }

    missing = required - set(error_df.columns)
    if missing:
        print(f"Skip scaled error plots: missing columns {sorted(missing)}")
        return

    input_distributions = _scaled_input_distributions(config, error_df)

    for input_distribution in input_distributions:
        if input_distribution not in set(error_df["input_distribution"]):
            continue

        _plot_scaled_mixed_error_quantile(error_df, input_distribution, results_dir)
        _plot_scaled_mixed_error_distribution(error_df, input_distribution, results_dir)
        _plot_scaled_parallel_error_median(error_df, input_distribution, results_dir)
    summary_dist, summary_n = build_summary_tables(error_df, results_dir)
    print(summary_dist.to_string(index=False))

    generate_dashboard(
        error_df=error_df,
        results_dir=results_dir,
        summary_dist=summary_dist,
        summary_n=summary_n,
        config=config if "config" in locals() else None,
    )

    print(summary_dist.to_string(index=False))

def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def safe_median(df, col):
    if col not in df.columns or df.empty:
        return float("nan")
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    return s.median() if len(s) else float("nan")


def build_summary_tables(df, out_dir):
    """
    Generate:
      1. summary_by_distribution.csv
      2. summary_by_distribution_n.csv

    Expected useful columns:
      input_distribution
      n
      method
      precision
      threads
      kappa_dot / dot_condition_number
      mixed_error_vs_fp64
      scaled_mixed_error
      parallel_error_vs_serial_same_precision
      runtime_s / time_s / runtime
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- column name compatibility ----------
    kappa_col = None
    for c in ["kappa_dot", "dot_condition_number", "condition_number"]:
        if c in df.columns:
            kappa_col = c
            break

    runtime_col = None
    for c in ["runtime_s", "time_s", "runtime", "seconds"]:
        if c in df.columns:
            runtime_col = c
            break

    mixed_rel_col = None
    for c in ["mixed_error_vs_fp64", "error_vs_serial_fp64"]:
        if c in df.columns:
            mixed_rel_col = c
            break

    scaled_mixed_col = None
    for c in ["scaled_mixed_error", "scaled_error_vs_fp64"]:
        if c in df.columns:
            scaled_mixed_col = c
            break

    parallel_col = None
    for c in ["parallel_error_vs_serial_same_precision", "parallel_error_vs_serial"]:
        if c in df.columns:
            parallel_col = c
            break

    # ---------- normalize labels ----------
    d = df.copy()
    d["precision_str"] = d["precision"].astype(str).str.lower()
    d["method_str"] = d["method"].astype(str).str.lower()

    is_serial = d["method_str"].str.contains("serial")
    is_openmp = d["method_str"].str.contains("openmp") | d["method_str"].str.contains("omp")

    is_fp64 = d["precision_str"].str.contains("fp64") & ~d["precision_str"].str.contains("mixed")
    is_mixed = (
            d["precision_str"].str.contains("mixed")
            | d["precision_str"].str.contains("fp32")
    )

    # ---------- helper: speedup ----------
    speedup_rows = []

    if runtime_col is not None:
        for (dist, prec, n), g in d.groupby(["input_distribution", "precision", "n"]):
            g_serial = g[is_serial.loc[g.index]]
            g_omp = g[is_openmp.loc[g.index]]

            if g_serial.empty or g_omp.empty:
                continue

            serial_median = g_serial[runtime_col].median()

            omp_summary = (
                g_omp.groupby("threads")[runtime_col]
                .median()
                .reset_index(name="omp_runtime_median")
            )
            omp_summary["speedup"] = serial_median / omp_summary["omp_runtime_median"]

            best = omp_summary.loc[omp_summary["speedup"].idxmax()]

            speedup_rows.append({
                "input_distribution": dist,
                "precision": prec,
                "n": n,
                "best_threads": int(best["threads"]),
                "best_speedup": best["speedup"],
            })

    speedup_df = pd.DataFrame(speedup_rows)

    # ---------- table by distribution + n ----------
    rows_n = []

    for (dist, n), g in d.groupby(["input_distribution", "n"]):
        g_serial = g[is_serial.loc[g.index]]
        g_openmp = g[is_openmp.loc[g.index]]

        g_serial_mixed = g_serial[is_mixed.loc[g_serial.index]]
        g_openmp_fp64 = g_openmp[is_fp64.loc[g_openmp.index]]
        g_openmp_mixed = g_openmp[is_mixed.loc[g_openmp.index]]

        row = {
            "input_distribution": dist,
            "n": int(n),
            "rows": len(g),
        }

        if kappa_col is not None:
            row["kappa_dot_median"] = safe_median(g_serial, kappa_col)
            row["kappa_dot_q25"] = g_serial[kappa_col].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.25)
            row["kappa_dot_q75"] = g_serial[kappa_col].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.75)

        if mixed_rel_col is not None:
            row["mixed_rel_error_median"] = safe_median(g_serial_mixed, mixed_rel_col)
            row["mixed_rel_error_q25"] = g_serial_mixed[mixed_rel_col].replace([np.inf, -np.inf],
                                                                               np.nan).dropna().quantile(0.25)
            row["mixed_rel_error_q75"] = g_serial_mixed[mixed_rel_col].replace([np.inf, -np.inf],
                                                                               np.nan).dropna().quantile(0.75)

        if scaled_mixed_col is not None:
            row["scaled_mixed_error_median"] = safe_median(g_serial_mixed, scaled_mixed_col)
            row["scaled_mixed_error_q25"] = g_serial_mixed[scaled_mixed_col].replace([np.inf, -np.inf],
                                                                                     np.nan).dropna().quantile(0.25)
            row["scaled_mixed_error_q75"] = g_serial_mixed[scaled_mixed_col].replace([np.inf, -np.inf],
                                                                                     np.nan).dropna().quantile(0.75)

        if parallel_col is not None:
            row["parallel_error_fp64_median"] = safe_median(g_openmp_fp64, parallel_col)
            row["parallel_error_mixed_median"] = safe_median(g_openmp_mixed, parallel_col)

        if not speedup_df.empty:
            sg = speedup_df[
                (speedup_df["input_distribution"] == dist)
                & (speedup_df["n"] == n)
                ]

            sg_fp64 = sg[sg["precision"].astype(str).str.lower().str.contains("fp64")]
            sg_mixed = sg[
                sg["precision"].astype(str).str.lower().str.contains("mixed")
                | sg["precision"].astype(str).str.lower().str.contains("fp32")
                ]

            if not sg_fp64.empty:
                best_fp64 = sg_fp64.loc[sg_fp64["best_speedup"].idxmax()]
                row["best_fp64_speedup"] = best_fp64["best_speedup"]
                row["best_fp64_threads"] = int(best_fp64["best_threads"])

            if not sg_mixed.empty:
                best_mixed = sg_mixed.loc[sg_mixed["best_speedup"].idxmax()]
                row["best_mixed_speedup"] = best_mixed["best_speedup"]
                row["best_mixed_threads"] = int(best_mixed["best_threads"])

        rows_n.append(row)

    summary_n = pd.DataFrame(rows_n).sort_values(["input_distribution", "n"])
    summary_n.to_csv(out_dir / "summary_by_distribution_n.csv", index=False)

    # ---------- table by distribution ----------
    rows_dist = []

    for dist, g in summary_n.groupby("input_distribution"):
        row = {
            "input_distribution": dist,
            "n_min": int(g["n"].min()),
            "n_max": int(g["n"].max()),
        }

        for col in [
            "kappa_dot_median",
            "mixed_rel_error_median",
            "scaled_mixed_error_median",
            "parallel_error_fp64_median",
            "parallel_error_mixed_median",
            "best_fp64_speedup",
            "best_mixed_speedup",
        ]:
            if col in g.columns:
                row[col] = g[col].replace([np.inf, -np.inf], np.nan).dropna().median()

        if "best_fp64_speedup" in g.columns:
            idx = g["best_fp64_speedup"].idxmax()
            row["max_fp64_speedup"] = g.loc[idx, "best_fp64_speedup"]
            row["max_fp64_speedup_n"] = int(g.loc[idx, "n"])
            row["max_fp64_speedup_threads"] = int(g.loc[idx, "best_fp64_threads"])

        if "best_mixed_speedup" in g.columns:
            idx = g["best_mixed_speedup"].idxmax()
            row["max_mixed_speedup"] = g.loc[idx, "best_mixed_speedup"]
            row["max_mixed_speedup_n"] = int(g.loc[idx, "n"])
            row["max_mixed_speedup_threads"] = int(g.loc[idx, "best_mixed_threads"])

        rows_dist.append(row)

    summary_dist = pd.DataFrame(rows_dist).sort_values("input_distribution")
    summary_dist.to_csv(out_dir / "summary_by_distribution.csv", index=False)

    print(f"[summary] wrote {out_dir / 'summary_by_distribution_n.csv'}")
    print(f"[summary] wrote {out_dir / 'summary_by_distribution.csv'}")

    return summary_dist, summary_n


def _fmt_sci(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        return f"{x:.2e}"
    except Exception:
        return "NA"


def _fmt_num(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.3g}"
    except Exception:
        return "NA"


def _median_positive(df, col):
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return None
    return vals.median()


def _median_all(df, col):
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return vals.median()


def _html_table(df, max_rows=30):
    if df is None or len(df) == 0:
        return "<p>No summary table available.</p>"

    shown = df.head(max_rows).copy()
    return shown.to_html(
        index=False,
        escape=True,
        classes="summary-table"
    )


def _collect_pngs(results_dir):
    return sorted([
        p for p in Path(results_dir).glob("*.png")
        if p.is_file()
    ])


def _plot_card(path):
    name = path.stem.replace("_", " ")
    safe_name = html_lib.escape(name)
    rel = html_lib.escape(path.name)

    return f"""
    <div class="plot-card">
        <h4>{safe_name}</h4>
        <a href="{rel}" target="_blank">
            <img src="{rel}" alt="{safe_name}">
        </a>
    </div>
    """


def generate_dashboard(error_df, results_dir, summary_dist=None, summary_n=None, config=None):
    results_dir = Path(results_dir)
    dashboard_path = results_dir / "dashboard.html"

    template_path = Path(__file__).resolve().parent / "dashboard_template.html"
    template = Template(template_path.read_text(encoding="utf-8"))

    input_distributions = sorted(
        str(x) for x in error_df["input_distribution"].dropna().unique()
    )

    pngs = _collect_pngs(results_dir)

    overview_rows = []
    for dist in input_distributions:
        sub = error_df[error_df["input_distribution"] == dist]

        overview_rows.append({
            "input_distribution": dist,
            "rows": len(sub),
            "n_values": sub["n"].nunique() if "n" in sub.columns else None,
            "median_kappa_dot": _median_positive(sub, "kappa_dot"),
            "median_mixed_error": _median_positive(sub, "mixed_error_vs_fp64"),
            "median_scaled_mixed_error": _median_positive(sub, "scaled_mixed_error"),
            "median_parallel_error": _median_positive(sub, "parallel_error_vs_serial_same_precision"),
            "median_scaled_parallel_error": _median_positive(sub, "scaled_parallel_error"),
        })

    overview_df = pd.DataFrame(overview_rows)

    overview_html_rows = []
    for _, row in overview_df.iterrows():
        overview_html_rows.append(f"""
        <tr>
            <td>{html_lib.escape(str(row["input_distribution"]))}</td>
            <td>{_fmt_num(row["rows"])}</td>
            <td>{_fmt_num(row["n_values"])}</td>
            <td>{_fmt_sci(row["median_kappa_dot"])}</td>
            <td>{_fmt_sci(row["median_mixed_error"])}</td>
            <td>{_fmt_sci(row["median_scaled_mixed_error"])}</td>
            <td>{_fmt_sci(row["median_parallel_error"])}</td>
            <td>{_fmt_sci(row["median_scaled_parallel_error"])}</td>
        </tr>
        """)

    if config:
        config_html = f"""
        <pre>{html_lib.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre>
        """
    else:
        config_html = "<p>No config object passed to dashboard.</p>"

    distribution_sections = []
    for dist in input_distributions:
        dist_key = dist.lower().replace(" ", "_")

        dist_pngs = [
            p for p in pngs
            if dist_key in p.stem.lower()
               or dist.lower() in p.stem.lower()
        ]

        priority = [
            "condition",
            "mixed",
            "scaled",
            "parallel",
            "runtime",
            "speedup",
        ]

        def sort_key(p):
            stem = p.stem.lower()
            for i, key in enumerate(priority):
                if key in stem:
                    return i
            return 99

        dist_pngs = sorted(dist_pngs, key=sort_key)

        if not dist_pngs:
            plots_html = "<p>No plots found for this distribution.</p>"
        else:
            plots_html = "\n".join(_plot_card(p) for p in dist_pngs)

        distribution_sections.append(f"""
        <details open>
            <summary>{html_lib.escape(dist)}</summary>
            <div class="plot-grid">
                {plots_html}
            </div>
        </details>
        """)

    csv_links = []
    for csv_name in [
        "summary_by_distribution.csv",
        "summary_by_n.csv",
    ]:
        csv_path = results_dir / csv_name
        if csv_path.exists():
            csv_links.append(
                f'<li><a href="{csv_name}" target="_blank">{csv_name}</a></li>'
            )

    if csv_links:
        csv_html = "<ul>" + "\n".join(csv_links) + "</ul>"
    else:
        csv_html = "<p>No summary CSV files found.</p>"

    rendered_html = template.safe_substitute(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_distribution_count=len(input_distributions),
        row_count=f"{len(error_df):,}",
        plot_count=len(pngs),
        overview_rows="".join(overview_html_rows),
        config_html=config_html,
        csv_html=csv_html,
        summary_dist_html=_html_table(summary_dist),
        distribution_sections="".join(distribution_sections),
    )

    dashboard_path.write_text(rendered_html, encoding="utf-8")
    print(f"[dashboard] saved to {dashboard_path}")
_original_main_before_scaled = main


def main():
    _original_main_before_scaled()
    _run_scaled_error_plots_after_main()


# === end scaled-error plots injected ===


if __name__ == "__main__":
    main()
