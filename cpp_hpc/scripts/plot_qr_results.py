import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, NullFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "qr"
CSV_PATH = RESULTS_DIR / "qr_serial_results.csv"

REQUIRED_COLUMNS = {
    "matrix_type",
    "m",
    "n",
    "trial",
    "runtime_seconds",
    "relative_residual",
    "orthogonality_error",
    "status",
}


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["matrix_type", "m", "n"], as_index=False)
        .agg(
            runtime_median=("runtime_seconds", "median"),
            runtime_min=("runtime_seconds", "min"),
            runtime_max=("runtime_seconds", "max"),
            residual_median=("relative_residual", "median"),
            residual_min=("relative_residual", "min"),
            residual_max=("relative_residual", "max"),
            orth_median=("orthogonality_error", "median"),
            orth_min=("orthogonality_error", "min"),
            orth_max=("orthogonality_error", "max"),
            n_obs=("trial", "count"),
        )
        .sort_values(["matrix_type", "n"])
    )


def configure_log_x_axis(ax, x_values):
    x_values = sorted(set(int(x) for x in x_values))
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(x_values) * 0.85, max(x_values) * 1.15)


def subplot_grid(n_panels: int):
    if n_panels <= 2:
        return 1, n_panels
    return 2, math.ceil(n_panels / 2)


def plot_metric(
        df: pd.DataFrame,
        summary: pd.DataFrame,
        metric_col: str,
        median_col: str,
        min_col: str,
        max_col: str,
        ylabel: str,
        title: str,
        output_name: str,
        ylog: bool = True,
        xlog: bool = True,
        show_minmax_band: bool = False,
):
    matrix_types = list(summary["matrix_type"].drop_duplicates())
    nrows, ncols = subplot_grid(len(matrix_types))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.8 * nrows),
        sharey=True,
    )

    if not isinstance(axes, (list, tuple)):
        axes_flat = [axes]
    else:
        axes_flat = list(axes)

    # NumPy arrays from matplotlib are not list/tuple.
    try:
        axes_flat = axes.ravel().tolist()
    except AttributeError:
        pass

    for ax, matrix_type in zip(axes_flat, matrix_types):
        sub = df[df["matrix_type"] == matrix_type].copy()
        sub_summary = summary[summary["matrix_type"] == matrix_type].copy()

        if show_minmax_band:
            ax.fill_between(
                sub_summary["n"],
                sub_summary[min_col],
                sub_summary[max_col],
                alpha=0.16,
                label="min-max range",
            )

        ax.scatter(
            sub["n"],
            sub[metric_col],
            alpha=0.45,
            s=35,
            label="individual runs",
        )

        ax.plot(
            sub_summary["n"],
            sub_summary[median_col],
            marker="o",
            linewidth=2.0,
            label="median over trials",
        )

        if xlog:
            configure_log_x_axis(ax, sub["n"])
        if ylog:
            ax.set_yscale("log")

        ax.set_title(matrix_type.replace("_", " "))
        ax.set_xlabel("Matrix width n")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", frameon=True)

    for ax in axes_flat[len(matrix_types):]:
        ax.axis("off")

    axes_flat[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = RESULTS_DIR / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    failed = df[df["status"] != "PASS"]
    if not failed.empty:
        print(f"[warning] ignoring {len(failed)} non-PASS rows")

    df = df[df["status"] == "PASS"].copy()
    if df.empty:
        raise RuntimeError("No PASS rows found in the benchmark CSV.")

    summary = make_summary(df)
    summary_path = RESULTS_DIR / "qr_serial_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    plot_metric(
        df=df,
        summary=summary,
        metric_col="runtime_seconds",
        median_col="runtime_median",
        min_col="runtime_min",
        max_col="runtime_max",
        ylabel="Runtime (seconds)",
        title="Serial Householder QR: runtime vs matrix size",
        output_name="qr_runtime_vs_size.png",
        show_minmax_band=False,
    )

    plot_metric(
        df=df,
        summary=summary,
        metric_col="relative_residual",
        median_col="residual_median",
        min_col="residual_min",
        max_col="residual_max",
        ylabel="Relative residual ||A - QR||_F / ||A||_F",
        title="Serial Householder QR: residual error vs matrix size",
        output_name="qr_residual_vs_size.png",
        show_minmax_band=True,
    )

    plot_metric(
        df=df,
        summary=summary,
        metric_col="orthogonality_error",
        median_col="orth_median",
        min_col="orth_min",
        max_col="orth_max",
        ylabel="Orthogonality error ||I - Q^T Q||_F",
        title="Serial Householder QR: orthogonality error vs matrix size",
        output_name="qr_orthogonality_vs_size.png",
        show_minmax_band=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
