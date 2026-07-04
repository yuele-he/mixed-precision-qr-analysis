"""Numerical error experiments for QR factorization.

This module contains computation, tabulation, persistence, and optional
experiment-level parallelism. Plotting functions are kept in ``plotter.py`` so
that experiments, figures, and notebooks remain cleanly separated.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import linregress

from qr_factorization import (
    block_qr,
    householder_qr,
    householder_qr_mixed_fma,
    householder_qr_mixed_fma_store32,
    qr_wy,
)

Array = np.ndarray
MetricName = Literal["reconstruction", "orthogonality", "least_squares"]
ReferenceMode = Literal["working", "original"]
DataType = Literal["uniform", "normal"]
ParallelBackend = Literal["threads", "processes", "none"]
ExperimentKind = Literal["method", "precision"]


METRIC_NAMES: tuple[MetricName, ...] = (
    "reconstruction",
    "orthogonality",
    "least_squares",
)

METRIC_LABELS = {
    "reconstruction": "Reconstruction",
    "orthogonality": "Orthogonality",
    "least_squares": "Least Squares",
}


@dataclass(frozen=True)
class PrecisionMode:
    """Description of one precision experiment mode.

    Parameters
    ----------
    input_dtype:
        Precision used when the test matrix is first rounded.
    compute_dtype:
        Precision used inside the QR algorithm.
    factor_dtype:
        Precision used to store the returned Q and R factors.
    label:
        Human-readable label for plots and tables.
    """

    input_dtype: type
    compute_dtype: type
    factor_dtype: type
    label: str


PRECISION_MODES: dict[str, PrecisionMode] = {
    "float64": PrecisionMode(np.float64, np.float64, np.float64, "Float64"),
    "float32": PrecisionMode(np.float32, np.float32, np.float32, "Float32"),
    # Baseline: FP32-rounded input, then the whole QR factorization is done in FP64.
    # Kept as a reference baseline, not the main dissertation-style mixed-FMA mode.
    "mixed32_64": PrecisionMode(np.float32, np.float64, np.float64, "FP32 input + FP64 QR"),
    # Optional stricter storage baseline: factors are rounded back to FP32 at the end.
    "mixed32_64_store32": PrecisionMode(np.float32, np.float64, np.float32, "FP32 input + FP64 QR, store FP32"),
    # Dissertation-style mixed-FMA model:
    # FP32 operands with FP64 accumulation/factor storage.
    "mixed_fma32_64": PrecisionMode(np.float32, np.float64, np.float64, "Mixed-FMA, FP64 acc."),
    # Conservative storage-limited mixed-FMA model:
    # FP32 storage between Householder steps, FP64 local product/add updates.
    "mixed_fma32_64_store32": PrecisionMode(np.float32, np.float64, np.float32, "Mixed-FMA, FP32 storage"),
}


def get_qr_method(method_name: str, block_size: int = 32) -> Callable[[Array], tuple[Array, Array]]:
    """Return a QR routine by name."""
    normalized = method_name.strip().lower().replace("_", " ")

    if normalized in {"householder", "householder qr"}:
        return householder_qr
    if normalized in {"mixed householder", "mixed householder qr", "mixed-fma householder", "mixed fma householder", "mixed-fma householder qr", "mixed fma householder qr"}:
        return householder_qr_mixed_fma
    if normalized in {"wy", "wy qr"}:
        return qr_wy
    if normalized in {"block", "block qr"}:
        return lambda A: block_qr(A, block_size=block_size)

    raise ValueError(f"Unknown QR method: {method_name!r}")


def get_qr_methods(block_size: int = 32) -> dict[str, Callable[[Array], tuple[Array, Array]]]:
    """Return the standard method set used in the QR experiments."""
    return {
        "Householder": householder_qr,
        "WY": qr_wy,
        "Block QR": lambda A: block_qr(A, block_size=block_size),
    }


def generate_qr_problem(
    m: int,
    n: int,
    *,
    rng: np.random.Generator,
    data_type: DataType = "uniform",
) -> tuple[Array, Array]:
    """Generate a random QR test problem in float64.

    ``uniform`` uses U(1, 2), which matches the dissertation-style non-zero mean
    random data. ``normal`` uses N(0, 1), useful for zero-mean experiments.
    """
    if data_type == "uniform":
        A = rng.uniform(1, 2, size=(m, n))
        A = A / np.linalg.norm(A, ord="fro")
        b = rng.uniform(1, 2, size=(m, 1))
    elif data_type == "normal":
        A = rng.standard_normal(size=(m, n))
        b = rng.standard_normal(size=(m, 1))
    else:
        raise ValueError("data_type must be either 'uniform' or 'normal'.")

    return A.astype(np.float64), b.astype(np.float64)


def compute_qr_errors(A: Array, b: Array, qr_func: Callable[[Array], tuple[Array, Array]]) -> dict[MetricName, float]:
    """Compute QR error metrics by first calling ``qr_func(A)``."""
    Q, R = qr_func(A)
    return compute_qr_errors_from_factors(A=A, b=b, Q=Q, R=R)


def compute_qr_errors_from_factors(A: Array, b: Array, Q: Array, R: Array) -> dict[MetricName, float]:
    """Compute reconstruction, orthogonality, and least-squares residual errors.

    The metrics are evaluated in float64 so that the error measurement is not
    accidentally limited by the working precision of the experiment.
    """
    A64 = np.asarray(A, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    Q64 = np.asarray(Q, dtype=np.float64)
    R64 = np.asarray(R, dtype=np.float64)

    reconstruction = np.linalg.norm(A64 - Q64 @ R64, ord="fro") / np.linalg.norm(A64, ord="fro")

    identity = np.eye(Q64.shape[1], dtype=np.float64)
    orthogonality = np.linalg.norm(Q64.T @ Q64 - identity, ord="fro") / np.linalg.norm(Q64, ord="fro")

    rhs = Q64.T @ b64
    x = np.linalg.lstsq(R64, rhs, rcond=None)[0]
    least_squares = np.linalg.norm(R64 @ x - rhs, ord=2) / np.linalg.norm(b64, ord=2)

    return {
        "reconstruction": float(reconstruction),
        "orthogonality": float(orthogonality),
        "least_squares": float(least_squares),
    }


def factorize_with_precision(
    A_reference: Array,
    *,
    method_name: str,
    precision_mode: str,
    block_size: int = 32,
) -> tuple[Array, Array, Array]:
    """Factorize ``A_reference`` under a selected precision mode.

    Returns ``Q, R, A_working`` where ``A_working`` is the matrix actually seen by
    the algorithm after input rounding.
    """
    if precision_mode not in PRECISION_MODES:
        valid = ", ".join(PRECISION_MODES)
        raise ValueError(f"Unknown precision_mode={precision_mode!r}. Valid modes: {valid}")

    mode = PRECISION_MODES[precision_mode]

    A_working = np.asarray(A_reference, dtype=mode.input_dtype)

    if precision_mode in {"mixed_fma32_64", "mixed_fma32_64_store32"}:
        normalized_method = method_name.strip().lower().replace("_", " ")
        if normalized_method not in {
            "householder",
            "householder qr",
            "mixed householder",
            "mixed householder qr",
            "mixed-fma householder",
            "mixed fma householder",
            "mixed-fma householder qr",
            "mixed fma householder qr",
        }:
            raise ValueError(
                f"precision_mode={precision_mode!r} is currently implemented only for Householder QR. "
                "Use method_name='Householder' or 'Mixed-FMA Householder QR'."
            )

        if precision_mode == "mixed_fma32_64":
            # Dissertation-style model: FP32 operands + FP64 accumulation/storage.
            Q, R = householder_qr_mixed_fma(
                A_working,
                operand_dtype=mode.input_dtype,
                accumulator_dtype=mode.compute_dtype,
            )
        else:
            # Storage-limited variant: round back to FP32 after each reflector update.
            Q, R = householder_qr_mixed_fma_store32(
                A_working,
                storage_dtype=mode.input_dtype,
                accumulator_dtype=mode.compute_dtype,
            )

        return Q.astype(mode.factor_dtype), R.astype(mode.factor_dtype), A_working

    A_compute = A_working.astype(mode.compute_dtype)

    qr_func = get_qr_method(method_name, block_size=block_size)
    Q, R = qr_func(A_compute)

    return Q.astype(mode.factor_dtype), R.astype(mode.factor_dtype), A_working


def _method_trial_task(task: dict[str, object]) -> dict[str, object]:
    """Run one independent method-comparison trial.

    This is used by both serial and parallel method-comparison runners.
    """
    n = int(task["n"])
    m = int(task["m"])
    trial = int(task["trial"])
    method_name = str(task["method"])
    dtype = np.dtype(str(task["precision"])).type
    seed = int(task["seed"])
    data_type = task["data_type"]
    block_size = int(task["block_size"])

    rng = np.random.default_rng(seed)
    A_ref, b_ref = generate_qr_problem(m, n, rng=rng, data_type=data_type)  # type: ignore[arg-type]
    A = A_ref.astype(dtype)
    b = b_ref.astype(dtype)

    qr_func = get_qr_method(method_name, block_size=block_size)
    errors = compute_qr_errors(A, b, qr_func)

    return {
        "n": n,
        "m": m,
        "trial": trial,
        "method": method_name,
        "precision": np.dtype(dtype).name,
        **errors,
    }


def _precision_trial_task(task: dict[str, object]) -> dict[str, object]:
    """Run one independent precision-comparison trial.

    This is used by both serial and parallel precision-comparison runners.
    """
    n = int(task["n"])
    m = int(task["m"])
    trial = int(task["trial"])
    method_name = str(task["method"])
    precision_mode = str(task["precision_mode"])
    seed = int(task["seed"])
    data_type = task["data_type"]
    reference = task["reference"]
    block_size = int(task["block_size"])

    rng = np.random.default_rng(seed)
    A_ref, b_ref = generate_qr_problem(m, n, rng=rng, data_type=data_type)  # type: ignore[arg-type]

    mode = PRECISION_MODES[precision_mode]
    b_working = b_ref.astype(mode.input_dtype)

    Q, R, A_working = factorize_with_precision(
        A_ref,
        method_name=method_name,
        precision_mode=precision_mode,
        block_size=block_size,
    )

    if reference == "working":
        A_for_error = A_working
        b_for_error = b_working
    elif reference == "original":
        A_for_error = A_ref
        b_for_error = b_ref
    else:
        raise ValueError("reference must be either 'working' or 'original'.")

    errors = compute_qr_errors_from_factors(A_for_error, b_for_error, Q, R)

    return {
        "n": n,
        "m": m,
        "trial": trial,
        "method": method_name,
        "precision_mode": precision_mode,
        "precision_label": mode.label,
        "reference": reference,
        **errors,
    }


def _make_seed_sequence(seed: int, n_tasks: int) -> list[int]:
    """Generate deterministic independent seeds for experiment tasks."""
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(n_tasks)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in child_sequences]


def _execute_tasks(
    tasks: list[dict[str, object]],
    task_func: Callable[[dict[str, object]], dict[str, object]],
    *,
    parallel: bool = False,
    backend: ParallelBackend = "threads",
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Execute experiment tasks either serially or in parallel."""
    if not tasks:
        return pd.DataFrame()

    if (not parallel) or backend == "none" or max_workers == 1:
        rows = []
        for i, task in enumerate(tasks, start=1):
            rows.append(task_func(task))
            if show_progress and (i == len(tasks) or i % max(1, len(tasks) // 10) == 0):
                print(f"Completed {i}/{len(tasks)} tasks")
        return pd.DataFrame(rows)

    if backend == "threads":
        executor_cls = ThreadPoolExecutor
    elif backend == "processes":
        executor_cls = ProcessPoolExecutor
    else:
        raise ValueError("backend must be one of: 'threads', 'processes', 'none'.")

    rows: list[dict[str, object]] = []
    completed = 0

    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(task_func, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if show_progress and (completed == len(tasks) or completed % max(1, len(tasks) // 10) == 0):
                print(f"Completed {completed}/{len(tasks)} tasks")

    # Return a stable ordering independent of completion order.
    df = pd.DataFrame(rows)
    sort_columns = [c for c in ["n", "trial", "method", "precision", "precision_mode", "reference"] if c in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


def run_qr_method_experiments(
    n_values: Iterable[int],
    *,
    methods: Iterable[str] = ("Householder", "WY", "Block QR"),
    precisions: Iterable[type] = (np.float32, np.float64),
    repeats: int = 20,
    block_size: int = 32,
    seed: int = 42,
    data_type: DataType = "uniform",
    parallel: bool = False,
    backend: ParallelBackend = "threads",
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compare QR methods across sizes and ordinary precisions.

    ``parallel=True`` parallelizes independent experiment trials. It does not
    implement a parallel QR algorithm.
    """
    n_values = [int(n) for n in n_values]
    methods = list(methods)
    precisions = [np.dtype(dtype).name for dtype in precisions]

    tasks: list[dict[str, object]] = []
    for n in n_values:
        m = 2 * n
        for trial in range(repeats):
            for precision in precisions:
                for method_name in methods:
                    tasks.append({
                        "n": n,
                        "m": m,
                        "trial": trial,
                        "method": method_name,
                        "precision": precision,
                        "block_size": block_size,
                        "data_type": data_type,
                    })

    seeds = _make_seed_sequence(seed, len(tasks))
    for task, task_seed in zip(tasks, seeds):
        task["seed"] = task_seed

    return _execute_tasks(
        tasks,
        _method_trial_task,
        parallel=parallel,
        backend=backend,
        max_workers=max_workers,
        show_progress=show_progress,
    )


def run_qr_precision_experiments(
    n_values: Iterable[int],
    *,
    methods: Iterable[str] = ("Householder", "Block QR"),
    precision_modes: Iterable[str] = ("float64", "float32", "mixed_fma32_64"),
    repeats: int = 20,
    block_size: int = 32,
    seed: int = 42,
    data_type: DataType = "uniform",
    reference: ReferenceMode = "working",
    parallel: bool = False,
    backend: ParallelBackend = "threads",
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compare how precision affects QR errors.

    Default modes compare Float64, Float32, and ``mixed_fma32_64``.
    ``mixed_fma32_64`` is currently Householder-only and simulates
    FP32 operands with FP64 accumulation/factor storage, following the
    dissertation-style mixed-FMA model.

    ``reference='working'`` evaluates QR quality relative to the rounded matrix
    actually given to the algorithm. This isolates factorization error.

    ``reference='original'`` evaluates errors relative to the original float64
    matrix. This includes both input rounding error and factorization error.

    ``parallel=True`` parallelizes independent experiment trials. It does not
    implement a parallel QR algorithm.
    """
    n_values = [int(n) for n in n_values]
    methods = list(methods)
    precision_modes = list(precision_modes)

    invalid_modes = [mode for mode in precision_modes if mode not in PRECISION_MODES]
    if invalid_modes:
        raise ValueError(f"Unknown precision modes: {invalid_modes}")

    tasks: list[dict[str, object]] = []
    for n in n_values:
        m = 2 * n
        for trial in range(repeats):
            for precision_mode in precision_modes:
                for method_name in methods:
                    tasks.append({
                        "n": n,
                        "m": m,
                        "trial": trial,
                        "method": method_name,
                        "precision_mode": precision_mode,
                        "block_size": block_size,
                        "data_type": data_type,
                        "reference": reference,
                    })

    seeds = _make_seed_sequence(seed, len(tasks))
    for task, task_seed in zip(tasks, seeds):
        task["seed"] = task_seed

    return _execute_tasks(
        tasks,
        _precision_trial_task,
        parallel=parallel,
        backend=backend,
        max_workers=max_workers,
        show_progress=show_progress,
    )


def fit_error_growth(
    df: pd.DataFrame,
    *,
    group_columns: Iterable[str] = ("method", "precision"),
    metrics: Iterable[MetricName] = METRIC_NAMES,
    statistic: Literal["max", "mean", "median"] = "max",
) -> pd.DataFrame:
    """Fit ``error ≈ exp(c) * n**k`` on a log-log scale.

    The fit is performed for each group, after aggregating repeated trials at
    each matrix size.
    """
    rows: list[dict[str, object]] = []
    group_columns = list(group_columns)

    for group_key, group in df.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_dict = dict(zip(group_columns, group_key))

        for metric in metrics:
            summary = group.groupby("n")[metric].agg(statistic).reset_index().sort_values("n")
            x = summary["n"].to_numpy(dtype=float)
            y = summary[metric].to_numpy(dtype=float)

            k, c, r_value, _p_value, std_err = linregress(np.log(x), np.log(y + 1e-300))

            rows.append({
                **key_dict,
                "metric": metric,
                "statistic": statistic,
                "k": float(k),
                "c": float(c),
                "r_squared": float(r_value ** 2),
                "std_err": float(std_err),
            })

    return pd.DataFrame(rows)


def make_fit_table(fit_df: pd.DataFrame, *, round_digits: int = 3) -> pd.DataFrame:
    """Create a readable table of fitted growth exponents."""
    table = fit_df.copy()
    table["metric"] = table["metric"].map(METRIC_LABELS).fillna(table["metric"])

    for column in ["k", "c", "r_squared", "std_err"]:
        if column in table:
            table[column] = table[column].round(round_digits)

    index_columns = [c for c in ["method", "precision", "precision_mode", "reference", "statistic"] if c in table.columns]

    return table.pivot_table(
        index=index_columns,
        columns="metric",
        values=["k", "c", "r_squared"],
    ).reset_index()


def save_results(df: pd.DataFrame, path: str | Path, *, json_indent: int = 2) -> Path:
    """Save experiment results to CSV or JSON.

    The file type is inferred from the suffix:
        - .csv  -> CSV table
        - .json -> JSON records
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=json_indent)
    else:
        raise ValueError("Result path must end with '.csv' or '.json'.")

    return path


def load_results(path: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV or JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path, orient="records")

    raise ValueError("Result path must end with '.csv' or '.json'.")


def get_or_run_results(
    path: str | Path,
    runner: Callable[[], pd.DataFrame],
    *,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load existing results, or run and save new results.

    Parameters
    ----------
    path:
        CSV or JSON file path.
    runner:
        Zero-argument function that returns a DataFrame when recomputation is needed.
    regenerate:
        If True, ignore existing file and recompute.
    """
    path = Path(path)

    if path.exists() and not regenerate:
        print(f"Loading existing results from: {path}")
        return load_results(path)

    print(f"Generating new results and saving to: {path}")
    df = runner()
    save_results(df, path)
    return df


if __name__ == "__main__":
    n_values = np.round(np.logspace(np.log10(20), np.log10(256), num=6)).astype(int)

    df_methods = run_qr_method_experiments(n_values, repeats=3, parallel=True, max_workers=2)
    fit_methods = fit_error_growth(df_methods)
    print(make_fit_table(fit_methods))
