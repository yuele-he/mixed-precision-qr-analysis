import numpy as np
import pandas as pd
from scipy.stats import linregress

from qr_factorization import householder_qr, qr_wy, block_qr


QR_METHODS = {
    "Householder": householder_qr,
    "WY": qr_wy,
}


METRIC_LABELS = {
    "reconstruction": "Reconstruction",
    "orthogonality": "Orthogonality",
    "least_squares": "Least Squares",
}


def get_qr_methods(block_size=32):
    """
    Return QR methods used in the experiments.

    The block QR method needs block_size, so it is wrapped by a lambda.
    """
    methods = dict(QR_METHODS)
    methods["Block QR"] = lambda A: block_qr(A, block_size=block_size)
    return methods


def compute_qr_errors(A, b, qr_func):
    """
    Compute three QR error metrics.

    Metrics
    -------
    reconstruction:
        ||A - QR||_F / ||A||_F

    orthogonality:
        ||Q^T Q - I||_F / ||Q||_F

    least_squares:
        ||R x - Q^T b||_2 / ||b||_2
    """
    Q, R = qr_func(A)

    reconstruction = (
        np.linalg.norm(A - Q @ R, ord="fro")
        / np.linalg.norm(A, ord="fro")
    )

    identity = np.eye(Q.shape[1], dtype=Q.dtype)

    orthogonality = (
        np.linalg.norm(Q.T @ Q - identity, ord="fro")
        / np.linalg.norm(Q, ord="fro")
    )

    rhs = Q.T @ b
    x = np.linalg.lstsq(R, rhs, rcond=None)[0]

    least_squares = (
        np.linalg.norm(R @ x - rhs, ord=2)
        / np.linalg.norm(b, ord=2)
    )

    return {
        "reconstruction": float(reconstruction),
        "orthogonality": float(orthogonality),
        "least_squares": float(least_squares),
    }


def generate_qr_problem(m, n, dtype=np.float64, rng=None, data_type="uniform"):
    """
    Generate a random test problem A and b.

    data_type
    ---------
    uniform:
        A, b ~ U(1, 2), closer to your dissertation setting.

    normal:
        A, b ~ N(0, 1), useful for zero-mean experiments.
    """
    if rng is None:
        rng = np.random.default_rng()

    if data_type == "uniform":
        A = rng.uniform(1, 2, size=(m, n)).astype(dtype)
        b = rng.uniform(1, 2, size=(m, 1)).astype(dtype)

        A = A / np.linalg.norm(A, ord="fro")

    elif data_type == "normal":
        A = rng.standard_normal(size=(m, n)).astype(dtype)
        b = rng.standard_normal(size=(m, 1)).astype(dtype)

    else:
        raise ValueError("data_type must be either 'uniform' or 'normal'.")

    return A, b


def run_qr_experiments(
    n_values,
    precisions=(np.float32, np.float64),
    repeats=20,
    block_size=32,
    seed=42,
    data_type="uniform",
):
    """
    Run QR error experiments for several matrix sizes and precisions.

    Matrix shape:
        A in R^{m x n}, where m = 2n

    Returns
    -------
    pandas.DataFrame with columns:
        n, m, trial, precision, method,
        reconstruction, orthogonality, least_squares
    """
    rng = np.random.default_rng(seed)
    methods = get_qr_methods(block_size=block_size)

    rows = []

    for dtype in precisions:
        precision_label = np.dtype(dtype).name

        for n in n_values:
            n = int(n)
            m = 2 * n

            for trial in range(repeats):
                A, b = generate_qr_problem(
                    m=m,
                    n=n,
                    dtype=dtype,
                    rng=rng,
                    data_type=data_type,
                )

                for method_name, qr_func in methods.items():
                    errors = compute_qr_errors(A, b, qr_func)

                    rows.append({
                        "n": n,
                        "m": m,
                        "trial": trial,
                        "precision": precision_label,
                        "method": method_name,
                        **errors,
                    })

    return pd.DataFrame(rows)


def fit_error_growth(df):
    """
    Fit log-log growth model for each method, precision, and metric.

    Model:
        error ≈ exp(c) * n^k

    The fit uses the maximum error over repeated trials for each matrix size.
    """
    rows = []

    metrics = ["reconstruction", "orthogonality", "least_squares"]

    for (method, precision), group in df.groupby(["method", "precision"]):
        for metric in metrics:
            summary = (
                group.groupby("n")[metric]
                .max()
                .reset_index()
                .sort_values("n")
            )

            x = summary["n"].to_numpy(dtype=float)
            y = summary[metric].to_numpy(dtype=float)

            k, c, r_value, p_value, std_err = linregress(
                np.log(x),
                np.log(y + 1e-300),
            )

            rows.append({
                "method": method,
                "precision": precision,
                "metric": metric,
                "k": float(k),
                "c": float(c),
                "r_squared": float(r_value ** 2),
                "std_err": float(std_err),
            })

    return pd.DataFrame(rows)


def make_fit_table(fit_df):
    """
    Create a readable fit-parameter table.
    """
    table = fit_df.copy()
    table["metric"] = table["metric"].map(METRIC_LABELS)
    table["k"] = table["k"].round(3)
    table["c"] = table["c"].round(3)
    table["r_squared"] = table["r_squared"].round(3)

    return table.pivot_table(
        index=["method", "precision"],
        columns="metric",
        values=["k", "c", "r_squared"],
    ).reset_index()