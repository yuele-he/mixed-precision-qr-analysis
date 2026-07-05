"""QR factorization algorithms used in the numerical error experiments.

The implementations here are intentionally written in pure NumPy so that the
rounding behaviour is easy to inspect. They are not meant to replace highly
optimized LAPACK routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

try:
    from src.mixed_kernels import dot_low_product_high_accumulation
except ImportError:
    from mixed_kernels import dot_low_product_high_accumulation

Array = np.ndarray


def _as_floating_array(A: Array, dtype=None) -> Array:
    """Return a floating-point copy of ``A``.

    If ``dtype`` is provided, the matrix is converted to that floating-point
    dtype. Otherwise, the input precision is preserved when it is already a
    floating-point array; non-floating inputs are converted to FP64.
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(f"dtype must be floating point, got {dtype}.")
        return np.asarray(A, dtype=dtype).copy()

    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.floating):
        A = A.astype(np.float64)
    return A.copy()


def _as_dtype(dtype) -> np.dtype:
    """Normalize and validate a floating-point dtype."""
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"Expected a floating-point dtype, got {dtype}.")
    return dtype


def _mixed_dot(a, b, *, operand_dtype, product_dtype, accumulator_dtype):
    """Convenience wrapper around the main mixed dot-product kernel."""
    return dot_low_product_high_accumulation(
        a,
        b,
        operand_dtype=operand_dtype,
        product_dtype=product_dtype,
        accumulator_dtype=accumulator_dtype,
    )


def householder_vector(x: Array) -> tuple[Array, float]:
    """Compute a Householder vector ``v`` and scalar ``beta``.

    The reflector is

        H = I - beta * v v.T,

    with ``v[0] = 1`` whenever the reflector is non-trivial. This follows the
    standard stable construction used in QR factorization.
    """
    x = np.asarray(x).reshape(-1, 1)
    dtype = x.dtype

    if x.size == 0:
        raise ValueError("householder_vector requires a non-empty vector.")

    sigma = (x[1:].T @ x[1:]).item()
    alpha = x[0, 0].item()

    v = x.copy()

    if sigma == 0:
        v.fill(dtype.type(0.0))
        v[0, 0] = dtype.type(1.0)
        beta = 0.0 if alpha >= 0 else 2.0
        return v, beta

    norm_x = np.sqrt(alpha * alpha + sigma)
    if alpha <= 0:
        v0 = alpha - norm_x
    else:
        v0 = -sigma / (alpha + norm_x)

    v[0, 0] = dtype.type(v0)
    beta = float(2.0 * v0 * v0 / (sigma + v0 * v0))
    v = v / v[0, 0]

    return v.astype(dtype, copy=False), beta


def householder_qr(A: Array) -> tuple[Array, Array]:
    """Compute QR factorization using Householder reflections."""
    A = _as_floating_array(A)
    dtype = A.dtype
    m, n = A.shape

    Q = np.eye(m, dtype=dtype)

    for k in range(min(m, n)):
        v, beta = householder_vector(A[k:, k])
        if beta == 0.0:
            continue

        A[k:, k:] -= beta * v @ (v.T @ A[k:, k:])
        Q[:, k:] -= (Q[:, k:] @ v) @ (beta * v.T)

    R = np.triu(A[:m, :n])
    return Q, R


def householder_qr_with_mixed_dot(
    A: Array,
    *,
    operand_dtype=np.float32,
    product_dtype=np.float32,
    accumulator_dtype=np.float64,
    storage_dtype=None,
) -> tuple[Array, Array]:
    """Householder QR using explicit configurable mixed-precision dot products.

    Main arithmetic model inside dot products:

        low-precision operands + low-precision products + high-precision
        accumulation.

    Defaults:
        FP32 operands + FP32 products + FP64 accumulation.

    Parameters
    ----------
    A:
        Input matrix.
    operand_dtype:
        Precision used to round operands in each mixed dot product.
    product_dtype:
        Precision used to round each product before accumulation.
    accumulator_dtype:
        Precision used for accumulation and the main working arithmetic.
    storage_dtype:
        Optional precision used to store ``A`` and ``Q`` between Householder
        steps. If ``None``, values are stored in ``accumulator_dtype``. Setting
        ``storage_dtype=np.float32`` simulates a storage-limited model.

    Notes
    -----
    This is a research/teaching implementation. It replaces the BLAS-like
    operations ``v.T @ A`` and ``Q @ v`` with explicit mixed dot products so the
    rounding model is visible and configurable. It is not intended to be fast.
    """
    operand_dtype = _as_dtype(operand_dtype)
    product_dtype = _as_dtype(product_dtype)
    accumulator_dtype = _as_dtype(accumulator_dtype)
    storage_dtype = accumulator_dtype if storage_dtype is None else _as_dtype(storage_dtype)

    A_work = np.asarray(A, dtype=storage_dtype).astype(accumulator_dtype, copy=True)
    m, n = A_work.shape
    Q = np.eye(m, dtype=accumulator_dtype)

    for k in range(min(m, n)):
        # Reflector generation is kept in accumulator precision so that the
        # experiment isolates the dot-product rounding inside reflector updates.
        x = A_work[k:, k].astype(accumulator_dtype, copy=True)
        v, beta = householder_vector(x)
        v = v.reshape(-1).astype(accumulator_dtype, copy=False)

        if beta == 0.0:
            continue

        beta_acc = accumulator_dtype.type(beta)

        # A[k:, k:] = A[k:, k:] - beta * v * (v.T @ A[k:, k:])
        for j in range(k, n):
            tau = _mixed_dot(
                v,
                A_work[k:, j],
                operand_dtype=operand_dtype,
                product_dtype=product_dtype,
                accumulator_dtype=accumulator_dtype,
            )
            A_work[k:, j] = A_work[k:, j] - beta_acc * v * tau

        # Q[:, k:] = Q[:, k:] - (Q[:, k:] @ v) * beta * v.T
        for i in range(m):
            tau = _mixed_dot(
                Q[i, k:],
                v,
                operand_dtype=operand_dtype,
                product_dtype=product_dtype,
                accumulator_dtype=accumulator_dtype,
            )
            Q[i, k:] = Q[i, k:] - beta_acc * tau * v

        # Optional storage rounding after each step.
        if storage_dtype != accumulator_dtype:
            A_work = A_work.astype(storage_dtype).astype(accumulator_dtype)
            Q = Q.astype(storage_dtype).astype(accumulator_dtype)

    R = np.triu(A_work[:m, :n])
    return Q.astype(accumulator_dtype, copy=False), R.astype(accumulator_dtype, copy=False)


def qr_wy(A: Array) -> tuple[Array, Array]:
    """Compute QR factorization using a compact WY-style representation."""
    A = _as_floating_array(A)
    dtype = A.dtype
    m, n = A.shape

    Y: Array | None = None
    W: Array | None = None

    for j in range(min(m, n)):
        v_hat, beta = householder_vector(A[j:, j])
        if beta == 0.0:
            continue

        A[j:, j:] -= beta * v_hat @ (v_hat.T @ A[j:, j:])

        v_full = np.zeros((m, 1), dtype=dtype)
        v_full[j:] = v_hat

        if Y is None or W is None:
            Y = v_full
            W = beta * v_full
        else:
            ytv = Y.T @ v_full
            z = beta * v_full - beta * W @ ytv
            Y = np.hstack((Y, v_full))
            W = np.hstack((W, z))

    if Y is None or W is None:
        Q = np.eye(m, dtype=dtype)
    else:
        Q = np.eye(m, dtype=dtype) - W @ Y.T

    R = np.triu(A[:m, :n])
    return Q, R


def block_qr(A: Array, block_size: int = 64) -> tuple[Array, Array]:
    """Compute a simple block QR factorization."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    A = _as_floating_array(A)
    dtype = A.dtype
    m, n = A.shape

    Q_total = np.eye(m, dtype=dtype)

    for start in range(0, n, block_size):
        panel_width = min(block_size, n - start)
        panel = A[start:, start:start + panel_width]

        Q_panel, _ = qr_wy(panel)

        A[start:, start:] = Q_panel.T @ A[start:, start:]

        Q_block = np.eye(m, dtype=dtype)
        Q_block[start:, start:] = Q_panel
        Q_total = Q_total @ Q_block

    R = np.triu(A[:m, :n])
    return Q_total, R


def householder_qr_mixed_fma(
    A: Array,
    *,
    operand_dtype: type = np.float32,
    accumulator_dtype: type = np.float64,
) -> tuple[Array, Array]:
    """Legacy compatibility wrapper for a rounded-input QR experiment.

    Despite the historical name, this is not an explicit FMA QR kernel. It
    rounds the input matrix to ``operand_dtype``, promotes it to
    ``accumulator_dtype``, and then calls ordinary Householder QR.
    """
    A_operands = np.asarray(A, dtype=operand_dtype)
    A_acc = A_operands.astype(accumulator_dtype)
    Q, R = householder_qr(A_acc)
    return Q.astype(accumulator_dtype, copy=False), R.astype(accumulator_dtype, copy=False)


def householder_qr_mixed_fma_store32(
    A: Array,
    *,
    storage_dtype: type = np.float32,
    accumulator_dtype: type = np.float64,
) -> tuple[Array, Array]:
    """Legacy compatibility wrapper for a storage-limited QR experiment.

    Despite the historical name, this is not an explicit FMA QR kernel. It stores
    the working arrays in ``storage_dtype`` between steps and computes local
    Householder updates in ``accumulator_dtype``.
    """
    A_work = np.asarray(A, dtype=storage_dtype).copy()
    m, n = A_work.shape

    Q_work = np.eye(m, dtype=storage_dtype)

    for k in range(min(m, n)):
        x_acc = A_work[k:, k].astype(accumulator_dtype)
        v_acc, beta = householder_vector(x_acc)
        v_acc = v_acc.astype(accumulator_dtype, copy=False)

        if beta == 0.0:
            continue

        A_block_acc = A_work[k:, k:].astype(accumulator_dtype)
        A_update_acc = A_block_acc - beta * v_acc @ (v_acc.T @ A_block_acc)
        A_work[k:, k:] = A_update_acc.astype(storage_dtype)

        Q_block_acc = Q_work[:, k:].astype(accumulator_dtype)
        Q_update_acc = Q_block_acc - (Q_block_acc @ v_acc) @ (beta * v_acc.T)
        Q_work[:, k:] = Q_update_acc.astype(storage_dtype)

    R_work = np.triu(A_work[:m, :n])
    return Q_work, R_work


@dataclass
class QRExperiment:
    """Small convenience wrapper kept for backward compatibility."""

    dtypes: Iterable[np.dtype] = (np.float64,)
    sizes: Iterable[tuple[int, int]] = ((200, 100),)
    block_size: int = 64

    def __post_init__(self) -> None:
        self.methods: dict[str, Callable[[Array], tuple[Array, Array]]] = {
            "Householder": householder_qr,
            "WY": qr_wy,
            "Block QR": lambda A: block_qr(A, block_size=self.block_size),
            "Mixed-dot Householder": householder_qr_with_mixed_dot,
        }

    def run_single(self, m: int = 200, n: int = 100, dtype=np.float64) -> dict[str, dict[str, float]]:
        """Run one random QR experiment and return basic error metrics."""
        try:
            from src.error_analysis import compute_qr_errors
        except ImportError:
            from error_analysis import compute_qr_errors

        rng = np.random.default_rng(42)
        A = rng.standard_normal((m, n)).astype(dtype)
        b = rng.uniform(1, 2, size=(m, 1)).astype(dtype)

        return {
            name: compute_qr_errors(A, b, fn)
            for name, fn in self.methods.items()
        }

    def run_sweep(self) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Run a lightweight sweep over ``self.sizes`` and ``self.dtypes``."""
        try:
            from src.error_analysis import compute_qr_errors
        except ImportError:
            from error_analysis import compute_qr_errors

        rng = np.random.default_rng(42)
        output: dict[str, dict[str, dict[str, list[float]]]] = {}

        for dtype in self.dtypes:
            precision = np.dtype(dtype).name
            output[precision] = {
                name: {"sizes": [], "reconstruction": [], "orthogonality": [], "least_squares": []}
                for name in self.methods
            }

            for m, n in self.sizes:
                A = rng.standard_normal((m, n)).astype(dtype)
                b = rng.uniform(1, 2, size=(m, 1)).astype(dtype)

                for name, fn in self.methods.items():
                    errors = compute_qr_errors(A, b, fn)
                    output[precision][name]["sizes"].append(int(n))
                    for metric, value in errors.items():
                        output[precision][name][metric].append(float(value))

        return output


if __name__ == "__main__":
    experiment = QRExperiment()
    print(experiment.run_single())
