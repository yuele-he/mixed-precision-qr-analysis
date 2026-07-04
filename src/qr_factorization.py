"""QR factorization algorithms used in the numerical error experiments.

The implementations here are intentionally written in pure NumPy so that the
rounding behaviour is easy to inspect. They are not meant to replace highly
optimized LAPACK routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

Array = np.ndarray


def _as_floating_array(A: Array) -> Array:
    """Return a floating-point copy of ``A`` while preserving its precision."""
    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.floating):
        A = A.astype(np.float64)
    return A.copy()


def householder_vector(x: Array) -> tuple[Array, float]:
    """Compute a Householder vector ``v`` and scalar ``beta``.

    The reflector is

        H = I - beta * v v.T,

    with ``v[0] = 1`` whenever the reflector is non-trivial. This follows the
    standard stable construction used in QR factorization.

    Parameters
    ----------
    x:
        One-dimensional vector to be reflected.

    Returns
    -------
    v:
        Column vector defining the Householder reflector.
    beta:
        Scalar coefficient of the reflector.
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
    """Compute QR factorization using Householder reflections.

    Parameters
    ----------
    A:
        Matrix with shape ``(m, n)``.

    Returns
    -------
    Q, R:
        ``Q`` has shape ``(m, m)`` and ``R`` has shape ``(m, n)``.
    """
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


def qr_wy(A: Array) -> tuple[Array, Array]:
    """Compute QR factorization using a compact WY-style representation.

    This version accumulates Householder reflectors in the form

        Q = I - W Y.T.

    The update exposes matrix-matrix operations and is therefore a useful
    stepping stone toward block QR algorithms.
    """
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
    """Compute a simple block QR factorization.

    The implementation applies QR factorization to panels of columns and updates
    the remaining trailing matrix. It is written for numerical experiments, not
    as a high-performance production implementation.
    """
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
    """Simulated mixed-FMA Householder QR with high-precision accumulation.

    This is the default mixed-precision model used for the dissertation-style
    experiments:

        low-precision operands + high-precision accumulation/storage.

    Software model
    --------------
    - The input matrix is first rounded to ``operand_dtype``; by default FP32.
    - The rounded operands are promoted to ``accumulator_dtype``; by default
      FP64.
    - The Householder QR computation is then carried out in the accumulator
      precision, and the returned Q and R factors are kept in that precision.

    This mirrors the mixed-FMA idea ``C = C + A * B`` where A and B are lower
    precision operands but C is a higher-precision accumulator. In NumPy this is
    a software-level simulation, not a hardware-level FMA instruction.

    Compared with ``householder_qr_mixed_fma_store32``, this version does not
    round the working matrix back to FP32 after each Householder update. It is
    therefore expected to be much closer to float64 QR than to pure float32 QR.
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
    """Conservative simulated mixed-FMA Householder QR with FP32 storage.

    Software model
    --------------
    - ``A`` and ``Q`` are stored in ``storage_dtype``; by default FP32.
    - At each Householder step, the active vector and matrix blocks are
      converted to ``accumulator_dtype``; by default FP64.
    - The reflector application, including inner products and matrix updates,
      is computed in accumulator precision.
    - The updated blocks are rounded back to ``storage_dtype`` before the next
      step.

    This is useful for studying a storage-limited mixed-precision model. It is
    usually less accurate than ``householder_qr_mixed_fma`` because every step
    is limited by FP32 storage rounding.
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
        }

    def run_single(self, m: int = 200, n: int = 100, dtype=np.float64) -> dict[str, dict[str, float]]:
        """Run one random QR experiment and return basic error metrics."""
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
