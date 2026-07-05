"""Reusable mixed-precision arithmetic kernels.

This module contains low-level dot-product kernels used by both the
inner-product experiments and QR factorization experiments.  The functions are
written with explicit Python/NumPy loops so that the simulated rounding model is
clear and configurable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

Array = np.ndarray
DTypeLike = Any


def _as_floating_array(x: Any, dtype: DTypeLike) -> Array:
    """Return ``x`` as a one-dimensional floating-point NumPy array.

    Parameters
    ----------
    x:
        Input vector-like object.
    dtype:
        Target floating-point dtype, for example ``np.float32`` or
        ``np.float64``.
    """
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype must be a floating-point type, got {dtype}.")
    return np.asarray(x, dtype=dtype).ravel()


def _check_same_shape(a: Array, b: Array) -> None:
    """Raise an informative error if two vectors have different shapes."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def dot_fp64(a: Any, b: Any) -> np.float64:
    """Compute an explicit-loop FP64 inner product.

    This is mainly used as a simple high-precision baseline inside the software
    experiments. For final validation, an arbitrary-precision reference such as
    ``mpmath`` may still be preferable.
    """
    a64 = _as_floating_array(a, np.float64)
    b64 = _as_floating_array(b, np.float64)
    _check_same_shape(a64, b64)

    acc = np.float64(0.0)
    for i in range(a64.size):
        acc = np.float64(acc + np.float64(a64[i] * b64[i]))
    return np.float64(acc)


def dot_fp32(a: Any, b: Any) -> np.float32:
    """Compute an explicit-loop FP32 inner product.

    Model:
        FP32 operands + FP32 products + FP32 accumulation.
    """
    a32 = _as_floating_array(a, np.float32)
    b32 = _as_floating_array(b, np.float32)
    _check_same_shape(a32, b32)

    acc = np.float32(0.0)
    for i in range(a32.size):
        prod = np.float32(a32[i] * b32[i])
        acc = np.float32(acc + prod)
    return np.float32(acc)


def dot_low_product_high_accumulation(
    a: Any,
    b: Any,
    *,
    operand_dtype: DTypeLike = np.float32,
    product_dtype: DTypeLike = np.float32,
    accumulator_dtype: DTypeLike = np.float64,
) -> np.floating:
    """Mixed-precision dot product with configurable precision levels.

    Arithmetic model:

        a_low    = fl_operand(a)
        b_low    = fl_operand(b)
        prod_low = fl_product(a_low[i] * b_low[i])
        acc      = fl_accumulator(acc + prod_low)

    Defaults:
        FP32 operands + FP32 products + FP64 accumulation.

    This is the main software-simulated mixed-precision model for the thesis
    extension. It is deliberately different from a fused multiply-add model:
    the product is rounded to ``product_dtype`` before it is accumulated.
    """
    operand_dtype = np.dtype(operand_dtype)
    product_dtype = np.dtype(product_dtype)
    accumulator_dtype = np.dtype(accumulator_dtype)

    if not np.issubdtype(product_dtype, np.floating):
        raise TypeError(f"product_dtype must be floating point, got {product_dtype}.")
    if not np.issubdtype(accumulator_dtype, np.floating):
        raise TypeError(
            f"accumulator_dtype must be floating point, got {accumulator_dtype}."
        )

    a_low = _as_floating_array(a, operand_dtype)
    b_low = _as_floating_array(b, operand_dtype)
    _check_same_shape(a_low, b_low)

    acc = accumulator_dtype.type(0.0)
    for i in range(a_low.size):
        prod_low = product_dtype.type(a_low[i] * b_low[i])
        acc = accumulator_dtype.type(acc + accumulator_dtype.type(prod_low))
    return acc


def dot_low_inputs_high_fma(
    a: Any,
    b: Any,
    *,
    operand_dtype: DTypeLike = np.float32,
    accumulator_dtype: DTypeLike = np.float64,
) -> np.float64:
    """Experimental model: low-precision inputs with FP64 FMA accumulation.

    Arithmetic model:

        a_low = fl_operand(a)
        b_low = fl_operand(b)
        acc   = fl_64(a_low[i] * b_low[i] + acc)

    Important:
        Python ``math.fma`` works on Python floats, which are double precision
        in ordinary CPython builds. Therefore this function rounds inputs to
        low precision first, but the multiply-add itself is a fused FP64
        operation. It is NOT the same as low-precision product followed by
        high-precision accumulation.
    """
    accumulator_dtype = np.dtype(accumulator_dtype)
    if accumulator_dtype != np.dtype(np.float64):
        raise NotImplementedError(
            "dot_low_inputs_high_fma currently supports only FP64 accumulation, "
            "because Python math.fma operates on Python float values."
        )

    a_low = _as_floating_array(a, operand_dtype)
    b_low = _as_floating_array(b, operand_dtype)
    _check_same_shape(a_low, b_low)

    acc = 0.0
    for i in range(a_low.size):
        acc = math.fma(float(a_low[i]), float(b_low[i]), acc)
    return np.float64(acc)
