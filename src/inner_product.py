import logging

import numpy as np
from matplotlib import pyplot as plt
from mpmath import mp

try:  # Works when running from the project root as a package.
    from src.core.base_experiment import BaseExperiment
    from src.core.plotter import plot_error_vs_size
    from src.mixed_kernels import (
        dot_low_inputs_high_fma,
        dot_low_product_high_accumulation,
    )
except ImportError:  # Works when running files directly from inside src/.
    from core.base_experiment import BaseExperiment
    from core.plotter import plot_error_vs_size
    from mixed_kernels import (
        dot_low_inputs_high_fma,
        dot_low_product_high_accumulation,
    )

logger = logging.getLogger(__name__)
mp.dps = 50  # set mpmath to 50-digit precision


class InnerProductExperiment(BaseExperiment):
    """
    Experiment to evaluate the absolute error of inner product computations
    under different floating-point precisions.

    Attributes
    ----------
    value_range : tuple
        Range (min, max) for random input vector elements.
    """

    def __init__(self, dtype_list, sizes, repeats, value_range=(0, 2)):
        super().__init__(dtype_list, sizes, repeats)
        self.value_range = value_range

    def compute_reference_inner_product(self, a, b, dtype):
        """
        Compute a reference inner product using higher precision.
        """
        if dtype in (np.float16, np.float32):
            return np.dot(a.astype(np.float64), b.astype(np.float64))
        return float(sum(mp.mpf(x) * mp.mpf(y) for x, y in zip(a, b)))

    def run_single_size(self, size, dtype):
        """
        Run one inner product computation and return absolute error.
        """
        a = np.random.uniform(*self.value_range, size=size)
        b = np.random.uniform(*self.value_range, size=size)

        a_low = a.astype(dtype)
        b_low = b.astype(dtype)

        approx = np.dot(a_low, b_low)
        exact = self.compute_reference_inner_product(a, b, dtype)
        return abs(approx - exact)


class InnerProductExperimentMixed(InnerProductExperiment):
    """
    Mixed-precision inner product experiment.

    Main model:
        low-precision operands + low-precision products + high-precision
        accumulation.

    By default, ``dtype`` controls the low precision used for operands and
    products, while accumulation is performed in FP64. This matches the
    dissertation-style software simulation more closely than the FMA model.
    """

    def __init__(
        self,
        dtype_list,
        sizes,
        repeats,
        value_range=(0, 2),
        *,
        accumulator_dtype=np.float64,
    ):
        super().__init__(dtype_list, sizes, repeats, value_range)
        self.accumulator_dtype = accumulator_dtype

    def run_single_size(self, size, dtype):
        a = np.random.uniform(*self.value_range, size=size)
        b = np.random.uniform(*self.value_range, size=size)

        approx = dot_low_product_high_accumulation(
            a,
            b,
            operand_dtype=dtype,
            product_dtype=dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        exact = self.compute_reference_inner_product(a, b, dtype)
        return abs(approx - exact)


class InnerProductExperimentFMA(InnerProductExperiment):
    """
    Experimental low-input/high-FMA inner product experiment.

    This model first rounds operands to ``dtype`` and then uses Python
    ``math.fma`` for FP64 fused multiply-add accumulation. It is included as an
    alternative model and should not be described as low-precision product with
    high-precision accumulation.
    """

    def __init__(
        self,
        dtype_list,
        sizes,
        repeats,
        value_range=(0, 2),
        *,
        accumulator_dtype=np.float64,
    ):
        super().__init__(dtype_list, sizes, repeats, value_range)
        self.accumulator_dtype = accumulator_dtype

    def run_single_size(self, size, dtype):
        a = np.random.uniform(*self.value_range, size=size)
        b = np.random.uniform(*self.value_range, size=size)

        approx = dot_low_inputs_high_fma(
            a,
            b,
            operand_dtype=dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        exact = self.compute_reference_inner_product(a, b, dtype)
        return abs(approx - exact)


if __name__ == "__main__":
    dtype_list = [np.float32, np.float64]
    sizes = [int(1.5 ** i) for i in range(2, 10)]
    repeats = 100
    value_range = (0, 2)

    exp = InnerProductExperiment(dtype_list, sizes, repeats, value_range)
    exp.run()

    fig, ax = plt.subplots(figsize=(8, 6))

    for dtype in dtype_list:
        error_matrix = np.array(exp.results[dtype])
        plot_error_vs_size(
            sizes=sizes,
            error_matrix=error_matrix,
            dtype=dtype,
            label=dtype.__name__,
            color="royalblue" if dtype == np.float32 else "darkorange",
            ax=ax,
        )

    ax.set_title("Inner Product Error over Vector Length")
    plt.tight_layout()
    plt.show()
