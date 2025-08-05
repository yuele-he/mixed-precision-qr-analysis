import numpy as np
import logging

from matplotlib import pyplot as plt
from mpmath import mp

from core.plotter import plot_error_vs_size
from src.core.base_experiment import BaseExperiment

logger = logging.getLogger(__name__)
mp.dps = 50  # set mpmath to 50-digit precision


class InnerProductExperiment(BaseExperiment):
    """
    Experiment to evaluate the relative error of inner product computations
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
        Compute ground truth inner product using extended precision.

        Parameters
        ----------
        a, b : ndarray
            Input vectors.
        dtype : np.dtype
            Precision used; determines whether to use float64 or mpmath.

        Returns
        -------
        float
            Ground truth dot product.
        """
        if dtype in (np.float16, np.float32):
            return np.dot(a.astype(np.float64), b.astype(np.float64))
        else:
            return float(sum(mp.mpf(x) * mp.mpf(y) for x, y in zip(a, b)))

    def run_single_size(self, size, dtype):
        """
        Run one inner product computation and return relative error.

        Parameters
        ----------
        size : int
            Length of input vectors.
        dtype : np.dtype
            Precision for low-precision computation.

        Returns
        -------
        float
            Relative error between low-precision and high-precision result.
        """
        a = np.random.uniform(*self.value_range, size=size)
        b = np.random.uniform(*self.value_range, size=size)

        a_low = a.astype(dtype)
        b_low = b.astype(dtype)

        approx = np.dot(a_low, b_low)
        exact = self.compute_reference_inner_product(a, b, dtype)
        abs_error = abs(approx - exact)
        return abs_error


if __name__ == "__main__":
    dtype_list = [np.float32, np.float64]
    sizes = [int(1.5 ** i) for i in range(2, 10)]  # e.g., 16 to 8192
    repeats = 100
    value_range = (0, 2)

    # Run experiment
    exp = InnerProductExperiment(dtype_list, sizes, repeats, value_range)
    exp.run()

    fig, ax = plt.subplots(figsize=(8, 6))

    for dtype in dtype_list:
        error_matrix = np.array(exp.results[dtype])  # shape: [repeats, len(sizes)]
        plot_error_vs_size(
            sizes=sizes,
            error_matrix=error_matrix,
            dtype=dtype,
            label=dtype.__name__,
            color="royalblue" if dtype == np.float32 else "darkorange",
            ax=ax
        )

    ax.set_title("Inner Product Error over Vector Length")
    plt.tight_layout()
    plt.show()
