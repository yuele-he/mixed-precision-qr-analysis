# src/qr_decomp.py
import numpy as np
from core.plotter import plot_qr_metrics_bar, plot_qr_metrics_line


def householder_vector(x):
    """
    Compute Householder vector v and scalar beta such that:
    P = I - beta * v v.T, and Px = ||x|| * e1

    Parameters:
        x: numpy array of shape (m,)

    Returns:
        v: Householder vector (unit direction)
        beta: scalar
    """
    x = x.reshape(-1, 1)  # ensure column vector
    sigma = np.dot(x[1:].T, x[1:])[0, 0]
    v = x.copy()
    if sigma == 0 and x[0] >= 0:
        beta = 0.0
    elif sigma == 0 and x[0] < 0:
        beta = -2.0
    else:
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2.0 * v[0] ** 2 / (sigma + v[0] ** 2)
        v = v / v[0]  # normalize so that v[0] == 1

    return v, beta

def householder_qr(A):
    """
    Perform QR decomposition using Householder reflections.

    Parameters:
        A: (m x n) numpy array

    Returns:
        Q: orthogonal matrix (m x m)
        R: upper triangular matrix (m x n)
    """
    A = A.copy()
    m, n = A.shape
    Q = np.eye(m)
    for k in range(min(m, n)):
        # Select vector x from column k below the diagonal
        x = A[k:, k]
        v, beta = householder_vector(x)
        v = v.reshape(-1, 1)

        # Apply Householder transformation to A[k:, k:]
        A_k = A[k:, k:]
        A[k:, k:] = A_k - beta * v @ (v.T @ A_k)

        # Apply transformation to Q (construct Q from I via reflections)
        Q_k = Q[:, k:]
        Q[:, k:] = Q_k - Q_k @ v @ (beta * v.T)

    R = A
    return Q, R

def qr_wy(A):
    """
    QR factorization using WY representation: Q = I - W Yᵗ

    Returns:
        Q (m x m), R (m x n)
    """
    A = A.copy()
    m, n = A.shape
    Y = []
    W = []

    for j in range(n):
        x = A[j:, j]
        v_hat, beta = householder_vector(x)

        # Apply reflection to A[j:, j:]
        v_hat = v_hat.reshape(-1, 1)
        A[j:, j:] -= beta * v_hat @ (v_hat.T @ A[j:, j:])

        # Build full-size v
        v_full = np.zeros((m, 1))
        v_full[j:] = v_hat

        if j == 0:
            Y = v_full
            W = beta * v_full
        else:
            # Compute z = β v - β W (Yᵗ v)
            YTV = Y.T @ v_full
            z = beta * v_full - beta * W @ YTV
            Y = np.hstack((Y, v_full))
            W = np.hstack((W, z))

    Q = np.eye(m) - W @ Y.T
    R = np.triu(A)
    return Q, R

def block_qr(A, block_size=64):
    """
    Block-wise QR decomposition (not parallelized, but structured).

    Parameters:
        A: (m x n) matrix
        block_size: column block size (b)

    Returns:
        Q: (m x m)
        R: (m x n)
    """
    A = A.copy()
    m, n = A.shape
    Q_total = np.eye(m)

    for i in range(0, n, block_size):
        b = min(block_size, n - i)
        A11 = A[i:, i:i+b]

        # QR on the current block
        Qi, Ri = qr_wy(A11)

        # Apply to current and remaining columns
        A[i:, i:] = Qi.T @ A[i:, i:]
        Q_block = np.eye(m)
        Q_block[i:, i:] = Qi
        Q_total = Q_total @ Q_block  # Accumulate global Q

    R = A
    return Q_total, R

# -------------------- #
# QRExperiment 类封装 #
# -------------------- #

class QRExperiment:
    """
    Run QR decomposition experiments with multiple methods,
    supporting single test or parameter sweep.
    """

    def __init__(self, dtypes=[np.float64], sizes=[(200, 100)], block_size=64):
        """
        Parameters
        ----------
        dtypes : list of np.dtype
            Data types to test (e.g. [np.float32, np.float64])
        sizes : list of tuple
            Matrix sizes to test (list of (m, n))
        block_size : int
            Block size for block_qr
        """
        self.dtypes = dtypes
        self.sizes = sizes
        self.block_size = block_size
        self.methods = {
            "householder": householder_qr,
            "wy": qr_wy,
            "block": block_qr
        }

    def run_single(self, m=200, n=100, dtype=np.float64):
        """
        Run QR decomposition once on a random matrix.

        Returns
        -------
        dict : { method: {"residual": float, "orthogonality": float} }
        """
        A = np.random.randn(m, n).astype(dtype)
        b = np.random.uniform(1, 2, size=(m, 1)).astype(dtype)

        results = {}
        for name, fn in self.methods.items():
            Q, R = fn(A)
            residual = np.linalg.norm(A - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
            orthogonality = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])) / np.linalg.norm(Q)
            rhs = Q.T @ b
            x = np.linalg.lstsq(R, rhs, rcond=None)[0]
            ls = np.linalg.norm(R @ x - Q.T @ b) / np.linalg.norm(b)
            results[name] = {
                "residual": residual,
                "orthogonality": orthogonality,
                "least_squares": ls
            }
        return results

    def run_sweep(self):
        """
        Run QR decomposition for all sizes and dtypes.

        Returns
        -------
        dict :
            {
                "float32": {
                    "householder": {"sizes": [...], "residuals": [...], "orthogonality": [...]},
                    "wy": {...},
                    "block": {...}
                },
                "float64": {...}
            }
        """
        sweep_results = {}

        for dtype in self.dtypes:
            dtype_name = np.dtype(dtype).name

            sweep_results[dtype_name] = {
                name: {"sizes": [], "residuals": [], "orthogonality": []}
                for name in self.methods
            }

            for (m, n) in self.sizes:
                A = np.random.randn(m, n).astype(dtype)

                for name, fn in self.methods.items():
                    Q, R = fn(A)

                    residual = np.linalg.norm(Q @ R - A, ord="fro") / np.linalg.norm(A, ord="fro")
                    orthogonality = (
                            np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), ord="fro")
                            / np.linalg.norm(Q, ord="fro")
                    )

                    sweep_results[dtype_name][name]["sizes"].append(m)
                    sweep_results[dtype_name][name]["residuals"].append(float(residual))
                    sweep_results[dtype_name][name]["orthogonality"].append(float(orthogonality))

        return sweep_results

if __name__ == "__main__":
    exp = QRExperiment()
    single_res = exp.run_single(m=200, n=100, dtype=np.float64)
    plot_qr_metrics_bar(single_res, title="Single QR Decomposition Test")

    # sizes = [(100, 50), (200, 100), (400, 200)]
    # exp = QRExperiment(dtypes=[np.float32, np.float64], sizes=sizes, block_size=32)
    # sweep_res = exp.run_sweep()
    # plot_qr_metrics_line(sweep_res, title="QR Decomposition Error vs Size")