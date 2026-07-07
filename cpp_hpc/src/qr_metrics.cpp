#include "qr.hpp"

#include <cmath>
#include <stdexcept>

double relative_residual_fro(const std::vector<double>& A,
                             const QRResult& qr)
{
    const std::size_t m = qr.m;
    const std::size_t n = qr.n;

    if (A.size() != m * n)
    {
        throw std::invalid_argument("A size does not match QRResult dimensions.");
    }

    double residual_sq = 0.0;
    double norm_a_sq = 0.0;

    // Compute A - Q R.
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            double qr_ij = 0.0;

            for (std::size_t k = 0; k < n; ++k)
            {
                qr_ij += qr.Q[cm_idx(i, k, m)] * qr.R[cm_idx(k, j, n)];
            }

            const double a_ij = A[cm_idx(i, j, m)];
            const double diff = a_ij - qr_ij;

            residual_sq += diff * diff;
            norm_a_sq += a_ij * a_ij;
        }
    }

    if (norm_a_sq == 0.0)
    {
        return std::sqrt(residual_sq);
    }

    return std::sqrt(residual_sq) / std::sqrt(norm_a_sq);
}

double orthogonality_error_fro(const QRResult& qr)
{
    const std::size_t m = qr.m;
    const std::size_t n = qr.n;

    double error_sq = 0.0;

    // Compute ||I - Q^T Q||_F.
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            double dot = 0.0;

            for (std::size_t r = 0; r < m; ++r)
            {
                dot += qr.Q[cm_idx(r, i, m)] * qr.Q[cm_idx(r, j, m)];
            }

            const double target = (i == j) ? 1.0 : 0.0;
            const double diff = target - dot;
            error_sq += diff * diff;
        }
    }

    return std::sqrt(error_sq);
}
