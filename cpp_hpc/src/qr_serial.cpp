#include "qr.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

QRResult householder_qr_serial(const std::vector<double>& A,
                               std::size_t m,
                               std::size_t n)
{
    if (m < n)
    {
        throw std::invalid_argument("householder_qr_serial requires m >= n.");
    }
    if (A.size() != m * n)
    {
        throw std::invalid_argument("Input matrix size does not match m*n.");
    }

    // Working copy of A. This will become R in its upper triangular part.
    std::vector<double> R_work = A;

    // Full Q, initialized as I_m.
    // We build Q = H_0 H_1 ... H_{n-1}.
    std::vector<double> Q_full(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i)
    {
        Q_full[cm_idx(i, i, m)] = 1.0;
    }

    for (std::size_t k = 0; k < n; ++k)
    {
        const std::size_t len = m - k;

        // x = R_work[k:m, k]
        std::vector<double> v(len, 0.0);

        double norm_x_sq = 0.0;
        for (std::size_t t = 0; t < len; ++t)
        {
            const double val = R_work[cm_idx(k + t, k, m)];
            v[t] = val;
            norm_x_sq += val * val;
        }

        const double norm_x = std::sqrt(norm_x_sq);
        if (norm_x == 0.0)
        {
            continue;
        }

        // Stable Householder choice.
        // alpha = -sign(x0) * ||x||
        const double x0 = v[0];
        const double alpha = (x0 >= 0.0) ? -norm_x : norm_x;

        // v = x - alpha e1
        v[0] -= alpha;

        double v_norm_sq = 0.0;
        for (double val : v)
        {
            v_norm_sq += val * val;
        }

        if (v_norm_sq == 0.0)
        {
            continue;
        }

        const double beta = 2.0 / v_norm_sq;

        // Apply H = I - beta v v^T to R_work[k:m, k:n] from the left.
        for (std::size_t j = k; j < n; ++j)
        {
            double dot = 0.0;
            for (std::size_t t = 0; t < len; ++t)
            {
                dot += v[t] * R_work[cm_idx(k + t, j, m)];
            }

            for (std::size_t t = 0; t < len; ++t)
            {
                R_work[cm_idx(k + t, j, m)] -= beta * v[t] * dot;
            }
        }

        // Clean tiny numerical values below the diagonal in the active column.
        for (std::size_t i = k + 1; i < m; ++i)
        {
            R_work[cm_idx(i, k, m)] = 0.0;
        }

        // Accumulate Q = Q * H.
        // Since H only acts on coordinates k:m, update columns k:m of Q_full.
        for (std::size_t i = 0; i < m; ++i)
        {
            double dot = 0.0;
            for (std::size_t t = 0; t < len; ++t)
            {
                dot += Q_full[cm_idx(i, k + t, m)] * v[t];
            }

            for (std::size_t t = 0; t < len; ++t)
            {
                Q_full[cm_idx(i, k + t, m)] -= beta * dot * v[t];
            }
        }
    }

    // Extract reduced Q: first n columns of Q_full.
    std::vector<double> Q(m * n, 0.0);
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            Q[cm_idx(i, j, m)] = Q_full[cm_idx(i, j, m)];
        }
    }

    // Extract R: n x n upper triangular.
    std::vector<double> R(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i <= j && i < n; ++i)
        {
            R[cm_idx(i, j, n)] = R_work[cm_idx(i, j, m)];
        }
    }

    return QRResult{m, n, std::move(Q), std::move(R)};
}
