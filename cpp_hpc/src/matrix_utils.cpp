#include "matrix_utils.hpp"
#include "qr.hpp"

#include <random>
#include <stdexcept>
#include <vector>

std::vector<double> make_random_matrix(std::size_t m,
                                       std::size_t n,
                                       unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A(m * n, 0.0);

    // Column-major storage.
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            A[cm_idx(i, j, m)] = dist(gen);
        }
    }

    return A;
}

std::vector<double> make_identity_like_matrix(std::size_t m,
                                              std::size_t n)
{
    if (m < n)
    {
        throw std::invalid_argument("make_identity_like_matrix requires m >= n.");
    }

    std::vector<double> A(m * n, 0.0);

    for (std::size_t j = 0; j < n; ++j)
    {
        A[cm_idx(j, j, m)] = 1.0;
    }

    return A;
}

std::vector<double> make_nearly_dependent_matrix(std::size_t m,
                                                 std::size_t n,
                                                 unsigned int seed,
                                                 double perturbation)
{
    if (m < n)
    {
        throw std::invalid_argument("make_nearly_dependent_matrix requires m >= n.");
    }
    if (perturbation <= 0.0)
    {
        throw std::invalid_argument("perturbation must be positive.");
    }

    std::mt19937 gen(seed);
    std::normal_distribution<double> base_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, perturbation);

    std::vector<double> A(m * n, 0.0);
    std::vector<double> base_col(m, 0.0);

    for (std::size_t i = 0; i < m; ++i)
    {
        base_col[i] = base_dist(gen);
    }

    for (std::size_t j = 0; j < n; ++j)
    {
        const double scale = 1.0 + 0.1 * static_cast<double>(j);

        for (std::size_t i = 0; i < m; ++i)
        {
            A[cm_idx(i, j, m)] = scale * base_col[i] + noise_dist(gen);
        }
    }

    return A;
}
