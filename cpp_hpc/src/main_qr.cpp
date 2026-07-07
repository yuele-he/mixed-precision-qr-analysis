#include "qr.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<double> make_random_matrix(std::size_t m,
                                       std::size_t n,
                                       unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A(m * n, 0.0);

    // Column-major.
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
    std::vector<double> A(m * n, 0.0);

    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            A[cm_idx(i, j, m)] = (i == j) ? 1.0 : 0.0;
        }
    }

    return A;
}

void run_case(const std::string& name,
              const std::vector<double>& A,
              std::size_t m,
              std::size_t n)
{
    QRResult qr = householder_qr_serial(A, m, n);

    const double residual = relative_residual_fro(A, qr);
    const double orth_error = orthogonality_error_fro(qr);

    std::cout << "case: " << name << "\n";
    std::cout << "m=" << m << ", n=" << n << "\n";
    std::cout << "relative residual:   " << std::scientific << std::setprecision(6)
        << residual << "\n";
    std::cout << "orthogonality error: " << std::scientific << std::setprecision(6)
        << orth_error << "\n";

    const bool pass = (residual < 1e-12) && (orth_error < 1e-12);
    std::cout << "status: " << (pass ? "PASS" : "CHECK") << "\n";
    std::cout << "----------------------------------------\n";
}

int main()
{
    std::cout << "Serial Householder QR correctness test\n";
    std::cout << "Column-major storage, reduced QR, m >= n\n";
    std::cout << "========================================\n";

    {
        const std::size_t m = 3;
        const std::size_t n = 2;
        auto A = make_random_matrix(m, n, 1);
        run_case("random_3x2", A, m, n);
    }

    {
        const std::size_t m = 4;
        const std::size_t n = 3;
        auto A = make_random_matrix(m, n, 2);
        run_case("random_4x3", A, m, n);
    }

    {
        const std::size_t m = 8;
        const std::size_t n = 4;
        auto A = make_random_matrix(m, n, 3);
        run_case("random_8x4", A, m, n);
    }

    {
        const std::size_t m = 6;
        const std::size_t n = 4;
        auto A = make_identity_like_matrix(m, n);
        run_case("identity_like_6x4", A, m, n);
    }

    return 0;
}
