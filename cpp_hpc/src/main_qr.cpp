#include "matrix_utils.hpp"
#include "qr.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
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

} // namespace

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

    {
        const std::size_t m = 8;
        const std::size_t n = 4;
        auto A = make_nearly_dependent_matrix(m, n, 4, 1e-8);
        run_case("nearly_dependent_8x4", A, m, n);
    }

    return 0;
}
