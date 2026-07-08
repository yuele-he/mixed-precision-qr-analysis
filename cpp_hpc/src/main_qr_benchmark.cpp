#include "matrix_utils.hpp"
#include "qr.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct BenchmarkCase
{
    std::string matrix_type;
    std::size_t m;
    std::size_t n;
    double perturbation = 0.0;
};

    std::vector<double> make_matrix_for_case(const BenchmarkCase& test_case,
                                             unsigned int seed)
    {
        if (test_case.matrix_type == "nearly_dependent")
        {
            return make_nearly_dependent_matrix(
                test_case.m,
                test_case.n,
                seed,
                test_case.perturbation
            );
        }

        return make_random_matrix(test_case.m, test_case.n, seed);
    }
} // namespace

int main()
{
    std::cout << "Serial Householder QR benchmark\n";
    std::cout << "================================\n";

    const int trials = 3;

    std::vector<BenchmarkCase> cases;

    // Square matrices: n x n.
    for (std::size_t n : {16, 32, 64, 128, 256, 384})
    {
        cases.push_back({"square", n, n});
    }

    // Tall-skinny matrices: m = 2n.
    for (std::size_t n : {16, 32, 64, 128, 256})
    {
        cases.push_back({"tall_skinny", 2 * n, n});
    }

    // Stress cases for numerical-error diagnostics.
    for (std::size_t n : {16, 32, 64, 128})
    {
        cases.push_back({"nearly_dependent", 2 * n, n, 1e-8});
    }

    const std::filesystem::path output_dir = "results/qr";
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path output_file = output_dir / "qr_serial_results.csv";
    std::ofstream csv(output_file);

    if (!csv.is_open())
    {
        throw std::runtime_error("Failed to open output CSV file: " + output_file.string());
    }

    csv << "matrix_type,m,n,trial,runtime_seconds,relative_residual,orthogonality_error,status\n";

    for (const auto& test_case : cases)
    {
        for (int trial = 0; trial < trials; ++trial)
        {
            const unsigned int seed = static_cast<unsigned int>(
                1000 + 97 * trial + test_case.m + 13 * test_case.n
            );

            auto A = make_matrix_for_case(test_case, seed);

            const auto start = std::chrono::high_resolution_clock::now();
            QRResult qr = householder_qr_serial(A, test_case.m, test_case.n);
            const auto end = std::chrono::high_resolution_clock::now();

            const double runtime_seconds = std::chrono::duration<double>(end - start).count();
            const double residual = relative_residual_fro(A, qr);
            const double orth_error = orthogonality_error_fro(qr);

            const bool pass = (residual < 1e-10) && (orth_error < 1e-10);
            const std::string status = pass ? "PASS" : "CHECK";

            csv << test_case.matrix_type << ","
                << test_case.m << ","
                << test_case.n << ","
                << trial << ","
                << std::setprecision(16) << runtime_seconds << ","
                << std::scientific << std::setprecision(8) << residual << ","
                << std::scientific << std::setprecision(8) << orth_error << ","
                << status << "\n";

            std::cout << std::left << std::setw(18) << test_case.matrix_type
                << " m=" << std::setw(5) << test_case.m
                << " n=" << std::setw(5) << test_case.n
                << " trial=" << trial
                << " runtime=" << std::scientific << std::setprecision(4)
                << runtime_seconds
                << " residual=" << residual
                << " orth=" << orth_error
                << " " << status
                      << "\n";
        }
    }

    csv.close();

    std::cout << "--------------------------------\n";
    std::cout << "Saved results to: " << output_file << "\n";

    return 0;
}
