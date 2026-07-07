#pragma once

#include <cstddef>
#include <vector>

struct QRResult
{
    std::size_t m;
    std::size_t n;

    // Column-major storage.
    // Q is m x n reduced Q.
    // R is n x n upper triangular.
    std::vector<double> Q;
    std::vector<double> R;
};

inline std::size_t cm_idx(std::size_t i, std::size_t j, std::size_t rows)
{
    return i + j * rows;
}

QRResult householder_qr_serial(const std::vector<double>& A,
                               std::size_t m,
                               std::size_t n);

double relative_residual_fro(const std::vector<double>& A,
                             const QRResult& qr);

double orthogonality_error_fro(const QRResult& qr);
