#pragma once

#include <cstddef>
#include <vector>

// Generate an m x n dense matrix in column-major order.
std::vector<double> make_random_matrix(std::size_t m,
                                       std::size_t n,
                                       unsigned int seed);

// Generate the first n columns of I_m, stored as an m x n matrix.
std::vector<double> make_identity_like_matrix(std::size_t m,
                                              std::size_t n);

// Generate an m x n matrix with nearly linearly dependent columns.
// Each column is a scaled copy of one base column plus Gaussian noise.
std::vector<double> make_nearly_dependent_matrix(std::size_t m,
                                                 std::size_t n,
                                                 unsigned int seed,
                                                 double perturbation);
