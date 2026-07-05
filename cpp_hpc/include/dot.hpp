#ifndef DOT_HPP
#define DOT_HPP

#include <vector>

double dot_serial_fp64(const std::vector<double>& x, const std::vector<double>& y);
double dot_openmp_fp64(const std::vector<double>& x, const std::vector<double>& y);
double dot_serial_mixed_fp32_fp64(const std::vector<double>& x, const std::vector<double>& y);
double dot_openmp_mixed_fp32_fp64(const std::vector<double>& x, const std::vector<double>& y);

#endif
