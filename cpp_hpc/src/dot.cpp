#include "dot.hpp"

#include <omp.h>
#include <cstddef>
#include <stdexcept>

static void check_size(std::size_t n, std::size_t m)
{
    if (n != m)
    {
        throw std::invalid_argument("Vector size mismatch.");
    }
}

double dot_serial_fp64(const std::vector<double>& x, const std::vector<double>& y)
{
    check_size(x.size(), y.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

double dot_openmp_fp64(const std::vector<double>& x, const std::vector<double>& y)
{
    check_size(x.size(), y.size());
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
    {
        sum += x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
    }
    return sum;
}

double dot_serial_mixed_fp32_fp64(const std::vector<double>& x, const std::vector<double>& y)
{
    check_size(x.size(), y.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        float xi = static_cast<float>(x[i]);
        float yi = static_cast<float>(y[i]);
        float prod_low = xi * yi;
        sum += static_cast<double>(prod_low);
    }
    return sum;
}

double dot_openmp_mixed_fp32_fp64(const std::vector<double>& x, const std::vector<double>& y)
{
    check_size(x.size(), y.size());
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
    {
        const std::size_t idx = static_cast<std::size_t>(i);
        float xi = static_cast<float>(x[idx]);
        float yi = static_cast<float>(y[idx]);
        float prod_low = xi * yi;
        sum += static_cast<double>(prod_low);
    }
    return sum;
}
