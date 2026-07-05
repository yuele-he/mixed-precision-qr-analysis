#include "config.hpp"
#include "dot.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct VectorPair
{
    std::vector<double> x;
    std::vector<double> y;
};

VectorPair generate_vector_pair(std::size_t n, unsigned seed_x, unsigned seed_y, const std::string& distribution)
{
    std::mt19937 gen_x(seed_x);
    std::mt19937 gen_y(seed_y);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::uniform_real_distribution<double> log_scale(-8.0, 8.0);

    VectorPair pair;
    pair.x.resize(n);
    pair.y.resize(n);

    if (distribution == "normal")
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            pair.x[i] = normal(gen_x);
            pair.y[i] = normal(gen_y);
        }
    }
    else if (distribution == "positive_uniform")
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            pair.x[i] = uniform01(gen_x);
            pair.y[i] = uniform01(gen_y);
        }
    }
    else if (distribution == "alternating_sign")
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const double sign = (i % 2 == 0) ? 1.0 : -1.0;
            pair.x[i] = sign * std::abs(normal(gen_x));
            pair.y[i] = std::abs(normal(gen_y));
        }
    }
    else if (distribution == "ill_scaled")
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const double sx = std::pow(10.0, log_scale(gen_x));
            const double sy = std::pow(10.0, log_scale(gen_y));
            pair.x[i] = normal(gen_x) * sx;
            pair.y[i] = normal(gen_y) * sy;
        }
    }
    else if (distribution == "nearly_canceling")
    {
        std::normal_distribution<double> normal(0.0, 1.0);

        const double epsilon = 1.0e-7;

        std::size_t i = 0;

        for (; i + 1 < n; i += 2)
        {
            const double a = normal(gen_x);
            const double b = normal(gen_y);
            const double noise = normal(gen_x);

            pair.x[i] = a;
            pair.y[i] = b;

            pair.x[i + 1] = a;
            pair.y[i + 1] = -b * (1.0 + epsilon * noise);
        }

        if (i < n)
        {
            pair.x[i] = 0.0;
            pair.y[i] = 0.0;
        }
    }
    else
    {
        throw std::runtime_error("Unsupported input distribution: " + distribution);
    }

    return pair;
}

template <typename Func>
double run_once(Func&& f, double& result)
{
    auto start = Clock::now();
    result = f();
    auto end = Clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double relative_error(double value, double reference)
{
    if (reference == 0.0)
    {
        return std::abs(value - reference);
    }
    return std::abs(value - reference) / std::abs(reference);
}

// Sum of magnitudes of the exact FP64 products.
// This is used to estimate the conditioning of a dot product:
//     kappa_dot = sum_i |x_i y_i| / |x^T y|.
// When cancellation makes |x^T y| small, kappa_dot becomes large.
double sum_abs_products_fp64(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("Vector size mismatch.");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sum += std::abs(x[i] * y[i]);
    }
    return sum;
}

double safe_ratio(double numerator, double denominator)
{
    if (denominator == 0.0)
    {
        if (numerator == 0.0)
        {
            return 0.0;
        }
        return std::numeric_limits<double>::infinity();
    }
    return numerator / denominator;
}

double dot_condition_number(double sum_abs_products, double reference)
{
    return safe_ratio(sum_abs_products, std::abs(reference));
}

double scaled_abs_error(double value, double reference, double sum_abs_products)
{
    return safe_ratio(std::abs(value - reference), sum_abs_products);
}

void write_threads(std::ofstream& out, bool is_serial, int threads)
{
    if (is_serial)
    {
        out << "";
    }
    else
    {
        out << threads;
    }
}

void run_runtime_benchmark(const ExperimentConfig& config, const std::string& runtime_csv)
{
    std::ofstream out(runtime_csv);
    if (!out)
    {
        throw std::runtime_error("Cannot open runtime output file: " + runtime_csv);
    }

    out << "experiment,input_distribution,method,precision,threads,n,trial,repeat,seed_x,seed_y,runtime,result\n";
    out << std::setprecision(17);

    for (const std::string& distribution : config.input_distributions)
    {
        for (std::size_t n : config.sizes)
        {
            for (int trial = 0; trial < config.runtime_trials; ++trial)
            {
                const unsigned seed_x = config.base_seed_x + static_cast<unsigned>(trial);
                const unsigned seed_y = config.base_seed_y + static_cast<unsigned>(trial);
                std::cout << "[runtime] distribution=" << distribution << ", n=" << n << ", trial=" << trial <<
                    std::endl;

                VectorPair pair = generate_vector_pair(n, seed_x, seed_y, distribution);

                for (int repeat = 0; repeat < config.runtime_repeats; ++repeat)
                {
                    double result = 0.0;
                    double runtime = run_once([&]() { return dot_serial_fp64(pair.x, pair.y); }, result);
                    out << "runtime," << distribution << ",serial,fp64,," << n << "," << trial << "," << repeat << ","
                        << seed_x << "," << seed_y << "," << runtime << "," << result << "\n";

                    runtime = run_once([&]() { return dot_serial_mixed_fp32_fp64(pair.x, pair.y); }, result);
                    out << "runtime," << distribution << ",serial,mixed_fp32_fp64,," << n << "," << trial << "," <<
                        repeat << ","
                        << seed_x << "," << seed_y << "," << runtime << "," << result << "\n";
                }

                for (int threads : config.thread_counts)
                {
                    omp_set_num_threads(threads);
                    for (int repeat = 0; repeat < config.runtime_repeats; ++repeat)
                    {
                        double result = 0.0;
                        double runtime = run_once([&]() { return dot_openmp_fp64(pair.x, pair.y); }, result);
                        out << "runtime," << distribution << ",openmp,fp64," << threads << "," << n << "," << trial <<
                            "," << repeat << ","
                            << seed_x << "," << seed_y << "," << runtime << "," << result << "\n";

                        runtime = run_once([&]() { return dot_openmp_mixed_fp32_fp64(pair.x, pair.y); }, result);
                        out << "runtime," << distribution << ",openmp,mixed_fp32_fp64," << threads << "," << n << "," <<
                            trial << "," << repeat << ","
                            << seed_x << "," << seed_y << "," << runtime << "," << result << "\n";
                    }
                }
            }
        }
    }
}

void run_error_analysis(const ExperimentConfig& config, const std::string& error_csv)
{
    std::ofstream out(error_csv);
    if (!out)
    {
        throw std::runtime_error("Cannot open error output file: " + error_csv);
    }

    out <<
        "experiment,input_distribution,method,precision,threads,n,trial,seed_x,seed_y,result,reference_fp64,reference_same_precision,abs_reference_fp64,abs_reference_same_precision,sum_abs_products,dot_condition_number_fp64,dot_condition_number_same_precision,abs_error_vs_serial_fp64,scaled_error_vs_sum_abs_products,scaled_parallel_error_vs_sum_abs_products,error_vs_serial_fp64,mixed_error_vs_fp64,parallel_error_vs_serial_same_precision\n";
    out << std::setprecision(17);

    for (const std::string& distribution : config.input_distributions)
    {
        for (std::size_t n : config.sizes)
        {
            for (int trial = 0; trial < config.error_trials; ++trial)
            {
                const unsigned seed_x = config.base_seed_x + 100000u + static_cast<unsigned>(trial);
                const unsigned seed_y = config.base_seed_y + 100000u + static_cast<unsigned>(trial);
                std::cout << "[error] distribution=" << distribution << ", n=" << n << ", trial=" << trial << std::endl;

                VectorPair pair = generate_vector_pair(n, seed_x, seed_y, distribution);
                const double reference_fp64 = dot_serial_fp64(pair.x, pair.y);
                const double reference_mixed = dot_serial_mixed_fp32_fp64(pair.x, pair.y);
                const double abs_reference_fp64 = std::abs(reference_fp64);
                const double abs_reference_mixed = std::abs(reference_mixed);
                const double sum_abs_products = sum_abs_products_fp64(pair.x, pair.y);
                const double condition_number_fp64 = dot_condition_number(sum_abs_products, reference_fp64);
                const double condition_number_mixed = dot_condition_number(sum_abs_products, reference_mixed);

                double value = reference_fp64;
                out << "error," << distribution << ",serial,fp64,," << n << "," << trial << "," << seed_x << "," <<
                    seed_y << ","
                    << value << "," << reference_fp64 << "," << reference_fp64 << "," << abs_reference_fp64 << "," <<
                    abs_reference_fp64 << ","
                    << sum_abs_products << "," << condition_number_fp64 << "," << condition_number_fp64 << ","
                    << 0.0 << "," << 0.0 << "," << 0.0 << ","
                    << 0.0 << ",," << 0.0 << "\n";

                value = reference_mixed;
                const double mixed_error = relative_error(value, reference_fp64);
                const double mixed_abs_error = std::abs(value - reference_fp64);
                const double mixed_scaled_error = scaled_abs_error(value, reference_fp64, sum_abs_products);
                out << "error," << distribution << ",serial,mixed_fp32_fp64,," << n << "," << trial << "," << seed_x <<
                    "," << seed_y << ","
                    << value << "," << reference_fp64 << "," << reference_mixed << "," << abs_reference_fp64 << "," <<
                    abs_reference_mixed << ","
                    << sum_abs_products << "," << condition_number_fp64 << "," << condition_number_mixed << ","
                    << mixed_abs_error << "," << mixed_scaled_error << "," << 0.0 << ","
                    << mixed_error << "," << mixed_error << "," << 0.0 << "\n";

                for (int threads : config.thread_counts)
                {
                    omp_set_num_threads(threads);

                    value = dot_openmp_fp64(pair.x, pair.y);
                    const double fp64_abs_error = std::abs(value - reference_fp64);
                    const double fp64_scaled_error = scaled_abs_error(value, reference_fp64, sum_abs_products);
                    const double fp64_relative_error = relative_error(value, reference_fp64);
                    out << "error," << distribution << ",openmp,fp64," << threads << "," << n << "," << trial << "," <<
                        seed_x << "," << seed_y << ","
                        << value << "," << reference_fp64 << "," << reference_fp64 << "," << abs_reference_fp64 << ","
                        << abs_reference_fp64 << ","
                        << sum_abs_products << "," << condition_number_fp64 << "," << condition_number_fp64 << ","
                        << fp64_abs_error << "," << fp64_scaled_error << "," << fp64_scaled_error << ","
                        << fp64_relative_error << ",," << fp64_relative_error << "\n";

                    value = dot_openmp_mixed_fp32_fp64(pair.x, pair.y);
                    const double openmp_mixed_abs_error_fp64 = std::abs(value - reference_fp64);
                    const double openmp_mixed_scaled_error_fp64 = scaled_abs_error(
                        value, reference_fp64, sum_abs_products);
                    const double openmp_mixed_scaled_parallel_error = scaled_abs_error(
                        value, reference_mixed, sum_abs_products);
                    const double openmp_mixed_relative_error_fp64 = relative_error(value, reference_fp64);
                    const double openmp_mixed_relative_parallel_error = relative_error(value, reference_mixed);
                    out << "error," << distribution << ",openmp,mixed_fp32_fp64," << threads << "," << n << "," << trial
                        << "," << seed_x << "," << seed_y << ","
                        << value << "," << reference_fp64 << "," << reference_mixed << "," << abs_reference_fp64 << ","
                        << abs_reference_mixed << ","
                        << sum_abs_products << "," << condition_number_fp64 << "," << condition_number_mixed << ","
                        << openmp_mixed_abs_error_fp64 << "," << openmp_mixed_scaled_error_fp64 << "," <<
                        openmp_mixed_scaled_parallel_error << ","
                        << openmp_mixed_relative_error_fp64 << "," << openmp_mixed_relative_error_fp64 << "," <<
                        openmp_mixed_relative_parallel_error << "\n";
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    std::string config_path = "cpp_hpc/config.json";
    if (argc >= 2)
    {
        config_path = argv[1];
    }

    try
    {
        ExperimentConfig config = load_config(config_path);
        std::filesystem::create_directories(config.results_dir);

        const std::string runtime_csv = join_path(config.results_dir, config.runtime_results_file);
        const std::string error_csv = join_path(config.results_dir, config.error_results_file);

        std::cout << "Loaded config: " << config_path << std::endl;
        std::cout << "Results directory: " << config.results_dir << std::endl;
        std::cout << "Runtime CSV: " << runtime_csv << std::endl;
        std::cout << "Error CSV: " << error_csv << std::endl;

        run_runtime_benchmark(config, runtime_csv);
        run_error_analysis(config, error_csv);

        std::cout << "Done. Results written to " << config.results_dir << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
