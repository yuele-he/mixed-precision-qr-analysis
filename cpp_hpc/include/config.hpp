#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>

struct ExperimentConfig
{
    std::vector<std::size_t> sizes;
    std::vector<int> thread_counts;
    int runtime_trials = 1;
    int runtime_repeats = 20;
    int error_trials = 20;
    unsigned base_seed_x = 123;
    unsigned base_seed_y = 456;
    std::vector<std::string> input_distributions;
    std::string plot_input_distribution = "normal";
    std::string results_dir = "cpp_hpc/results";
    std::string runtime_results_file = "dot_runtime_results.csv";
    std::string error_results_file = "dot_error_results.csv";
};

ExperimentConfig load_config(const std::string& path);
std::string join_path(const std::string& dir, const std::string& filename);

#endif
