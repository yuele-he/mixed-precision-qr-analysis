#include "config.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

static std::string read_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static std::vector<std::size_t> parse_size_t_array(const std::string& text, const std::string& key)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::runtime_error("Missing array key in config: " + key);
    }
    std::vector<std::size_t> values;
    std::string body = match[1].str();
    std::regex number_pattern("(\\d+)");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), number_pattern); it != std::sregex_iterator(); ++it)
    {
        values.push_back(static_cast<std::size_t>(std::stoull((*it)[1].str())));
    }
    return values;
}

static std::vector<int> parse_int_array(const std::string& text, const std::string& key)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::runtime_error("Missing array key in config: " + key);
    }
    std::vector<int> values;
    std::string body = match[1].str();
    std::regex number_pattern("(\\d+)");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), number_pattern); it != std::sregex_iterator(); ++it)
    {
        values.push_back(std::stoi((*it)[1].str()));
    }
    return values;
}

static std::vector<std::string> parse_string_array_optional(
    const std::string& text,
    const std::string& key,
    const std::vector<std::string>& default_value
)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return default_value;
    }
    std::vector<std::string> values;
    std::string body = match[1].str();
    std::regex string_pattern("\\\"([^\\\"]*)\\\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), string_pattern); it != std::sregex_iterator(); ++it)
    {
        values.push_back((*it)[1].str());
    }
    if (values.empty())
    {
        return default_value;
    }
    return values;
}

static int parse_int_value_optional(const std::string& text, const std::string& key, int default_value)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return default_value;
    }
    return std::stoi(match[1].str());
}

static unsigned parse_unsigned_value_optional(const std::string& text, const std::string& key, unsigned default_value)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return default_value;
    }
    return static_cast<unsigned>(std::stoul(match[1].str()));
}

static std::string parse_string_value_optional(const std::string& text, const std::string& key,
                                               const std::string& default_value)
{
    std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return default_value;
    }
    return match[1].str();
}

std::string join_path(const std::string& dir, const std::string& filename)
{
    if (dir.empty())
    {
        return filename;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\')
    {
        return dir + filename;
    }
    return dir + "/" + filename;
}

ExperimentConfig load_config(const std::string& path)
{
    const std::string text = read_file(path);

    ExperimentConfig config;
    config.sizes = parse_size_t_array(text, "sizes");
    config.thread_counts = parse_int_array(text, "thread_counts");
    config.runtime_trials = parse_int_value_optional(text, "runtime_trials", config.runtime_trials);
    config.runtime_repeats = parse_int_value_optional(text, "runtime_repeats", config.runtime_repeats);
    config.error_trials = parse_int_value_optional(text, "error_trials", config.error_trials);
    config.base_seed_x = parse_unsigned_value_optional(text, "base_seed_x", config.base_seed_x);
    config.base_seed_y = parse_unsigned_value_optional(text, "base_seed_y", config.base_seed_y);
    config.input_distributions = parse_string_array_optional(text, "input_distributions", {"normal"});
    config.plot_input_distribution = parse_string_value_optional(text, "plot_input_distribution",
                                                                 config.input_distributions.front());
    config.results_dir = parse_string_value_optional(text, "results_dir", config.results_dir);
    config.runtime_results_file = parse_string_value_optional(text, "runtime_results", config.runtime_results_file);
    config.error_results_file = parse_string_value_optional(text, "error_results", config.error_results_file);

    return config;
}
