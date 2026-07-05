#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-cpp_hpc/config.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"

echo "Compiling C++ OpenMP benchmark..."
g++ -O3 -std=c++17 -fopenmp \
  cpp_hpc/src/main.cpp cpp_hpc/src/dot.cpp cpp_hpc/src/config.cpp \
  -Icpp_hpc/include \
  -o cpp_hpc/dot_openmp

echo "Setting OpenMP environment..."
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "Cleaning old generated figures and summary CSVs..."
rm -f cpp_hpc/results/*.png
rm -f cpp_hpc/results/runtime_summary.csv \
      cpp_hpc/results/speedup_summary.csv \
      cpp_hpc/results/parallel_error_summary.csv \
      cpp_hpc/results/mixed_error_summary.csv \
      cpp_hpc/results/condition_number_summary.csv

echo "Running runtime benchmark and error analysis..."
./cpp_hpc/dot_openmp "$CONFIG_PATH"

echo "Plotting and generating summary CSVs..."
CPP_HPC_CONFIG="$CONFIG_PATH" python3 cpp_hpc/scripts/plot_dot_results.py

echo "Done."
echo "Results: cpp_hpc/results/"
