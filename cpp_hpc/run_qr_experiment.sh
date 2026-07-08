#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_DIR="${PROJECT_DIR}/results/qr"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/qr_run_${TIMESTAMP}.log"

echo "========================================" | tee "${LOG_FILE}"
echo "QR experiment pipeline" | tee -a "${LOG_FILE}"
echo "Project dir: ${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "Build dir:   ${BUILD_DIR}" | tee -a "${LOG_FILE}"
echo "Results dir: ${RESULTS_DIR}" | tee -a "${LOG_FILE}"
echo "Log file:    ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[1/5] Configuring CMake..." | tee -a "${LOG_FILE}"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[2/5] Building QR targets..." | tee -a "${LOG_FILE}"
cmake --build "${BUILD_DIR}" -j --target qr_householder qr_benchmark 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[3/5] Running QR correctness test..." | tee -a "${LOG_FILE}"
(
    cd "${PROJECT_DIR}"
    ./build/qr_householder
) 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[4/5] Running QR benchmark..." | tee -a "${LOG_FILE}"
(
    cd "${PROJECT_DIR}"
    ./build/qr_benchmark
) 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[5/5] Generating QR plots..." | tee -a "${LOG_FILE}"
(
    cd "${PROJECT_DIR}"
    python3 scripts/plot_qr_results.py
) 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "QR experiment completed successfully." | tee -a "${LOG_FILE}"
echo "Generated files:" | tee -a "${LOG_FILE}"
echo "  ${RESULTS_DIR}/qr_serial_results.csv" | tee -a "${LOG_FILE}"
echo "  ${RESULTS_DIR}/qr_serial_summary.csv" | tee -a "${LOG_FILE}"
echo "  ${RESULTS_DIR}/qr_runtime_vs_size.png" | tee -a "${LOG_FILE}"
echo "  ${RESULTS_DIR}/qr_residual_vs_size.png" | tee -a "${LOG_FILE}"
echo "  ${RESULTS_DIR}/qr_orthogonality_vs_size.png" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
