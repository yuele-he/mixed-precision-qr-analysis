cmake_minimum_required(VERSION 3.12)

project(ParallelMixedPrecisionDot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# -----------------------------
# Dot-product benchmark
# -----------------------------
add_executable(dot_openmp
        src/main.cpp
        src/dot.cpp
        src/config.cpp
)

target_include_directories(dot_openmp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dot_openmp PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------
# Serial Householder QR correctness test
# -----------------------------
add_executable(qr_householder
        src/main_qr.cpp
        src/qr_serial.cpp
        src/qr_metrics.cpp
)

target_include_directories(qr_householder PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(qr_householder PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------
# Serial Householder QR benchmark
# -----------------------------
add_executable(qr_benchmark
        src/main_qr_benchmark.cpp
        src/qr_serial.cpp
        src/qr_metrics.cpp
)

target_include_directories(qr_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(qr_benchmark PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(dot_openmp PRIVATE /O2)
    target_compile_options(qr_householder PRIVATE /O2)
    target_compile_options(qr_benchmark PRIVATE /O2)
else ()
    target_compile_options(dot_openmp PRIVATE -O3 -march=native)
    target_compile_options(qr_householder PRIVATE -O3 -march=native)
    target_compile_options(qr_benchmark PRIVATE -O3 -march=native)
endif ()
