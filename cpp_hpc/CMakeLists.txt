cmake_minimum_required(VERSION 3.12)

project(ParallelMixedPrecisionDot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

function(configure_common_warnings target_name)
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /O2)
    else ()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
    endif ()
endfunction()

# -----------------------------
# Dot-product kernels
# -----------------------------
add_library(dot_kernels
        src/dot.cpp
        src/config.cpp
)

target_include_directories(dot_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dot_kernels PUBLIC OpenMP::OpenMP_CXX)
configure_common_warnings(dot_kernels)

add_executable(dot_openmp
        src/main.cpp
)

target_link_libraries(dot_openmp PRIVATE dot_kernels)
configure_common_warnings(dot_openmp)

# -----------------------------
# Householder QR kernels
# -----------------------------
add_library(qr_kernels
        src/qr_serial.cpp
        src/qr_metrics.cpp
        src/matrix_utils.cpp
)

target_include_directories(qr_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

configure_common_warnings(qr_kernels)

add_executable(qr_householder
        src/main_qr.cpp
)

target_link_libraries(qr_householder PRIVATE qr_kernels)
configure_common_warnings(qr_householder)

add_executable(qr_benchmark
        src/main_qr_benchmark.cpp
)

target_link_libraries(qr_benchmark PRIVATE qr_kernels)
configure_common_warnings(qr_benchmark)
