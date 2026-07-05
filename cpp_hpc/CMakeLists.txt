cmake_minimum_required(VERSION 3.12)

project(ParallelMixedPrecisionDot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(dot_openmp
        src/main.cpp
        src/dot.cpp
        src/config.cpp
)

target_include_directories(dot_openmp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dot_openmp PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(dot_openmp PRIVATE /O2)
else ()
    target_compile_options(dot_openmp PRIVATE -O3 -march=native)
endif ()
