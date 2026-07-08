# C++ HPC Numerical Linear Algebra Experiments

This module contains two connected C++ experiments for numerical linear algebra and high-performance computing:

1. **Mixed-precision dot-product benchmark**: studies FP64 vs FP32-product/FP64-accumulation dot products under serial
   and OpenMP execution.
2. **Householder QR benchmark**: implements a serial Householder QR factorization and validates it using reconstruction
   residual and orthogonality error.

The project is intentionally correctness-first: the QR implementation is currently an explicit, educational baseline
before OpenMP, blocked updates, BLAS-backed kernels, or mixed precision are introduced.

## Build

From the repository root:

```bash
cmake -S cpp_hpc -B cpp_hpc/build
cmake --build cpp_hpc/build -j
```

Main executables:

```text
cpp_hpc/build/dot_openmp       dot-product benchmark
cpp_hpc/build/qr_householder   QR correctness test
cpp_hpc/build/qr_benchmark     QR serial benchmark
```

## Householder QR experiment

Run the correctness test:

```bash
cd cpp_hpc
./build/qr_householder
```

Run the serial benchmark:

```bash
cd cpp_hpc
./build/qr_benchmark
python3 scripts/plot_qr_results.py
```

Generated QR outputs:

```text
results/qr/qr_serial_results.csv
results/qr/qr_serial_summary.csv
results/qr/qr_runtime_vs_size.png
results/qr/qr_residual_vs_size.png
results/qr/qr_orthogonality_vs_size.png
```

QR metrics:

```text
relative residual      ||A - QR||_F / ||A||_F
orthogonality error    ||I - Q^T Q||_F
```

Current QR benchmark cases:

```text
square              m = n
Tall-skinny         m = 2n
nearly_dependent    columns are nearly linearly dependent
```

The plots show individual runs and the median over trials. For numerical-error plots, the shaded region shows the
observed min-max range across trials.

## Dot-product experiment

The dot-product benchmark separates two questions:

1. **Runtime benchmark**: how do vector size, OpenMP thread count, precision model, and input type affect performance?
2. **Error analysis**: how do mixed-precision arithmetic and OpenMP reduction order affect numerical error across random
   trials?

Run with the default configuration:

```bash
./cpp_hpc/run_openmp_experiment.sh
```

Run with a custom configuration:

```bash
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_nonnormal_quick.json
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_multi_input_quick.json
```

### Arithmetic models

```text
fp64                 FP64 product + FP64 accumulation
mixed_fp32_fp64      FP32 operands + FP32 product + FP64 accumulation
```

The mixed model corresponds to:

```cpp
float xi = static_cast<float>(x[i]);
float yi = static_cast<float>(y[i]);
float prod_low = xi * yi;
sum += static_cast<double>(prod_low);
```

### Input distributions

```text
normal              standard normal vectors
positive_uniform    positive entries in [0, 1]
alternating_sign    alternating signs to introduce cancellation
ill_scaled          entries with widely varying magnitudes
nearly_canceling    constructed products with strong cancellation
```

### Dot-product metrics

```text
mixed_error_vs_fp64
parallel_error_vs_serial_same_precision
sum_abs_products = sum_i |x_i y_i|
dot_condition_number_fp64 = sum_abs_products / abs(reference_fp64)
scaled_error_vs_sum_abs_products
```

`dot_condition_number_fp64` helps explain cancellation-sensitive cases. It is near 1 when products have the same sign
and becomes large when positive and negative products nearly cancel.

## Project structure

```text
include/
  config.hpp          dot-product configuration parser
  dot.hpp             dot-product kernels
  qr.hpp              QR data structure, indexing, metrics API
  matrix_utils.hpp    QR matrix generators

src/
  main.cpp                dot-product experiment driver
  dot.cpp                 serial/OpenMP dot kernels
  config.cpp              lightweight config parser
  main_qr.cpp             QR correctness test
  main_qr_benchmark.cpp   QR benchmark driver
  matrix_utils.cpp        matrix generation utilities
  qr_serial.cpp           serial Householder QR
  qr_metrics.cpp          QR residual and orthogonality metrics

scripts/
  plot_dot_results.py
  plot_qr_results.py
```

## Development notes

Current QR limitations are intentional:

```text
explicit Q_full construction
no compact Householder storage yet
no blocked update yet
no BLAS GEMM yet
no OpenMP QR update yet
no pivoting
only double precision
```

These limitations make the serial QR implementation easy to verify. The next natural steps are OpenMP parallelization of
the trailing matrix update, then blocked/BLAS-backed QR, then mixed-precision variants.
