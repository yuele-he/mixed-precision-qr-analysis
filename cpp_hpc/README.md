# C++ OpenMP Mixed-Precision Dot Product Benchmark

This module extends the Python floating-point error analysis project with C++ OpenMP kernels for inner products.

## Goal

The experiment separates two questions:

1. **Runtime benchmark**: how do vector size, OpenMP thread count, precision model, and input type affect performance?
2. **Error analysis**: how do mixed-precision arithmetic and OpenMP reduction order affect numerical error across random
   trials?

## Arithmetic models

The code currently implements:

- `fp64`: FP64 product + FP64 accumulation.
- `mixed_fp32_fp64`: FP32 operands + FP32 product + FP64 accumulation.

The mixed model corresponds to:

```text
float xi = static_cast<float>(x[i]);
float yi = static_cast<float>(y[i]);
float prod_low = xi * yi;
sum += static_cast<double>(prod_low);
```

## Input distributions

`config.json` controls the active input distributions:

```json
"input_distributions": ["normal"],
"plot_input_distributions": ["normal"]
```

Supported input distributions are:

```text
normal              standard normal vectors
positive_uniform    positive entries in [0, 1]
alternating_sign    alternating signs to introduce cancellation
ill_scaled          entries with widely varying magnitudes
nearly_canceling    constructed products with strong cancellation
```

For a quick non-normal diagnostic run, use:

```bash
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_nonnormal_quick.json
```

For a quick run over all supported input types, use:

```bash
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_multi_input_quick.json
```

For larger final runs, edit `config.json` directly. Enabling many input distributions multiplies runtime, so keep
`sizes`, `runtime_repeats`, and `error_trials` modest while testing.

## Output files

All output paths are controlled by `config.json`. The path prefix appears only once:

```json
"results_dir": "cpp_hpc/results"
```

CSV outputs:

```text
dot_runtime_results.csv      raw runtime repeats
dot_error_results.csv        raw error trials
runtime_summary.csv          runtime quantiles and best/median values
speedup_summary.csv          speedup vs serial same precision
parallel_error_summary.csv   parallel reduction difference quantiles
mixed_error_summary.csv      mixed arithmetic error quantiles
condition_number_summary.csv dot-product condition-number quantiles
```

Figures:

```text
runtime_benchmark.png
speedup_comparison.png
parallel_error_median.png
parallel_error_distribution_fixed_n.png
mixed_error_quantile_trend.png
mixed_error_distribution_fixed_n.png   # single-panel boxplots by representative n
condition_number_quantile_trend.png
condition_number_distribution_fixed_n.png
runtime_distribution_fixed_n.png
```

When more than one input distribution is plotted, the distribution name is inserted before the `.png` suffix, for
example:

```text
parallel_error_median_normal.png
parallel_error_median_positive_uniform.png
parallel_error_median_alternating_sign.png
```

## Visualization design

The plotting script uses:

- main trend figures for median/best trends;
- distribution figures for boxplot + jittered trial/repeat points;
- FP64 in a blue sequential palette;
- mixed precision in a red sequential palette;
- darker shades for larger thread counts;
- log-scale y-axis with decade-rounded limits;
- shared y-axis limits inside comparison figures, so FP64 and mixed panels are visually comparable;
- mixed-error distribution drawn as one single panel with one boxplot per representative vector size.

The log-scale limits are controlled by:

```json
"y_axis": {
  "round_to_decades": true,
  "pad_decades": 0.25,
  "min_decades": 1.5,
  "lower_percentile": 0,
  "upper_percentile": 100
}
```

If a plot is dominated by outliers, you can set `lower_percentile` / `upper_percentile`, for example `1` and `99`, to
make the visible log range more balanced.

## Recommended interpretation

- `mixed_error_vs_fp64` measures the arithmetic error of the mixed model relative to serial FP64.
- `parallel_error_vs_serial_same_precision` measures the relative difference between OpenMP and serial results under the
  same precision model.
- `sum_abs_products = sum_i |x_i y_i|` is the absolute product scale of the input.
- `dot_condition_number_fp64 = sum_abs_products / abs(reference_fp64)` estimates the sensitivity of the dot product to
  cancellation.
- `scaled_error_vs_sum_abs_products` normalizes absolute error by `sum_abs_products`, which is more stable than relative
  error when `x^T y` is close to zero.
- Runtime repeats are used for timing stability.
- Error trials use different random vectors to estimate the error distribution.

The main trend figures show medians or best runtime trends. Distribution figures show boxplots and jittered points at
representative vector sizes.

## Dot-product condition number

The error CSV now includes a cancellation-sensitive condition estimate:

```text
kappa_dot = sum_i |x_i y_i| / |x^T y|
```

In the CSV this appears as:

```text
sum_abs_products
dot_condition_number_fp64
dot_condition_number_same_precision
```

This value is near 1 when all product terms have the same sign and becomes large when positive and negative terms nearly
cancel. It helps explain why `nearly_canceling` inputs can show relative errors of order 1 even when the scaled absolute
error is much smaller.

The script also saves:

```text
condition_number_summary.csv
condition_number_quantile_trend.png
condition_number_distribution_fixed_n.png
```

## Run

From the project root in WSL/Linux:

```bash
./cpp_hpc/run_openmp_experiment.sh
```

With a custom config:

```bash
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_nonnormal_quick.json
./cpp_hpc/run_openmp_experiment.sh cpp_hpc/config_multi_input_quick.json
```

If Python plotting dependencies are missing:

```bash
sudo apt update
sudo apt install -y python3-pandas python3-matplotlib
```
