#include <benchmark/benchmark.h>
#include <blust/blust.hpp>


static void BM_TensorAdd(benchmark::State& state) {
  constexpr size_t n = 512, m = 512;
  blust::tensor t1({n, m}), t2({n, m}), r;
  blust::cpu_ops ops;

  for (auto _ : state) {
    r = ops.add(t1, t2);
  }
}

static void BM_TensorMatMul(benchmark::State& state) {
  constexpr size_t n = 1024, m = 1024, k = 1024;
  blust::tensor t1({n, m}), t2({m, k}), r;
  blust::cpu_ops ops;

  for (auto _ : state) {
    r = ops.mat_mul(t1, t2);
  }
}

// Register the function as a benchmark
// BENCHMARK(BM_TensorAdd);
BENCHMARK(BM_TensorMatMul);
BENCHMARK_MAIN();