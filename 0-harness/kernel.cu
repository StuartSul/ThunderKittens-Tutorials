#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>


// -----------------------
// Utilities
// -----------------------
#define CUDA_CHECK(call) do { \
    /* TODO (Section 3): implement error checking */  \
    (void)(call); \
} while(0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }


// -----------------------
// Section 1: CPU GEMM + stats
// -----------------------
static void fill_random(std::vector<float>& x, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    // TODO (Section 1): generate random floats and fill in `x` with dist(rng)
}

static void gemm_cpu_naive(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int M, int N, int K) {
    // TODO (Section 1): naive gemm with triple loop i, j, k (row-major)
}

static void print_stats(const char* name, const std::vector<float>& x) {
    double sum = 0.0;
    float max_abs = 0.0f;

    // TODO (Section 1): calculate the sum

    double avg = x.empty() ? 0.0 : sum / x.size();
    std::printf("%s: avg=%f, max_abs=%f\n", name, avg, max_abs);
}


// -----------------------
// Section 2: naive GPU GEMM kernel
// -----------------------
// One thread computes one output element C(i,j).
__global__ void gemm_gpu_naive(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    // TODO (Section 2): implement naive GEMM kernel
}


// -----------------------
// Section 3: harness + correctness helpers
// -----------------------
static float max_abs_error(const std::vector<float>& ref,
                           const std::vector<float>& got) {
    // TODO (Section 3): compute max |ref - got|
    return 0.0f;
}


int main() {
    int M = 256, N = 256, K = 256;

    // TODO (Section 1): allocate host A, B, C, C_ref (row-major)
    // std::vector<float> ...;
    // std::vector<float> ...;
    // std::vector<float> ...;
    // std::vector<float> ...;

    // TODO (Section 1): fill A,B with deterministic random
    // fill_random(...);
    // fill_random(...);

    // TODO (Section 1): run CPU GEMM into Cref, print stats
    // gemm_cpu_naive(...);
    // print_stats(...);

    // TODO (Section 3): allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    // CUDA_CHECK(cudaMalloc(...);
    // CUDA_CHECK(cudaMalloc(...);
    // CUDA_CHECK(cudaMalloc(...);

    // TODO (Section 3): initialize device memory
    // CUDA_CHECK(cudaMemcpy(d_A, ..., cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_B, ..., cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemset(d_C, ...));

    // TODO (Section 3): launch kernel
    // dim3 block(16, 16);
    // dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    // launch gemm_gpu_naive
    // CUDA_CHECK(cudaDeviceSynchronize());

    // TODO (Section 5): benchmark and report ms + TFLOPs
    // int warmup_iters  = 500;
    // int profile_iters = 100;
    // cudaEvent_t start, stop;
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));
    // ...
    // CUDA_CHECK(cudaEventDestroy(start));
    // CUDA_CHECK(cudaEventDestroy(stop));

    // TODO (Section 3): copy C back to host, print out max absolute error
    // CUDA_CHECK(cudaMemcpy(..., cudaMemcpyDeviceToHost));
    // std::printf("max_abs_error: %f\n", max_abs_error(h_C_ref, h_C));

    // TODO (Section 3): free device memory
    // CUDA_CHECK(cudaFree(...));
    // CUDA_CHECK(cudaFree(...));
    // CUDA_CHECK(cudaFree(...));

    return 0;
}
