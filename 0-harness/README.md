# Tutorial 0: Raw CUDA GEMM and Everything Around It

**TL;DR: Read the [Readings](#readings), and implement the [TODOs](#todos).**

## Goal

By the end of this tutorial, you will have:

1. A working naive CUDA matrix multiplication (GEMM) kernel in CUDA C++.
2. A correctness check / benchmarking harness around that kernel that follows standard industry conventions.
3. A clear way to run kernels and report performance (time, TFLOPs).

## Key Ideas

- Instead of learning about kernels first, we start by understanding what happens around a CUDA kernel: compilation, launches, synchronization, correctness checks, and measurement.
- We will implement a very simple GEMM kernel, but everything around it should be production-ready (i.e., ready to reuse in research or industry).

## Readings

- Intro to CUDA C++
  - [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
  - Ignore "2.1.3.1. Unified Memory" section and everything related to it. We'll stick to explicit memory management only.
- CUDA C++ Best Practices Guide - Timing section:
  - Read "9.1 Timing" (CPU timers vs CUDA events, plus the basic event timing example).
  - [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing)

## TODOs

You will follow these steps, and implement everything in a single file: `[kernel.cu](kernel.cu)`.

### Step 1. CPU GEMM

**Goal:** Get a correct CPU implementation you trust; performance doesn't matter yet.

**Background**

All matrices use row-major indexing. Think of it as `row * stride + col` where each row is stored as one contiguous block:

- `A` is **M×K** and `A(i,k)` lives at `A[i*K + k]`
- `B` is **K×N** and `B(k,j)` lives at `B[k*N + j]`
- `C` is **M×N** and `C(i,j)` lives at `C[i*N + j]`

To verify indexing is correct, start from a tiny example (2×2 matrices) and manually work out one or two expected outputs before trusting the random test.

**Tasks**

1. Implement `fill_random(x, seed)`
  - Fill every element once with uniform random values; you can simply use `dist(rng)`.
2. Implement `gemm_cpu_naive(A, B, C, M, N, K)`
  - loops: `i` in `[0,M)`, `j` in `[0,N)`, `k` in `[0,K)`
3. Implement `print_stats(name, x)`
4. In `main`, allocate and initialize memory.
  - Allocate `h_A` of size `M*K`, `h_B` size `K*N`, `h_C` size `M*N`, `h_C_ref` size `M*N`
    - `h_C` is later used to hold the results of GPU GEMM. Just allocate and ignore for now.
  - Fill `h_A`, `h_B` with random values using `fill_random`
  - Run CPU GEMM into `h_C_ref` and print its stats.
5. Run `make run`. Ensure that you get correct results!

Keep your first test small (e.g. `M=N=K=256`) so CPU reference runs quickly.

### Step 2. Naive GPU GEMM

**Goal:** Write the simplest possible CUDA GEMM kernel.

**Background**

The kernel code you should be filling in is:

```cpp
__global__ void gemm_gpu_naive(const float* A, const float* B, float* C,
                               int M, int N, int K);
```

The operands are in same row-major layout as the CPU:

- `A` points to `M*K` floats, `B` points to `K*N` floats, `C` points to `M*N` floats.

Inside the kernel, you can check the thread ID and the block ID to check if you are within bounds:

```cpp
int j = blockIdx.x * blockDim.x + threadIdx.x; // column in [0, N)
int i = blockIdx.y * blockDim.y + threadIdx.y; // row    in [0, M)
if (i >= M || j >= N) return;
```

Where:

- `blockIdx` chooses the block in the 2D grid
- `threadIdx` chooses your thread inside the block
- Every thread maps to one output element `C(i,j)`
- With a 16x16 block, for example, each block can compute up to 256 outputs.

The kernel perform a naive GEMM with a single for-loop.

**Tasks**

1. Implement the `gemm_gpu_naive` kernel body using the thread mapping, bounds check, and naive dot product. Assume that the kernel is given sufficient grid/block size.
2. We cannot run this yet! That will be done in step 3.

### Step 3. Kernel harness + correctness check against the CPU

**Goal:** Allocate GPU memory, run the kernel, copy results back, and compare to `gemm_cpu_naive`.

**Background**

The CPU (host) and GPU (device) have separate memory spaces. Data must be explicitly copied between them using `cudaMemcpy`. Also, every host-side CUDA API call can fail, so we must wrap them in an error-checking macro.

The usual pattern is: allocate device buffers, copy inputs, launch kernel, synchronize, copy output, validate, then free.

On the host side, you can launch the kernel with:

```cpp
dim3 block(16, 16);
dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
gemm_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

**Tasks**

1. Implement `CUDA_CHECK` so every CUDA call is checked. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda).
2. Allocate device buffers with cudaMalloc. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356).
3. Copy inputs (A and B) to device with `CUDA_CHECK(cudaMemcpy(..., cudaMemcpyHostToDevice))`. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8).
4. Set C to 0 with `CUDA_CHECK(cudaMemset(...))`. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a).
5. Launch the kernel and catch errors with `CUDA_CHECK(cudaDeviceSynchronize())`.
6. Copy C back to host with `CUDA_CHECK(cudaMemcpy(..., cudaMemcpyDeviceToHost))`.
7. Implement `max_abs_error(...)` and print it.
8. Clean up memory with `CUDA_CHECK(cudaFree(p))`. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094).
9. Run with `make run`! Do you see the correct results?

### Step 4. Benchmark harness + TFLOPs calculation

**Goal:** Time the kernel correctly with CUDA events and report throughput in TFLOPs.

**Background**

Kernel launches are asynchronous; the CPU returns immediately after submitting work to the GPU. CPU timers around a launch therefore don't measure kernel runtime. We must use CUDA events, which are recorded on the GPU timeline.

Minimal event-timing loop usually looks like the following:

1. Create two events: `cudaEventCreate(&start)`, `cudaEventCreate(&stop)`.
2. Warm up by launching the kernel `warmup_iters` times.
3. Record `start`, launch the kernel `profile_iters` times back-to-back, record `stop`.
4. `cudaEventSynchronize(stop)` then `cudaEventElapsedTime(&ms, start, stop)` to obtain time.
5. Calculate average ms per launch: `ms / profile_iters`.
6. For GEMM, each output element does `K` multiply-adds, so total FLOPs ≈ `2.0 * M * N * K`. Convert to seconds with `seconds = ms * 1e-3`, then compute TFLOPs: `tflops = (2.0 * M * N * K) / seconds / 1e12`.
7. Destroy CUDA events (`cudaEventDestroy`).

**Tasks**

1. Implement the event-timing loop directly in `main()` using the pattern above. Do 500 warmup iterations and 100 profiling iterations. See [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) for CUDA events.
2. Print out the TFLOPs. What numbers do you see? For your information, H100 GPUs have theoretical performance of 494.5 TFLOP/s for FP32 matrix multiplication. What is your kernel's utilization %?

## Conclusion

If you implemented everything correctly, you should see really low TFLOP/s, likely less than 1% of theoretical maximum. In the future tutorials, we will optimize our GEMM kernel step-by-step until we reach state-of-the-art.