# ThunderKittens-Tutorials

A hands-on tutorial series for writing high-performance CUDA kernels with [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

## Hardware Requirements

The first few levels can be completed on any Ampere or newer GPU architectures. Starting at level X (to be announced), however, you will need the H100s to run the tensor core matrix multiplications.

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Tutorials

0. [Harness](0-harness/) — Raw CUDA GEMM and correctness check / benchmarking harness
