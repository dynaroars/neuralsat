#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)     \
    for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))

#define CUDA_1D_KERNEL_LOOP(i, n)     \
    CUDA_1D_KERNEL_LOOP_T(i, n, int)

inline unsigned int GET_BLOCKS(
        const unsigned int THREADS,
        const int64_t N) {
    const int64_t kMaxGridNum = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    return (unsigned int) std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

// Temporarily counter latest MSVC update that causes incompatibility with CUDA
#if (_MSC_VER >= 1928)
#define floor floorf
#define ceil ceilf
#endif
