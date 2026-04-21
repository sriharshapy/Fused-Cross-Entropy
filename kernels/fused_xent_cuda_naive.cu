// Fused softmax + cross-entropy forward, hand-CUDA "naive" 3-pass variant.
//
// One block per row. Block size = 256 threads = 8 warps.
// Pass 1: row max via warp-shuffle + shared-mem block reduce.
// Pass 2: sum of exp(x - max), same reduction pattern.
// Pass 3: thread 0 gathers the target logit and writes loss.
//
// Loads are vectorized as half2 (4B per transaction per thread) for peak
// coalesced bandwidth; the odd V tail falls back to scalar half loads.
//
// Compiled from kernels/cuda_launcher.py with:
//   -O3 -use_fast_math
//   -gencode=arch=compute_75,code=sm_75   (Turing, T4)
//   -gencode=arch=compute_100,code=sm_100 (Blackwell, B200)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define XENT_BLOCK_SIZE 256
#define XENT_WARP_SIZE 32
#define XENT_NUM_WARPS (XENT_BLOCK_SIZE / XENT_WARP_SIZE)

namespace {

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = XENT_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = XENT_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    const int tid = threadIdx.x;
    const int warp = tid / XENT_WARP_SIZE;
    const int lane = tid % XENT_WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (tid < XENT_NUM_WARPS) ? smem[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    const int warp = tid / XENT_WARP_SIZE;
    const int lane = tid % XENT_WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (tid < XENT_NUM_WARPS) ? smem[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__global__ void fused_xent_naive_kernel(
    const __half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss,
    int N, int V
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const __half* row_ptr = logits + static_cast<size_t>(row) * V;

    __shared__ float smem_max[XENT_NUM_WARPS];
    __shared__ float smem_sum[XENT_NUM_WARPS];

    const int V2 = V >> 1;
    const __half2* __restrict__ row_ptr2 =
        reinterpret_cast<const __half2*>(row_ptr);

    // -----------------------------------------------------------------
    // Pass 1: row max, using half2 vectorized loads.
    // -----------------------------------------------------------------
    float local_max = -INFINITY;
    for (int i = tid; i < V2; i += XENT_BLOCK_SIZE) {
        __half2 x2 = row_ptr2[i];
        const float a = __half2float(__low2half(x2));
        const float b = __half2float(__high2half(x2));
        local_max = fmaxf(local_max, fmaxf(a, b));
    }
    // Odd-V tail: thread 0 handles the dangling scalar.
    if ((V & 1) && tid == 0) {
        const float a = __half2float(row_ptr[V - 1]);
        local_max = fmaxf(local_max, a);
    }
    const float row_max = block_reduce_max(local_max, smem_max);

    // -----------------------------------------------------------------
    // Pass 2: sum of exp(x - row_max).
    // -----------------------------------------------------------------
    float local_sum = 0.0f;
    for (int i = tid; i < V2; i += XENT_BLOCK_SIZE) {
        __half2 x2 = row_ptr2[i];
        const float a = __half2float(__low2half(x2));
        const float b = __half2float(__high2half(x2));
        local_sum += __expf(a - row_max) + __expf(b - row_max);
    }
    if ((V & 1) && tid == 0) {
        const float a = __half2float(row_ptr[V - 1]);
        local_sum += __expf(a - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, smem_sum);

    // -----------------------------------------------------------------
    // Pass 3: gather target logit and write per-row loss.
    // -----------------------------------------------------------------
    if (tid == 0) {
        const int64_t t = targets[row];
        const float x_t = __half2float(row_ptr[t]);
        loss[row] = logf(row_sum) + row_max - x_t;
    }
}

}  // namespace


torch::Tensor fused_xent_cuda_naive(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat16, "logits must be fp16");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be [N, V]");
    TORCH_CHECK(
        targets.dim() == 1 && targets.size(0) == logits.size(0),
        "targets must be [N] matching logits.size(0)"
    );
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    const int64_t N = logits.size(0);
    const int64_t V = logits.size(1);
    TORCH_CHECK(V > 0 && V <= INT_MAX, "V out of range");
    TORCH_CHECK(N > 0 && N <= INT_MAX, "N out of range");

    auto loss = torch::empty(
        {N},
        torch::dtype(torch::kFloat32).device(logits.device())
    );

    const dim3 grid(static_cast<unsigned int>(N));
    const dim3 block(XENT_BLOCK_SIZE);

    fused_xent_naive_kernel<<<
        grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(V)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return loss;
}
