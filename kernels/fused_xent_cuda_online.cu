// Fused softmax + cross-entropy forward, hand-CUDA "online" 1-pass variant.
//
// Implements Milakov & Gimelshein's online normalizer calculation: the row max
// (m) and row sum-of-exps (d) are merged into a single associative reduction
//
//   merge((m_a, d_a), (m_b, d_b)) = (max(m_a, m_b),
//                                    d_a * exp(m_a - m) + d_b * exp(m_b - m))
//
// so the full row is traversed exactly once. A tiny "pass 2" of a single
// scalar load gathers the target logit. This is the same algorithm used by
// the Triton implementation, written out in CUDA to isolate the language cost
// from the algorithmic cost in the benchmark.
//
// Compiled from kernels/cuda_launcher.py with:
//   -O3 -use_fast_math
//   -gencode=arch=compute_75,code=sm_75
//   -gencode=arch=compute_100,code=sm_100

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define XENT_BLOCK_SIZE 256
#define XENT_WARP_SIZE 32
#define XENT_NUM_WARPS (XENT_BLOCK_SIZE / XENT_WARP_SIZE)

namespace {

struct MD {
    float m;
    float d;
};

__device__ __forceinline__ MD md_merge(MD a, MD b) {
    MD r;
    r.m = fmaxf(a.m, b.m);
    // Identity-safe: if both operands are the (-inf, 0) identity, force d=0
    // to avoid NaN from (-inf) - (-inf). In practice the block is always
    // bigger than V for our shapes so this branch is only hit for padded lanes.
    r.d = (r.m == -INFINITY)
        ? 0.0f
        : (a.d * __expf(a.m - r.m) + b.d * __expf(b.m - r.m));
    return r;
}

__device__ __forceinline__ MD warp_reduce_md(MD val) {
    #pragma unroll
    for (int offset = XENT_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffffu, val.m, offset);
        other.d = __shfl_xor_sync(0xffffffffu, val.d, offset);
        val = md_merge(val, other);
    }
    return val;
}

__device__ __forceinline__ MD block_reduce_md(MD val, MD* smem) {
    const int tid = threadIdx.x;
    const int warp = tid / XENT_WARP_SIZE;
    const int lane = tid % XENT_WARP_SIZE;

    val = warp_reduce_md(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        MD identity = {-INFINITY, 0.0f};
        val = (tid < XENT_NUM_WARPS) ? smem[tid] : identity;
        val = warp_reduce_md(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__global__ void fused_xent_online_kernel(
    const __half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss,
    int N, int V
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const __half* row_ptr = logits + static_cast<size_t>(row) * V;
    const __half2* __restrict__ row_ptr2 =
        reinterpret_cast<const __half2*>(row_ptr);

    __shared__ MD smem[XENT_NUM_WARPS];

    // Per-thread online scan over the row, half2-vectorized.
    MD acc = {-INFINITY, 0.0f};
    const int V2 = V >> 1;
    for (int i = tid; i < V2; i += XENT_BLOCK_SIZE) {
        __half2 x2 = row_ptr2[i];
        const float a = __half2float(__low2half(x2));
        const float b = __half2float(__high2half(x2));
        MD ma = {a, 1.0f};
        MD mb = {b, 1.0f};
        acc = md_merge(acc, ma);
        acc = md_merge(acc, mb);
    }
    // Odd-V tail.
    if ((V & 1) && tid == 0) {
        const float a = __half2float(row_ptr[V - 1]);
        MD ma = {a, 1.0f};
        acc = md_merge(acc, ma);
    }

    const MD reduced = block_reduce_md(acc, smem);

    // Tiny pass 2: gather target logit (one scalar load per row).
    if (tid == 0) {
        const int64_t t = targets[row];
        const float x_t = __half2float(row_ptr[t]);
        loss[row] = logf(reduced.d) + reduced.m - x_t;
    }
}

}  // namespace


torch::Tensor fused_xent_cuda_online(torch::Tensor logits, torch::Tensor targets) {
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

    fused_xent_online_kernel<<<
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
