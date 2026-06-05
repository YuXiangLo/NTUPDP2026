// seam_dp.cu
// Fused vertical-seam dynamic-programming kernel for seam carving.
//
// The whole-image DP is launched as ONE kernel per seam instead of one tiny
// kernel per row.  A single thread block walks the rows top-to-bottom; within a
// row every column is processed in parallel.  The previous/current cumulative
// cost rows live in shared memory (ping-pong buffers).  argmin of the last row
// and the backtrack walk are done by thread 0 at the end so nothing has to come
// back to Python until the final seam is ready.
//
// Target: NVIDIA V100 (sm_70).  Build with the accompanying Makefile.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

// One block, blockDim.x threads, dynamic shared mem = 2 * W floats.
__global__ void seam_dp_kernel(const float* __restrict__ energy,
                               int* __restrict__ back,
                               int* __restrict__ seam,
                               int H, int W) {
    extern __shared__ float sh[];
    float* prev = sh;          // cumulative cost of row i-1
    float* curr = sh + W;      // cumulative cost of row i

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Row 0: cumulative cost == energy.
    for (int j = tid; j < W; j += nthreads) {
        prev[j] = energy[j];
    }
    __syncthreads();

    for (int i = 1; i < H; ++i) {
        const float* erow = energy + (size_t)i * W;
        int* brow = back + (size_t)i * W;
        for (int j = tid; j < W; j += nthreads) {
            const int la = j > 0 ? j - 1 : 0;          // left  neighbour (clamped)
            const int ra = j < W - 1 ? j + 1 : W - 1;  // right neighbour (clamped)
            // Order left -> center -> right with strict '<' so ties prefer the
            // lower column index, matching the torch reference (CUDA_v0).
            float c = prev[la];
            int arg = la;
            float m = prev[j];
            if (m < c) { c = m; arg = j; }
            m = prev[ra];
            if (m < c) { c = m; arg = ra; }
            curr[j] = erow[j] + c;
            brow[j] = arg;
        }
        __syncthreads();
        // Pointer swap (per-thread, deterministic). The sync above guarantees
        // every thread finished reading `prev` before it becomes the next curr.
        float* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // `prev` holds the last row's cumulative cost. argmin + backtrack on one
    // thread: H+W serial ops, done once per seam -> negligible.
    if (tid == 0) {
        int best = 0;
        float bestv = prev[0];
        for (int j = 1; j < W; ++j) {
            if (prev[j] < bestv) { bestv = prev[j]; best = j; }
        }
        seam[H - 1] = best;
        for (int i = H - 1; i > 0; --i) {
            best = back[(size_t)i * W + best];
            seam[i - 1] = best;
        }
    }
}

// Returns the seam as an int64 [H] tensor of column indices (one per row).
torch::Tensor find_seam(torch::Tensor energy) {
    TORCH_CHECK(energy.is_cuda(), "energy must be a CUDA tensor");
    TORCH_CHECK(energy.dim() == 2, "energy must be 2D (H, W)");
    energy = energy.to(torch::kFloat32).contiguous();

    const int H = energy.size(0);
    const int W = energy.size(1);

    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(energy.device());
    auto back = torch::empty({H, W}, i32);
    auto seam = torch::empty({H}, i32);

    int threads = W < 1024 ? ((W + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;

    const size_t shmem = 2 * (size_t)W * sizeof(float);
    // V100 allows up to 96 KB shared mem per block, but only via opt-in.
    TORCH_CHECK(shmem <= 96 * 1024,
                "image width ", W, " too large for shared-memory DP kernel "
                "(needs ", shmem, " bytes > 96KB); a global-memory fallback is required");
    cudaFuncSetAttribute(seam_dp_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem);

    auto stream = at::cuda::getCurrentCUDAStream();
    seam_dp_kernel<<<1, threads, shmem, stream>>>(
        energy.data_ptr<float>(),
        back.data_ptr<int>(),
        seam.data_ptr<int>(),
        H, W);
    C10_CUDA_CHECK(cudaGetLastError());

    return seam.to(torch::kInt64);
}

PYBIND11_MODULE(seam_cuda, m) {
    m.def("find_seam", &find_seam, "Fused vertical seam DP (CUDA)");
}
