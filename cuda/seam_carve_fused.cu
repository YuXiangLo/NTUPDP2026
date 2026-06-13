// seam_carve_fused.cu — Fused grayscale+energy+DP kernel.
//
//   ./seam_carve_fused <input.jpg> <num_seams> [output.png]
//
// v5 runs three separate passes: grayscale_kernel → energy_kernel → dp_kernel.
// Each pass reads/writes the full image from/to global memory.  the fused kernel fuses all
// three into a single kernel that computes grayscale, energy, and the DP
// cumulative cost row-by-row using a shared-memory sliding window, never
// writing intermediate gray[] or energy[] to global memory.
//
// Shared memory layout (5 rows × W floats):
//   gray_prev[W]  — grayscale for row i-1
//   gray_curr[W]  — grayscale for row i
//   gray_next[W]  — grayscale for row i+1 (prefetched)
//   dp_prev[W]    — DP cumulative cost for row i-1
//   dp_curr[W]    — DP cumulative cost for row i (being computed)
//
// Shared memory requirement: 5 × W × 4 bytes
//   1080p (W=1920): 37.5 KB — within default 48 KB
//   4K    (W=3840): 75 KB   — needs cudaFuncSetAttribute (V100 supports 96 KB)
//   8K    (W=7680): 150 KB  — exceeds V100; use seam_carve_wide instead
//
// Single block, identical seam_dp_prefetch_kernel thread model: each thread
// owns up to 2 columns (c0 = tid, c1 = tid + blockDim.x), width ≤ 2048.
//
// Target: NVIDIA V100 (sm_70).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) { \
             fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
             exit(1); } } while(0)

// ---------------------------------------------------------------------------
// Fused kernel
// ---------------------------------------------------------------------------

// Load one row of RGB float → grayscale, writing into a shared memory slot.
__device__ __forceinline__ void load_gray_row(const float* __restrict__ img,
                                               float* __restrict__ sh_row,
                                               int row, int W) {
    // Each thread handles its 2 columns.
    // NOTE: threadIdx.x and threadIdx.x + blockDim.x are the column indices.
    const int c0 = threadIdx.x;
    const int c1 = threadIdx.x + blockDim.x;
    if (c0 < W) {
        const float* p = img + ((size_t)row * W + c0) * 3;
        sh_row[c0] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
    if (c1 < W) {
        const float* p = img + ((size_t)row * W + c1) * 3;
        sh_row[c1] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

// Compute energy for column j given gray_prev (row i-1), gray_curr (row i),
// gray_next (row i+1) — all clamped at boundaries.
__device__ __forceinline__ float compute_energy(const float* __restrict__ gray_prev,
                                                 const float* __restrict__ gray_curr,
                                                 const float* __restrict__ gray_next,
                                                 int j, int W,
                                                 bool has_prev, bool has_next) {
    int l = j > 0   ? j - 1 : 0;
    int r = j < W-1 ? j + 1 : W - 1;
    float dx = fabsf(gray_curr[r] - gray_curr[l]);
    float dy = fabsf((has_next ? gray_next[j] : gray_curr[j]) -
                     (has_prev ? gray_prev[j] : gray_curr[j]));
    return dx + dy;
}

// DP step: pick minimum predecessor, write dp_curr[j] = energy + min(prev neighbors).
__device__ __forceinline__ void dp_step(const float* __restrict__ dp_prev,
                                         float* __restrict__ dp_curr,
                                         signed char* __restrict__ brow,
                                         int j, int W, float energy) {
    int la = j > 0   ? j - 1 : 0;
    int ra = j < W-1 ? j + 1 : W - 1;
    float c = dp_prev[la]; int arg = la;
    float m = dp_prev[j];  if (m < c) { c = m; arg = j;  }
    m = dp_prev[ra];       if (m < c) { c = m; arg = ra; }
    dp_curr[j] = energy + c;
    brow[j] = (signed char)(arg - j);
}

// Main fused kernel.
// Shared memory layout:  [gray_prev | gray_curr | gray_next | dp_prev | dp_curr]
// Each slot is W floats.  Total: 5 * W * sizeof(float).
__global__ void fused_gray_energy_dp_kernel(
        const float* __restrict__ img,
        signed char* __restrict__ back,
        int* __restrict__ seam,
        int H, int W)
{
    extern __shared__ float sh[];
    float* gray_prev = sh;
    float* gray_curr = sh +     W;
    float* gray_next = sh + 2 * W;
    float* dp_prev   = sh + 3 * W;
    float* dp_curr   = sh + 4 * W;

    const int c0 = threadIdx.x;
    const int c1 = threadIdx.x + blockDim.x;

    // --- Load grayscale for row 0 into gray_curr ---
    load_gray_row(img, gray_curr, 0, W);
    // --- Prefetch grayscale for row 1 into gray_next (if exists) ---
    if (H > 1) load_gray_row(img, gray_next, 1, W);
    __syncthreads();

    // Row 0: energy with no prev row (clamp dy to 0 by using itself as prev)
    // dp_prev[j] = energy[0][j]
    if (c0 < W) {
        float e = compute_energy(gray_curr, gray_curr, gray_next,
                                 c0, W, /*has_prev=*/false, /*has_next=*/(H>1));
        dp_prev[c0] = e;
    }
    if (c1 < W) {
        float e = compute_energy(gray_curr, gray_curr, gray_next,
                                 c1, W, /*has_prev=*/false, /*has_next=*/(H>1));
        dp_prev[c1] = e;
    }
    __syncthreads();

    // --- Row-by-row DP ---
    for (int i = 1; i < H; ++i) {
        // Rotate: gray_prev ← gray_curr, gray_curr ← gray_next.
        // We do this by swapping pointers (can't do pointer-swap in shared mem
        // directly, but we can index with a modular counter).
        // Actually: copy gray_curr → gray_prev (in shared), then load i+1 → gray_curr.
        // But copying W floats per step wastes bandwidth. Instead:
        // Keep 3 circular slots and rotate index.  We'll do an explicit copy.
        // (For the paper, bandwidth savings come from not writing gray/energy to HBM;
        //  intra-SM copies are free compared to HBM roundtrips.)

        // gray_prev ← gray_curr (copy in shared mem — fast L1 bandwidth)
        if (c0 < W) gray_prev[c0] = gray_curr[c0];
        if (c1 < W) gray_prev[c1] = gray_curr[c1];

        // gray_curr ← gray_next (already loaded)
        if (c0 < W) gray_curr[c0] = gray_next[c0];
        if (c1 < W) gray_curr[c1] = gray_next[c1];

        // Prefetch gray for row i+1 into gray_next
        bool has_next = (i + 1 < H);
        if (has_next) {
            load_gray_row(img, gray_next, i + 1, W);
        }
        __syncthreads();

        // Compute energy for row i, then DP step
        signed char* brow = back + (size_t)i * W;
        if (c0 < W) {
            float e = compute_energy(gray_prev, gray_curr, gray_next,
                                     c0, W, /*has_prev=*/true, /*has_next=*/has_next);
            dp_step(dp_prev, dp_curr, brow, c0, W, e);
        }
        if (c1 < W) {
            float e = compute_energy(gray_prev, gray_curr, gray_next,
                                     c1, W, /*has_prev=*/true, /*has_next=*/has_next);
            dp_step(dp_prev, dp_curr, brow, c1, W, e);
        }
        __syncthreads();

        // Rotate DP: dp_prev ← dp_curr
        float* tmp = dp_prev; dp_prev = dp_curr; dp_curr = tmp;
        // (pointer swap is fine since we just swap which slot we write next)
    }

    // Backtrack (thread 0 only)
    if (threadIdx.x == 0) {
        int best = 0; float bestv = dp_prev[0];
        for (int j = 1; j < W; ++j)
            if (dp_prev[j] < bestv) { bestv = dp_prev[j]; best = j; }
        seam[H - 1] = best;
        for (int i = H - 1; i > 0; --i) {
            best += back[(size_t)i * W + best];
            seam[i - 1] = best;
        }
    }
}

// remove_seam_kernel (identical to v5)
__global__ void remove_seam_kernel(const float* __restrict__ img,
                                    float* __restrict__ out,
                                    const int* __restrict__ seam, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < H && x < W - 1) {
        int s = seam[y], src = x < s ? x : x + 1;
        const float* sp = img + ((size_t)y * W + src) * 3;
        float* op = out + ((size_t)y * (W - 1) + x) * 3;
        op[0] = sp[0]; op[1] = sp[1]; op[2] = sp[2];
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.jpg> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    int num_seams = atoi(argv[2]);
    const char* out_path = argc >= 4 ? argv[3] : "carved.png";

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    if (W0 > 2048) {
        fprintf(stderr, "seam_carve_fused supports width up to 2048 (got %d); use seam_carve_wide\n", W0);
        return 1;
    }

    const size_t shmem_needed = 5 * (size_t)W0 * sizeof(float);
    if (shmem_needed > 96 * 1024) {
        fprintf(stderr,
            "seam_carve_fused needs %zu bytes shared mem for W=%d (>96KB); use seam_carve_wide\n",
            shmem_needed, W0);
        return 1;
    }
    printf("loaded %s : %dx%d, removing %d seams\n", in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2;
    signed char* d_back;
    int* d_seam;
    CUDA_CHECK(cudaMalloc(&d_img,  npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back, npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam, (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Allow up to 96 KB dynamic shared memory
    CUDA_CHECK(cudaFuncSetAttribute(fused_gray_energy_dp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        int threads = w <= 1024 ? ((w + 31) / 32) * 32 : 1024;
        if (threads < 32) threads = 32;
        size_t shmem = 5 * (size_t)w * sizeof(float);

        fused_gray_energy_dp_kernel<<<1, threads, shmem>>>(d_img, d_back, d_seam, H, w);
        remove_seam_kernel<<<grid2d(w-1), block2d>>>(d_img, d_img2, d_seam, H, w);

        float* tmp = d_img; d_img = d_img2; d_img2 = tmp;
        --w;
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaGetLastError());

    printf("GPU carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
           ms, ms / num_seams, w, H);

    const size_t out_pix = (size_t)w * H;
    float* h_out_f = (float*)malloc(out_pix * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out_f, d_img, out_pix * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned char* h_out = (unsigned char*)malloc(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = h_out_f[i] * 255.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        h_out[i] = (unsigned char)(v + 0.5f);
    }
    if (!stbi_write_png(out_path, w, H, 3, h_out, w * 3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    free(h_img); free(h_out_f); free(h_out);
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_back); cudaFree(d_seam);
    return 0;
}
