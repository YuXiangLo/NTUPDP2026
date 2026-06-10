// seam_carve_v6.cu - Standalone CUDA vertical seam carving, v6.
//
//   ./seam_carve_v6 <input.jpg> <num_seams> [output.png]
//
// v5 (seam_carve_v5.cu) generalized to arbitrary width. v4/v5 hard-coded "2
// columns per thread" (c0 = tid, c1 = tid + blockDim.x), which capped width at
// 2*1024 = 2048. v6 lifts that to ceil(W / blockDim.x) columns per thread via a
// grid-stride over columns, so the only remaining limit is shared memory: the
// DP keeps two cost rows in shared mem (2*W*4 bytes <= 96KB on V100 -> W up to
// ~12288).
//
// To keep v5's register-resident energy prefetch (no local-memory spill), the
// per-thread column count CPT is a COMPILE-TIME template parameter: the host
// computes cpt = ceil(W / nthreads) and dispatches to the matching kernel
// instantiation. For W <= 2048 this is CPT=2, i.e. byte-for-byte the same work
// as v5 (identical performance); a 3840-wide image runs CPT=4; etc. The prefetch
// arrays epf[CPT]/ecur[CPT] are fixed-size and fully unrolled, so they stay in
// registers.
//
// int8 relative back-pointers (-1/0/+1) are inherited from v5.
//
// Target: NVIDIA V100 (sm_70).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Largest per-thread column count we instantiate. CPT=8 with blockDim.x=1024
// covers widths up to 8192 while keeping register pressure modest.
#define MAX_CPT 8

// ---------------------------------------------------------------------------
// Kernels  (grayscale / energy / remove_seam are identical to v2/v4/v5)
// ---------------------------------------------------------------------------

__global__ void grayscale_kernel(const float* __restrict__ img,
                                 float* __restrict__ gray, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        const float* p = img + ((size_t)y * W + x) * 3;
        gray[(size_t)y * W + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

__global__ void energy_kernel(const float* __restrict__ gray,
                              float* __restrict__ energy, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        int l = x > 0 ? x - 1 : 0;
        int r = x < W - 1 ? x + 1 : W - 1;
        int u = y > 0 ? y - 1 : 0;
        int d = y < H - 1 ? y + 1 : H - 1;
        float dx = fabsf(gray[(size_t)y * W + r] - gray[(size_t)y * W + l]);
        float dy = fabsf(gray[(size_t)d * W + x] - gray[(size_t)u * W + x]);
        energy[(size_t)y * W + x] = dx + dy;
    }
}

// One DP cell. back stores the chosen previous column as a RELATIVE offset in
// {-1, 0, +1} (1 byte) rather than the absolute column index.
__device__ __forceinline__ void dp_cell(const float* __restrict__ prev,
                                         float* __restrict__ curr,
                                         signed char* __restrict__ brow,
                                         int j, int W, float e) {
    const int la = j > 0 ? j - 1 : 0;
    const int ra = j < W - 1 ? j + 1 : W - 1;
    float c = prev[la];
    int arg = la;
    float m = prev[j];
    if (m < c) { c = m; arg = j; }
    m = prev[ra];
    if (m < c) { c = m; arg = ra; }
    curr[j] = e + c;
    brow[j] = (signed char)(arg - j);   // -1 / 0 / +1
}

// v6 fused DP: single block, energy prefetched one row ahead, int8 back-pointers,
// CPT columns per thread (compile-time) so any width up to the shared-mem limit
// works while the prefetch arrays stay in registers.
template <int CPT>
__global__ void seam_dp_kernel(const float* __restrict__ energy,
                               signed char* __restrict__ back,
                               int* __restrict__ seam, int H, int W) {
    extern __shared__ float sh[];
    float* prev = sh;
    float* curr = sh + W;
    const int nt = blockDim.x;
    const int tid = threadIdx.x;

    // Row 0 init.
    #pragma unroll
    for (int k = 0; k < CPT; ++k) {
        int j = tid + k * nt;
        if (j < W) prev[j] = energy[j];
    }
    __syncthreads();

    // Prefetch row 1 energy into registers.
    float epf[CPT];
    #pragma unroll
    for (int k = 0; k < CPT; ++k) {
        int j = tid + k * nt;
        epf[k] = (H > 1 && j < W) ? energy[(size_t)W + j] : 0.0f;
    }

    for (int i = 1; i < H; ++i) {
        float ecur[CPT];
        #pragma unroll
        for (int k = 0; k < CPT; ++k) ecur[k] = epf[k];

        const int ni = i + 1;
        if (ni < H) {
            #pragma unroll
            for (int k = 0; k < CPT; ++k) {
                int j = tid + k * nt;
                epf[k] = (j < W) ? energy[(size_t)ni * W + j] : 0.0f;
            }
        }

        signed char* brow = back + (size_t)i * W;
        #pragma unroll
        for (int k = 0; k < CPT; ++k) {
            int j = tid + k * nt;
            if (j < W) dp_cell(prev, curr, brow, j, W, ecur[k]);
        }
        __syncthreads();
        float* tmp = prev; prev = curr; curr = tmp;
    }

    if (tid == 0) {
        int best = 0;
        float bestv = prev[0];
        for (int j = 1; j < W; ++j)
            if (prev[j] < bestv) { bestv = prev[j]; best = j; }
        seam[H - 1] = best;
        for (int i = H - 1; i > 0; --i) {
            best += back[(size_t)i * W + best];   // relative offset
            seam[i - 1] = best;
        }
    }
}

__global__ void remove_seam_kernel(const float* __restrict__ img,
                                   float* __restrict__ out,
                                   const int* __restrict__ seam, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output column, 0..W-2
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < H && x < W - 1) {
        int s = seam[y];
        int src = x < s ? x : x + 1;
        const float* sp = img + ((size_t)y * W + src) * 3;
        float* op = out + ((size_t)y * (W - 1) + x) * 3;
        op[0] = sp[0]; op[1] = sp[1]; op[2] = sp[2];
    }
}

// ---------------------------------------------------------------------------
// Host: dispatch the DP kernel by per-thread column count.
// ---------------------------------------------------------------------------

template <int CPT>
static void launch_dp(const float* energy, signed char* back, int* seam,
                      int H, int W, int nt, size_t shmem) {
    CUDA_CHECK(cudaFuncSetAttribute(seam_dp_kernel<CPT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)shmem));
    seam_dp_kernel<CPT><<<1, nt, shmem>>>(energy, back, seam, H, W);
}

static void run_dp(const float* energy, signed char* back, int* seam,
                   int H, int W, int nt, size_t shmem) {
    int cpt = (W + nt - 1) / nt;
    switch (cpt) {
        case 1: launch_dp<1>(energy, back, seam, H, W, nt, shmem); break;
        case 2: launch_dp<2>(energy, back, seam, H, W, nt, shmem); break;
        case 3: launch_dp<3>(energy, back, seam, H, W, nt, shmem); break;
        case 4: launch_dp<4>(energy, back, seam, H, W, nt, shmem); break;
        case 5: launch_dp<5>(energy, back, seam, H, W, nt, shmem); break;
        case 6: launch_dp<6>(energy, back, seam, H, W, nt, shmem); break;
        case 7: launch_dp<7>(energy, back, seam, H, W, nt, shmem); break;
        case 8: launch_dp<8>(energy, back, seam, H, W, nt, shmem); break;
        default:
            fprintf(stderr, "width %d needs %d cols/thread > MAX_CPT %d\n",
                    W, cpt, MAX_CPT);
            exit(1);
    }
}

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
    if (!pixels) {
        fprintf(stderr, "failed to load %s: %s\n", in_path, stbi_failure_reason());
        return 1;
    }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0);
        return 1;
    }
    // Shared-memory limit: two cost rows of W floats must fit in 96KB.
    const size_t max_shmem = 2 * (size_t)W0 * sizeof(float);
    if (max_shmem > 96 * 1024) {
        fprintf(stderr, "image width %d too wide for shared-memory DP (%zu bytes > 96KB)\n",
                W0, max_shmem);
        return 1;
    }
    // With blockDim.x = 1024 the widest CPT=MAX_CPT image is 1024*MAX_CPT wide.
    if (W0 > 1024 * MAX_CPT) {
        fprintf(stderr, "v6 supports width up to %d (got %d); raise MAX_CPT\n",
                1024 * MAX_CPT, W0);
        return 1;
    }
    printf("loaded %s : %dx%d, removing %d seams\n", in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    // Allocate GPU buffers at the original (max) size; width only shrinks.
    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;          // int8 relative back-pointers
    int* d_seam;
    CUDA_CHECK(cudaMalloc(&d_img, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back, npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam, (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x - 1) / block2d.x, (H + block2d.y - 1) / block2d.y);
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel<<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        int threads = w < 1024 ? ((w + 31) / 32) * 32 : 1024;
        if (threads < 32) threads = 32;
        size_t shmem = 2 * (size_t)w * sizeof(float);
        run_dp(d_energy, d_back, d_seam, H, w, threads, shmem);

        remove_seam_kernel<<<grid2d(w - 1), block2d>>>(d_img, d_img2, d_seam, H, w);
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

    // Download result and write out.
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
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_gray);
    cudaFree(d_energy); cudaFree(d_back); cudaFree(d_seam);
    return 0;
}
