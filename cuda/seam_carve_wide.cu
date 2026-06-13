// seam_carve_wide.cu — v5 with no 2048-column width limit.
//
//   ./seam_carve_wide <input.jpg> <num_seams> [output.png]
//
// v4/v5 hard-code two columns per thread (c0=tid, c1=tid+blockDim.x), so
// the DP block must fit the entire row in 2 × blockDim.x ≤ 2048 columns.
//
// v_wide replaces this with a grid-stride loop inside a single 1024-thread
// block: each thread handles columns {tid, tid+1024, tid+2048, …} up to W-1.
// Shared memory holds two W-float DP cost buffers (ping-pong) plus the
// int8 back-pointer write is still to global memory.
//
// Shared memory: 2 × W × 4 bytes.
//   4K (W=3840): 30 KB ✓
//   8K (W=7680): 60 KB ✓ (within V100's 96 KB with cudaFuncSetAttribute)
//  12K (W=12288): 96 KB — absolute ceiling for V100.
//
// Grayscale and energy kernels are standard 2-D grids (no width restriction).
// Only the DP kernel is specialized here.
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
// Kernels
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
        int l = x > 0   ? x-1 : 0,  r = x < W-1 ? x+1 : W-1;
        int u = y > 0   ? y-1 : 0,  d = y < H-1 ? y+1 : H-1;
        float dx = fabsf(gray[(size_t)y*W+r] - gray[(size_t)y*W+l]);
        float dy = fabsf(gray[(size_t)d*W+x] - gray[(size_t)u*W+x]);
        energy[(size_t)y * W + x] = dx + dy;
    }
}

// Single-block wide DP kernel.
// Shared memory: [prev_costs | curr_costs], each W floats.
// Each thread handles columns {tid, tid+1024, tid+2048, ...} (grid-stride).
__global__ void seam_dp_wide_kernel(const float* __restrict__ energy,
                                     signed char* __restrict__ back,
                                     int* __restrict__ seam,
                                     int H, int W) {
    extern __shared__ float sh[];
    float* prev = sh;           // W floats — DP costs for previous row
    float* curr = sh + W;       // W floats — DP costs for current row

    const int tid     = threadIdx.x;
    const int nthreads = blockDim.x;  // always 1024

    // --- Load row 0 into prev ---
    for (int c = tid; c < W; c += nthreads)
        prev[c] = energy[c];
    __syncthreads();

    // --- Row-by-row DP ---
    for (int i = 1; i < H; ++i) {
        const float* erow = energy + (size_t)i * W;
        signed char* brow = back  + (size_t)i * W;

        for (int c = tid; c < W; c += nthreads) {
            float e = erow[c];
            int la = c > 0   ? c - 1 : 0;
            int ra = c < W-1 ? c + 1 : W - 1;
            // Tie-break: left < centre < right  (strict <, leftmost wins)
            float best = prev[la]; int arg = la;
            float m    = prev[c];  if (m < best) { best = m; arg = c; }
            m          = prev[ra]; if (m < best) { best = m; arg = ra; }
            curr[c] = e + best;
            brow[c] = (signed char)(arg - c);
        }
        __syncthreads();

        // Swap ping-pong
        float* tmp = prev; prev = curr; curr = tmp;
    }

    // --- Backtrack (thread 0) ---
    if (tid == 0) {
        int best = 0; float bestv = prev[0];
        for (int j = 1; j < W; ++j)
            if (prev[j] < bestv) { bestv = prev[j]; best = j; }
        seam[H - 1] = best;
        for (int i = H - 1; i > 0; --i) {
            best += back[(size_t)i * W + best];
            seam[i - 1] = best;
        }
    }
}

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

    // Shared mem check: 2 × W × 4 bytes must fit in 96 KB
    const size_t shmem_needed = 2 * (size_t)W0 * sizeof(float);
    if (shmem_needed > 96 * 1024) {
        fprintf(stderr,
            "width %d requires %zu bytes shared mem (>96KB V100 limit)\n",
            W0, shmem_needed);
        return 1;
    }
    printf("loaded %s : %dx%d, removing %d seams\n", in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;
    CUDA_CHECK(cudaMalloc(&d_img,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,   npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,   npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back,   npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam,   (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Allow up to 96 KB dynamic shared memory for the wide DP kernel
    CUDA_CHECK(cudaFuncSetAttribute(seam_dp_wide_kernel,
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
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel<<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        // DP: single block, 1024 threads, shared mem = 2*w*sizeof(float)
        size_t shmem = 2 * (size_t)w * sizeof(float);
        seam_dp_wide_kernel<<<1, 1024, shmem>>>(d_energy, d_back, d_seam, H, w);

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
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_gray);
    cudaFree(d_energy); cudaFree(d_back); cudaFree(d_seam);
    return 0;
}
