// seam_carve_v3.cu - Standalone CUDA vertical seam carving, v3.
//
//   ./seam_carve_v3 <input.jpg> <num_seams> [output.png]
//
// Same pipeline as v2 (seam_carve.cu), but the cumulative-cost DP is spread
// across ALL SMs instead of running in a single block. v2 profiled at ~1.46
// ms/seam with the DP using grid (1,1,1) = 1 of 80 SMs and ~0.79% memory
// throughput (latency-bound on the serial 968-row chain). v3 launches the DP
// as a cooperative kernel: the cost rows live in two global ping-pong buffers
// so blocks on different SMs can share them, and a device-wide grid.sync()
// replaces __syncthreads() between rows.
//
// NOTE: cooperative-groups grid.sync() needs relocatable device code
// (-rdc=true, set in the Makefile). Whether this actually beats v2 is the open
// question — the per-row grid.sync() barrier is more expensive than a
// __syncthreads(), so the 968 serial barriers may eat the multi-SM gain. Let
// Nsight decide; if it doesn't win, the next step is blocked/wavefront tiling.
//
// Target: NVIDIA V100 (sm_70).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels  (grayscale / energy / remove_seam are identical to v2)
// ---------------------------------------------------------------------------

// Image is stored row-major, channel-last: img[(y*W + x)*3 + c], values in [0,1].

__global__ void grayscale_kernel(const float* __restrict__ img,
                                 float* __restrict__ gray, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        const float* p = img + ((size_t)y * W + x) * 3;
        gray[(size_t)y * W + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

// Energy = |gray[x+1]-gray[x-1]| + |gray[y+1]-gray[y-1]|, edges clamped.
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

// v3 DP: spread across ALL SMs. Cost rows live in global memory (two ping-pong
// buffers) so blocks on different SMs share them, and a device-wide grid.sync()
// replaces __syncthreads() between rows. Threads grid-stride over the columns.
// Launched via cudaLaunchCooperativeKernel.
__global__ void seam_dp_coop_kernel(const float* __restrict__ energy,
                                    int* __restrict__ back,
                                    float* __restrict__ cost_a,
                                    float* __restrict__ cost_b,
                                    int* __restrict__ seam, int H, int W) {
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    float* prev = cost_a;
    float* curr = cost_b;

    for (int j = tid; j < W; j += nthreads) prev[j] = energy[j];
    grid.sync();

    for (int i = 1; i < H; ++i) {
        const float* erow = energy + (size_t)i * W;
        int* brow = back + (size_t)i * W;
        for (int j = tid; j < W; j += nthreads) {
            const int la = j > 0 ? j - 1 : 0;
            const int ra = j < W - 1 ? j + 1 : W - 1;
            float c = prev[la];
            int arg = la;
            float m = prev[j];
            if (m < c) { c = m; arg = j; }
            m = prev[ra];
            if (m < c) { c = m; arg = ra; }
            curr[j] = erow[j] + c;
            brow[j] = arg;
        }
        grid.sync();
        float* tmp = prev; prev = curr; curr = tmp;
    }

    if (tid == 0) {
        int best = 0;
        float bestv = prev[0];
        for (int j = 1; j < W; ++j)
            if (prev[j] < bestv) { bestv = prev[j]; best = j; }
        seam[H - 1] = best;
        for (int i = H - 1; i > 0; --i) {
            best = back[(size_t)i * W + best];
            seam[i - 1] = best;
        }
    }
}

// Copy each row skipping the seam column -> compact image of width W-1.
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
    if (!pixels) {
        fprintf(stderr, "failed to load %s: %s\n", in_path, stbi_failure_reason());
        return 1;
    }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0);
        return 1;
    }
    printf("loaded %s : %dx%d, removing %d seams\n", in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    // Allocate GPU buffers at the original (max) size; width only shrinks.
    float *d_img, *d_img2, *d_gray, *d_energy;
    int *d_back, *d_seam;
    CUDA_CHECK(cudaMalloc(&d_img, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back, npix * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seam, (size_t)H * sizeof(int)));
    // Cost ping-pong rows (one float per column), global so all SMs share them.
    float *d_costA, *d_costB;
    CUDA_CHECK(cudaMalloc(&d_costA, (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costB, (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x - 1) / block2d.x, (H + block2d.y - 1) / block2d.y);
    };

    // Cooperative launch geometry: at least one block per SM (to use all of
    // them), capped at the max number that can be co-resident (required for
    // grid.sync). Threads grid-stride over the columns.
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int coop_ok = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_ok, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_ok) {
        fprintf(stderr, "device does not support cooperative launch\n");
        return 1;
    }
    const int coop_block = 128;
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (void*)seam_dp_coop_kernel, coop_block, 0));
    int max_blocks = blocks_per_sm * prop.multiProcessorCount;
    int need = (W0 + coop_block - 1) / coop_block;
    int coop_grid = prop.multiProcessorCount > need ? prop.multiProcessorCount : need;
    if (coop_grid > max_blocks) coop_grid = max_blocks;
    printf("v3 cooperative DP: grid=%d block=%d (%d SMs, %d blocks/SM)\n",
           coop_grid, coop_block, prop.multiProcessorCount, blocks_per_sm);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel<<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        void* kargs[] = {&d_energy, &d_back, &d_costA, &d_costB, &d_seam, &H, &w};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)seam_dp_coop_kernel,
                                               dim3(coop_grid), dim3(coop_block),
                                               kargs, 0, 0));

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
    cudaFree(d_costA); cudaFree(d_costB);
    return 0;
}
