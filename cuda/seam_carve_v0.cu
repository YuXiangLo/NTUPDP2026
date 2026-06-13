// seam_carve_v0.cu - Standalone CUDA vertical seam carving, v0 (NAIVE baseline).
//
//   ./seam_carve_v0 <input.jpg> <num_seams> [output.png]
//
// This is the deliberately-unoptimized baseline that mirrors the original
// PyTorch CUDA_v0: the cumulative-cost DP is computed ONE ROW PER KERNEL LAUNCH.
// For each of the H-1 rows we launch a fresh grid that reads the previous row's
// cost from global memory and writes this row's cost + back-pointer back to
// global memory. So every seam costs H kernel launches and H global round-trips
// of the whole cost matrix -- launch-overhead / memory-latency bound, with zero
// data reuse. v2 collapses this whole loop into ONE kernel (shared-memory
// ping-pong, __syncthreads per row); v4/v5/v6 then optimize that fused kernel.
//
// Comparing v0 -> v6 isolates the real end-to-end win (fusion + latency hiding)
// in one consistent toolchain, no torch/python needed.
//
// Target: NVIDIA V100 (sm_70).

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
// grayscale / energy / remove_seam are identical to every other version.
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

// Row 0 of the cost matrix is just the energy of row 0.
__global__ void copy_row0_kernel(const float* __restrict__ energy,
                                 float* __restrict__ cost, int W) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < W) cost[j] = energy[j];
}

// One DP row: cost[i][j] = energy[i][j] + min(cost[i-1][j-1..j+1]).
// Launched once per row from the host -> the kernel boundary IS the row barrier.
__global__ void dp_row_kernel(const float* __restrict__ energy,
                              float* __restrict__ cost, int* __restrict__ back,
                              int i, int W) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= W) return;
    const float* prev = cost + (size_t)(i - 1) * W;
    const int la = j > 0 ? j - 1 : 0;
    const int ra = j < W - 1 ? j + 1 : W - 1;
    float c = prev[la];
    int arg = la;
    float m = prev[j];
    if (m < c) { c = m; arg = j; }
    m = prev[ra];
    if (m < c) { c = m; arg = ra; }
    cost[(size_t)i * W + j] = energy[(size_t)i * W + j] + c;
    back[(size_t)i * W + j] = arg;
}

// Single-thread argmin on the last row + backtrack (cheap relative to the DP).
__global__ void backtrack_kernel(const float* __restrict__ cost,
                                 const int* __restrict__ back,
                                 int* __restrict__ seam, int H, int W) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const float* last = cost + (size_t)(H - 1) * W;
    int best = 0;
    float bestv = last[0];
    for (int j = 1; j < W; ++j)
        if (last[j] < bestv) { bestv = last[j]; best = j; }
    seam[H - 1] = best;
    for (int i = H - 1; i > 0; --i) {
        best = back[(size_t)i * W + best];
        seam[i - 1] = best;
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

    // Allocate at the original (max) size; width only shrinks.
    float *d_img, *d_img2, *d_gray, *d_energy, *d_cost;
    int *d_back, *d_seam;
    CUDA_CHECK(cudaMalloc(&d_img, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, npix * sizeof(float)));   // full cost matrix
    CUDA_CHECK(cudaMalloc(&d_back, npix * sizeof(int)));     // full back matrix
    CUDA_CHECK(cudaMalloc(&d_seam, (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x - 1) / block2d.x, (H + block2d.y - 1) / block2d.y);
    };
    const int tb = 256;  // 1D block for the per-row DP launches

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel<<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        // Naive DP: one kernel launch per row (the kernel boundary is the barrier).
        int g1 = (w + tb - 1) / tb;
        copy_row0_kernel<<<g1, tb>>>(d_energy, d_cost, w);
        for (int i = 1; i < H; ++i)
            dp_row_kernel<<<g1, tb>>>(d_energy, d_cost, d_back, i, w);
        backtrack_kernel<<<1, 1>>>(d_cost, d_back, d_seam, H, w);

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
    cudaFree(d_energy); cudaFree(d_cost); cudaFree(d_back); cudaFree(d_seam);
    return 0;
}
