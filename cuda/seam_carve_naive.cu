// seam_carve_naive.cu — Per-seam H2D/D2H transfer baseline.
//
//   ./seam_carve_naive <input.jpg> <num_seams> [output.png]
//
// Identical kernels to v5 but each seam does a full round-trip:
//   H2D(current frame) → kernels → D2H(output frame)
// This simulates the "naive integration" pattern where a host application
// calls GPU seam carving one seam at a time without keeping data resident.
//
// Reports TWO timing lines so run_matrix.py / sbatch_gpu.sh can capture either:
//   "GPU carving time: X ms" = total wall time *including* all transfers
//   "GPU kernel time: X ms"  = pure kernel time (no transfers)
//
// Use TRANSFER_MODE=per_seam in run_matrix.py; parse the "carving time" line.

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
// Kernels (identical to v5)
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
        int l = x > 0 ? x - 1 : 0, r = x < W-1 ? x+1 : W-1;
        int u = y > 0 ? y - 1 : 0, d = y < H-1 ? y+1 : H-1;
        float dx = fabsf(gray[(size_t)y*W+r] - gray[(size_t)y*W+l]);
        float dy = fabsf(gray[(size_t)d*W+x] - gray[(size_t)u*W+x]);
        energy[(size_t)y * W + x] = dx + dy;
    }
}

__device__ __forceinline__ void dp_cell(const float* __restrict__ prev,
                                         float* __restrict__ curr,
                                         signed char* __restrict__ brow,
                                         int j, int W, float e) {
    const int la = j > 0 ? j-1 : 0, ra = j < W-1 ? j+1 : W-1;
    float c = prev[la]; int arg = la;
    float m = prev[j]; if (m < c) { c = m; arg = j; }
    m = prev[ra];       if (m < c) { c = m; arg = ra; }
    curr[j] = e + c;
    brow[j] = (signed char)(arg - j);
}

__global__ void seam_dp_kernel(const float* __restrict__ energy,
                                signed char* __restrict__ back,
                                int* __restrict__ seam, int H, int W) {
    extern __shared__ float sh[];
    float* prev = sh, *curr = sh + W;
    const int nthreads = blockDim.x;
    const int c0 = threadIdx.x, c1 = threadIdx.x + nthreads;

    if (c0 < W) prev[c0] = energy[c0];
    if (c1 < W) prev[c1] = energy[c1];
    __syncthreads();

    float e0 = (H > 1 && c0 < W) ? energy[(size_t)W + c0] : 0.0f;
    float e1 = (H > 1 && c1 < W) ? energy[(size_t)W + c1] : 0.0f;

    for (int i = 1; i < H; ++i) {
        const float ec0 = e0, ec1 = e1;
        const int ni = i + 1;
        if (ni < H) {
            e0 = (c0 < W) ? energy[(size_t)ni * W + c0] : 0.0f;
            e1 = (c1 < W) ? energy[(size_t)ni * W + c1] : 0.0f;
        }
        signed char* brow = back + (size_t)i * W;
        if (c0 < W) dp_cell(prev, curr, brow, c0, W, ec0);
        if (c1 < W) dp_cell(prev, curr, brow, c1, W, ec1);
        __syncthreads();
        float* tmp = prev; prev = curr; curr = tmp;
    }

    if (threadIdx.x == 0) {
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
    if (W0 > 2048) {
        fprintf(stderr, "naive supports width up to 2048 (got %d); use seam_carve_wide\n", W0);
        return 1;
    }
    printf("loaded %s : %dx%d, removing %d seams\n", in_path, W0, H, num_seams);

    const size_t max_npix = (size_t)W0 * H;

    // Host float buffers — two ping-pong
    float* h_img[2];
    h_img[0] = (float*)malloc(max_npix * 3 * sizeof(float));
    h_img[1] = (float*)malloc(max_npix * 3 * sizeof(float));
    for (size_t i = 0; i < (size_t)W0 * H * 3; ++i)
        h_img[0][i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    // GPU buffers — sized for worst case
    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;
    CUDA_CHECK(cudaMalloc(&d_img,    max_npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,   max_npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,   max_npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, max_npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back,   max_npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam,   (size_t)H * sizeof(int)));

    const size_t max_shmem = 2 * (size_t)W0 * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(seam_dp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_shmem));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };

    // Timers
    cudaEvent_t t_total_0, t_total_1, t_kern_0, t_kern_1;
    CUDA_CHECK(cudaEventCreate(&t_total_0)); CUDA_CHECK(cudaEventCreate(&t_total_1));
    CUDA_CHECK(cudaEventCreate(&t_kern_0));  CUDA_CHECK(cudaEventCreate(&t_kern_1));

    float kern_ms_total = 0.0f;
    int cur = 0;    // which h_img[] slot is current
    int w = W0;

    CUDA_CHECK(cudaEventRecord(t_total_0));

    for (int s = 0; s < num_seams; ++s) {
        const size_t npix = (size_t)w * H;

        // H2D transfer
        CUDA_CHECK(cudaMemcpy(d_img, h_img[cur], npix * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Kernels
        CUDA_CHECK(cudaEventRecord(t_kern_0));
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel<<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        int threads = w < 1024 ? ((w + 31) / 32) * 32 : 1024;
        if (threads < 32) threads = 32;
        size_t shmem = 2 * (size_t)w * sizeof(float);
        seam_dp_kernel<<<1, threads, shmem>>>(d_energy, d_back, d_seam, H, w);

        remove_seam_kernel<<<grid2d(w-1), block2d>>>(d_img, d_img2, d_seam, H, w);
        CUDA_CHECK(cudaEventRecord(t_kern_1));
        CUDA_CHECK(cudaEventSynchronize(t_kern_1));

        float km = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&km, t_kern_0, t_kern_1));
        kern_ms_total += km;

        // D2H transfer (output frame is in d_img2)
        int next = 1 - cur;
        CUDA_CHECK(cudaMemcpy(h_img[next], d_img2, (size_t)(w-1) * H * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cur = next;
        --w;
    }

    CUDA_CHECK(cudaEventRecord(t_total_1));
    CUDA_CHECK(cudaEventSynchronize(t_total_1));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t_total_0, t_total_1));
    CUDA_CHECK(cudaGetLastError());

    // "carving time" = total including transfers (for sbatch harness)
    printf("GPU carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
           total_ms, total_ms / num_seams, w, H);
    printf("GPU kernel time: %.3f ms  (%.4f ms/seam)  [no transfers]\n",
           kern_ms_total, kern_ms_total / num_seams);
    printf("Transfer overhead: %.3f ms  (%.1f%%)\n",
           total_ms - kern_ms_total,
           (total_ms - kern_ms_total) / total_ms * 100.0f);

    // Write output from host buffer
    const size_t out_pix = (size_t)w * H;
    unsigned char* h_out = (unsigned char*)malloc(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = h_img[cur][i] * 255.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        h_out[i] = (unsigned char)(v + 0.5f);
    }
    if (!stbi_write_png(out_path, w, H, 3, h_out, w * 3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    free(h_img[0]); free(h_img[1]); free(h_out);
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_gray);
    cudaFree(d_energy); cudaFree(d_back); cudaFree(d_seam);
    return 0;
}
