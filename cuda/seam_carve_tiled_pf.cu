// seam_carve_tiled_pf.cu — Tiled multi-SM DP with software-pipelined energy prefetch.
//
// Identical to seam_carve_tiled.cu except the tile's inner row-loop prefetches
// the next row's energy into registers while computing the current row, hiding
// global-load latency the same way seam_carve_v4/v5/v6 do for the single-block
// kernel.
//
// Each thread owns at most 2 columns in the extended window (enough for all
// K∈{40,60,80} × T∈{32,64,128} sweep combinations). The prefetch arrays
// epf[2] / ecur[2] stay in registers; no local-memory spill.
//
// Default: TILE_T=64  STRIP_K=60  NT_TILE=256
// Override at compile time: nvcc -DTILE_T=128 -DSTRIP_K=40 ...
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
             fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(_e)); \
             exit(1); } } while (0)

#ifndef TILE_T
#define TILE_T   64
#endif
#ifndef STRIP_K
#define STRIP_K  60
#endif
#ifndef NT_TILE
#define NT_TILE  256
#endif

// ---------------------------------------------------------------------------
// Standard kernels
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
        int l = x > 0   ? x - 1 : 0;
        int r = x < W-1 ? x + 1 : W - 1;
        int u = y > 0   ? y - 1 : 0;
        int d = y < H-1 ? y + 1 : H - 1;
        float dx = fabsf(gray[(size_t)y*W+r] - gray[(size_t)y*W+l]);
        float dy = fabsf(gray[(size_t)d*W+x] - gray[(size_t)u*W+x]);
        energy[(size_t)y * W + x] = dx + dy;
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

__global__ void init_dp_row_kernel(const float* __restrict__ energy,
                                    float* __restrict__ prev, int W) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < W;
         c += gridDim.x * blockDim.x)
        prev[c] = energy[c];
}

// ---------------------------------------------------------------------------
// Tiled DP kernel with prefetch
// ---------------------------------------------------------------------------
// Each thread owns up to 2 columns in the extended window [ext_s, ext_s+ext_w).
// epf[2]: prefetched energy for the NEXT tile row (register-resident).
// ecur[2]: energy for the CURRENT tile row (copied from epf at start of each iter).
//
// Timeline per tile row t:
//   1. ecur ← epf  (copy register → register, free)
//   2. Issue __ldg load for row t+1 energy → epf  (async global load)
//   3. Compute DP using prev_sh (shared mem) + ecur (register)
//   4. __syncthreads()                          (epf load completes during 3+4)
//   5. Swap prev_sh / curr_sh
__global__ void seam_dp_tile_pf_kernel(
        const float* __restrict__ d_energy,
        signed char* __restrict__ d_back,
        const float* __restrict__ d_prev,
        float* __restrict__ d_next,
        int H, int W, int row_start, int tile_rows)
{
    const int k = blockIdx.x;

    const int col_start = (long long)k       * W / STRIP_K;
    const int col_end   = (long long)(k + 1) * W / STRIP_K;
    const int S         = col_end - col_start;

    const int halo_l = (col_start >= TILE_T) ? TILE_T : col_start;
    const int halo_r = (col_end + TILE_T <= W) ? TILE_T : (W - col_end);
    const int ext_s  = col_start - halo_l;
    const int ext_w  = S + halo_l + halo_r;

    extern __shared__ float sh[];
    float* prev_sh = sh;
    float* curr_sh = sh + ext_w;

    // Load initial prev row into shared memory
    for (int i = threadIdx.x; i < ext_w; i += blockDim.x)
        prev_sh[i] = d_prev[ext_s + i];
    __syncthreads();

    // Prefetch energy for the first tile row (row_start) into registers.
    // Each thread owns positions {tid, tid+NT} in the ext window.
    float epf[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int i = threadIdx.x + n * NT_TILE;
        if (i < ext_w)
            epf[n] = __ldg(&d_energy[(size_t)row_start * W + (ext_s + i)]);
    }

    // Process tile_rows rows
    for (int t = 0; t < tile_rows; ++t) {
        const int row = row_start + t;
        signed char* brow = d_back + (size_t)row * W;

        // Step 1: move prefetch into current (register → register)
        float ecur[2];
        #pragma unroll
        for (int n = 0; n < 2; ++n) ecur[n] = epf[n];

        // Step 2: issue prefetch for next row BEFORE shared-mem compute
        // (global load latency is hidden behind the shared-mem work below)
        const int nr = row + 1;
        if (t + 1 < tile_rows && nr < H) {
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int i = threadIdx.x + n * NT_TILE;
                if (i < ext_w)
                    epf[n] = __ldg(&d_energy[(size_t)nr * W + (ext_s + i)]);
            }
        }

        // Step 3: compute DP (shared memory + registers)
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int i = threadIdx.x + n * NT_TILE;
            if (i >= ext_w) break;
            const int gc = ext_s + i;

            int la = (i > 0 && gc > 0)           ? i - 1 : i;
            int ra = (i < ext_w-1 && gc < W-1)   ? i + 1 : i;

            float best = prev_sh[la]; int arg = la;
            float m    = prev_sh[i];  if (m < best) { best = m; arg = i;  }
            m          = prev_sh[ra]; if (m < best) { best = m; arg = ra; }

            curr_sh[i] = ecur[n] + best;

            const int ci = i - halo_l;
            if (ci >= 0 && ci < S)
                brow[gc] = (signed char)((ext_s + arg) - gc);
        }

        // Step 4: sync (prefetch load completes during this + step 3)
        __syncthreads();

        // Step 5: ping-pong
        float* tmp = prev_sh; prev_sh = curr_sh; curr_sh = tmp;
    }

    // Write valid center strip of last dp row to d_next
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        d_next[col_start + i] = prev_sh[halo_l + i];
}

__global__ void seam_backtrack_kernel(const float* __restrict__ d_prev,
                                       const signed char* __restrict__ d_back,
                                       int* __restrict__ d_seam, int H, int W) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int best = 0; float bestv = d_prev[0];
    for (int j = 1; j < W; ++j)
        if (d_prev[j] < bestv) { bestv = d_prev[j]; best = j; }
    d_seam[H - 1] = best;
    for (int i = H - 1; i > 0; --i) {
        best += d_back[(size_t)i * W + best];
        d_seam[i - 1] = best;
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.png> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    int num_seams        = atoi(argv[2]);
    const char* out_path = argc >= 4 ? argv[3] : "carved.png";

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s : %dx%d  removing %d seams\n", in_path, W0, H, num_seams);
    printf("tiled-pf DP: STRIP_K=%d  TILE_T=%d  NT_TILE=%d\n",
           STRIP_K, TILE_T, NT_TILE);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;
    float *d_prev, *d_next;

    CUDA_CHECK(cudaMalloc(&d_img,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,   npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,   npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back,   npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam,   (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,   (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next,   (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };
    auto tile_shmem = [&](int w) -> size_t {
        int s = (w + STRIP_K - 1) / STRIP_K;
        return 2 * (size_t)(s + 2 * TILE_T) * sizeof(float);
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel   <<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);
        init_dp_row_kernel<<<(w + 255)/256, 256>>>(d_energy, d_prev, w);

        const size_t sh = tile_shmem(w);
        int row = 1;
        while (row < H) {
            int tr = (row + TILE_T <= H) ? TILE_T : (H - row);
            seam_dp_tile_pf_kernel<<<STRIP_K, NT_TILE, sh>>>(
                d_energy, d_back, d_prev, d_next, H, w, row, tr);
            float* tmp = d_prev; d_prev = d_next; d_next = tmp;
            row += tr;
        }

        seam_backtrack_kernel<<<1, 1>>>(d_prev, d_back, d_seam, H, w);
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
        float v = h_out_f[i] * 255.0f + 0.5f;
        h_out[i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    if (!stbi_write_png(out_path, w, H, 3, h_out, w * 3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    free(h_img); free(h_out_f); free(h_out);
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_gray);
    cudaFree(d_energy); cudaFree(d_back); cudaFree(d_seam);
    cudaFree(d_prev); cudaFree(d_next);
    return 0;
}
