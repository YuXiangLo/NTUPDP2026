// seam_carve_persistent.cu — Tiled multi-SM DP as a persistent kernel.
//
// Difference from seam_carve_tiled_pf.cu:
//   All H/T tile steps for ONE seam's DP are fused into a single kernel
//   launch using cooperative_groups::grid.sync() instead of returning to
//   the host between tiles.  This eliminates ~H/T kernel-launch round-trips
//   per seam (67 at 4K T=64, 34 at 8K T=128).
//
//   Ping-pong between two global buffers d_buf[0] and d_buf[1] is selected
//   by tile index parity; no host pointer swap is required.
//
// Occupancy requirement for cooperative launch:
//   All STRIP_K blocks (60 × 256 = 15360 threads) must be simultaneously
//   resident.  V100 max = 80 SM × 2048 = 163840 threads — well within limit.
//
// Default: TILE_T=64  STRIP_K=60  NT_TILE=256
// Override: nvcc -DTILE_T=128 -DSTRIP_K=60 ...

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

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
// Standard kernels (unchanged from tiled_pf)
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
// Persistent tiled DP kernel with cooperative groups + prefetch
// ---------------------------------------------------------------------------
// One launch handles ALL tile steps for a single seam's DP.
// grid.sync() replaces the H/T host-side kernel re-launches.
//
// Ping-pong: d_buf[tile%2] → read (d_prev), d_buf[1-tile%2] → write (d_next).
// Row 0 initialised inside the kernel (no separate init_dp_row_kernel needed).
__global__ void seam_dp_persistent_pf_kernel(
        const float* __restrict__ d_energy,
        signed char* __restrict__ d_back,
        float* __restrict__ d_buf0,
        float* __restrict__ d_buf1,
        int H, int W)
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    const int k = blockIdx.x;

    const int col_start = (long long)k       * W / STRIP_K;
    const int col_end   = (long long)(k + 1) * W / STRIP_K;
    const int S         = col_end - col_start;

    const int halo_l = (col_start >= TILE_T) ? TILE_T : col_start;
    const int halo_r = (col_end + TILE_T <= W) ? TILE_T : (W - col_end);
    const int ext_s  = col_start - halo_l;
    const int ext_w  = S + halo_l + halo_r;

    // Initialize row 0: each block writes its center strip to d_buf0.
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        d_buf0[col_start + i] = d_energy[col_start + i];

    grid.sync();   // all blocks have written row 0 before any tile starts

    extern __shared__ float sh[];

    const int num_tiles = (H - 1 + TILE_T - 1) / TILE_T;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int row_start = 1 + tile * TILE_T;
        const int tile_rows = (row_start + TILE_T <= H) ? TILE_T : (H - row_start);

        // Ping-pong selection by tile parity
        const float* d_prev = (tile % 2 == 0) ? d_buf0 : d_buf1;
        float*       d_next = (tile % 2 == 0) ? d_buf1 : d_buf0;

        // --- Tiled DP with prefetch (same logic as tiled_pf kernel) --------

        float* prev_sh = sh;
        float* curr_sh = sh + ext_w;

        // Load previous dp row into shared memory
        for (int i = threadIdx.x; i < ext_w; i += blockDim.x)
            prev_sh[i] = d_prev[ext_s + i];
        __syncthreads();

        // Prefetch row_start energy into registers
        float epf[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int i = threadIdx.x + n * NT_TILE;
            if (i < ext_w)
                epf[n] = __ldg(&d_energy[(size_t)row_start * W + (ext_s + i)]);
        }

        for (int t = 0; t < tile_rows; ++t) {
            const int row = row_start + t;
            signed char* brow = d_back + (size_t)row * W;

            float ecur[2];
            #pragma unroll
            for (int n = 0; n < 2; ++n) ecur[n] = epf[n];

            const int nr = row + 1;
            if (t + 1 < tile_rows && nr < H) {
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    int i = threadIdx.x + n * NT_TILE;
                    if (i < ext_w)
                        epf[n] = __ldg(&d_energy[(size_t)nr * W + (ext_s + i)]);
                }
            }

            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int i = threadIdx.x + n * NT_TILE;
                if (i >= ext_w) break;
                const int gc = ext_s + i;

                int la = (i > 0 && gc > 0)         ? i - 1 : i;
                int ra = (i < ext_w-1 && gc < W-1) ? i + 1 : i;

                float best = prev_sh[la]; int arg = la;
                float m    = prev_sh[i];  if (m < best) { best = m; arg = i;  }
                m          = prev_sh[ra]; if (m < best) { best = m; arg = ra; }

                curr_sh[i] = ecur[n] + best;

                const int ci = i - halo_l;
                if (ci >= 0 && ci < S)
                    brow[gc] = (signed char)((ext_s + arg) - gc);
            }

            __syncthreads();
            float* tmp = prev_sh; prev_sh = curr_sh; curr_sh = tmp;
        }

        // Write valid center strip to d_next
        for (int i = threadIdx.x; i < S; i += blockDim.x)
            d_next[col_start + i] = prev_sh[halo_l + i];

        grid.sync();   // all blocks finish tile before next tile starts
    }
    // After loop: final dp row is in d_buf[num_tiles % 2]
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
    printf("persistent DP: STRIP_K=%d  TILE_T=%d  NT_TILE=%d\n",
           STRIP_K, TILE_T, NT_TILE);

    // Check cooperative launch support
    int supports_coop = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop,
                                      cudaDevAttrCooperativeLaunch, 0));
    if (!supports_coop) {
        fprintf(stderr, "device does not support cooperative launch\n");
        return 1;
    }

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;
    float *d_buf0, *d_buf1;

    CUDA_CHECK(cudaMalloc(&d_img,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,   npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,   npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back,   npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam,   (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_buf0,   (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buf1,   (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };

    // Shared memory per block: 2 × ext_w floats.
    // Compute worst-case ext_w for this image (strip 0 or last strip has smaller S
    // but halos don't change). Max strip width S_max = ceil(W0 / STRIP_K) + 1.
    auto tile_shmem = [&](int w) -> size_t {
        int s = (w + STRIP_K - 1) / STRIP_K;
        return 2 * (size_t)(s + 2 * TILE_T) * sizeof(float);
    };

    // Verify occupancy: cooperative launch requires all blocks simultaneously resident.
    {
        int max_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, seam_dp_persistent_pf_kernel, NT_TILE, tile_shmem(W0)));
        int num_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
        if (max_blocks * num_sm < STRIP_K) {
            fprintf(stderr,
                "cooperative launch impossible: max %d blocks total, need %d\n",
                max_blocks * num_sm, STRIP_K);
            return 1;
        }
        printf("occupancy OK: %d blocks/SM × %d SMs = %d >= %d needed\n",
               max_blocks, num_sm, max_blocks * num_sm, STRIP_K);
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
        energy_kernel   <<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);

        // Persistent DP: one cooperative launch for all H/T tiles
        const size_t sh = tile_shmem(w);
        void* args[] = { &d_energy, &d_back, &d_buf0, &d_buf1, &H, &w };
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)seam_dp_persistent_pf_kernel,
            STRIP_K, NT_TILE, args, sh, 0));

        // Final dp row is in d_buf[(num_tiles) % 2]
        const int num_tiles = (H - 1 + TILE_T - 1) / TILE_T;
        float* d_dp_final = (num_tiles % 2 == 0) ? d_buf0 : d_buf1;

        seam_backtrack_kernel<<<1, 1>>>(d_dp_final, d_back, d_seam, H, w);
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
    cudaFree(d_buf0); cudaFree(d_buf1);
    return 0;
}
