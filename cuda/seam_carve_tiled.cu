// seam_carve_tiled.cu — Tiled multi-SM DP seam carving.
//
//   ./seam_carve_tiled <input.png> <num_seams> [output.png]
//
// Motivation: the single-block DP in v6/wide occupies 1 of 80 V100 SMs.
// At 8K (W=7680, H=4320) that leaves 79 SMs idle. Naive (v0) uses all SMs by
// launching one kernel per row, but 4320 launches/seam costs ~13 ms in
// synchronization overhead alone.
//
// This kernel divides W columns into STRIP_K strips (one block each) and
// processes TILE_T rows per tile kernel launch, reducing launches per seam
// from H to ceil(H/TILE_T) while keeping all SMs active.
//
// Correctness (halo argument): each block loads TILE_T extra columns on each
// side as a "halo". After T rows, the error from the halo boundary propagates
// at most T columns inward — exactly into the halo, leaving the center strip
// [col_start, col_end) correct. Only center-strip back-pointers are written to
// global memory; the union of all K center strips covers [0, W) without overlap.
//
// Ping-pong buffers (d_prev / d_next): block k reads from d_prev before
// writing to d_next, eliminating any read-after-write race between blocks.
//
// Tunable at compile time: TILE_T, STRIP_K, NT_TILE.
// Default: TILE_T=64, STRIP_K=60, NT_TILE=256.
//   8K: ceil(4319/64)=68 launches  vs  4320 (Naive) or 1 (Wide).
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
// Standard grayscale / energy / remove-seam kernels (same as wide.cu)
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

// ---------------------------------------------------------------------------
// Tiled DP kernels
// ---------------------------------------------------------------------------

// Seed the ping-pong buffer with energy row 0 (DP base case, no back-pointer).
__global__ void init_dp_row_kernel(const float* __restrict__ energy,
                                    float* __restrict__ prev, int W) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < W;
         c += gridDim.x * blockDim.x)
        prev[c] = energy[c];
}

// Tile DP kernel — called once per group of TILE_T rows.
//
// Each block (blockIdx.x = k) owns center strip [col_start, col_end) and loads
// a halo of up to TILE_T columns on each side into shared memory.  After
// processing tile_rows rows entirely in shared memory, the block:
//   1. Writes the last dp row (center strip only) to d_next.
//   2. Has already written back-pointers (center strip, all tile rows) to d_back.
//
// d_prev and d_next are distinct ping-pong buffers → no read/write races.
__global__ void seam_dp_tile_kernel(
        const float* __restrict__ d_energy,   // [H × W]
        signed char* __restrict__ d_back,     // [H × W] int8 relative offsets
        const float* __restrict__ d_prev,     // [W] dp row before this tile
        float* __restrict__ d_next,           // [W] dp last row of this tile
        int H, int W, int row_start, int tile_rows)
{
    const int k = blockIdx.x;

    // Center strip for this block
    const int col_start = (long long)k       * W / STRIP_K;
    const int col_end   = (long long)(k + 1) * W / STRIP_K;
    const int S         = col_end - col_start;

    // Halo: up to TILE_T columns on each side, clamped at image edges
    const int halo_l  = (col_start >= TILE_T) ? TILE_T : col_start;
    const int halo_r  = (col_end + TILE_T <= W) ? TILE_T : (W - col_end);
    const int ext_s   = col_start - halo_l;
    const int ext_w   = S + halo_l + halo_r;   // ≤ S + 2*TILE_T

    // Shared memory: two rows of ext_w floats (ping-pong)
    extern __shared__ float sh[];
    float* prev_sh = sh;
    float* curr_sh = sh + ext_w;

    // Load d_prev[ext_s .. ext_s+ext_w) into prev_sh
    for (int i = threadIdx.x; i < ext_w; i += blockDim.x)
        prev_sh[i] = d_prev[ext_s + i];
    __syncthreads();

    // Process tile_rows rows
    for (int t = 0; t < tile_rows; ++t) {
        const int row             = row_start + t;
        const float* __restrict__ erow = d_energy + (size_t)row * W;
        signed char* brow         = d_back   + (size_t)row * W;

        for (int i = threadIdx.x; i < ext_w; i += blockDim.x) {
            const int gc = ext_s + i;   // global column

            // Neighbor local indices (clamped at shared-mem window AND image edge)
            int la = (i > 0 && gc > 0)       ? i - 1 : i;
            int ra = (i < ext_w-1 && gc < W-1) ? i + 1 : i;

            // DP: left→centre→right tie-break (strict <, leftmost wins)
            float best = prev_sh[la]; int arg = la;
            float m    = prev_sh[i];  if (m < best) { best = m; arg = i;  }
            m          = prev_sh[ra]; if (m < best) { best = m; arg = ra; }

            curr_sh[i] = erow[gc] + best;

            // Write back-pointer for center strip only
            const int ci = i - halo_l;
            if (ci >= 0 && ci < S)
                brow[gc] = (signed char)((ext_s + arg) - gc);  // ∈ {-1, 0, +1}
        }
        __syncthreads();

        // Ping-pong swap inside shared memory
        float* tmp = prev_sh; prev_sh = curr_sh; curr_sh = tmp;
    }

    // Write last dp row (center strip) to d_next
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        d_next[col_start + i] = prev_sh[halo_l + i];
}

// Find minimum of final dp row and backtrack (thread 0).
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
    const char* in_path   = argv[1];
    int num_seams         = atoi(argv[2]);
    const char* out_path  = argc >= 4 ? argv[3] : "carved.png";

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s : %dx%d  removing %d seams\n", in_path, W0, H, num_seams);
    printf("tiled DP: STRIP_K=%d  TILE_T=%d  NT_TILE=%d\n", STRIP_K, TILE_T, NT_TILE);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;
    float *d_prev, *d_next;   // ping-pong DP cost rows

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

    // Shared memory per tile block: 2 rows × (S + 2*TILE_T) floats.
    // Worst case: S = ceil(W/STRIP_K), so ext_w ≤ ceil(W/STRIP_K) + 2*TILE_T.
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

        // Seed DP with row 0
        init_dp_row_kernel<<<(w + 255)/256, 256>>>(d_energy, d_prev, w);

        // Tile DP: ceil((H-1) / TILE_T) kernel launches
        const size_t sh = tile_shmem(w);
        int row = 1;
        while (row < H) {
            int rows_this_tile = (row + TILE_T <= H) ? TILE_T : (H - row);
            seam_dp_tile_kernel<<<STRIP_K, NT_TILE, sh>>>(
                d_energy, d_back, d_prev, d_next, H, w, row, rows_this_tile);
            float* tmp = d_prev; d_prev = d_next; d_next = tmp;
            row += rows_this_tile;
        }

        // Backtrack seam from final dp row
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
