// seam_carve_batch.cu — Approximate batch multi-seam removal via Tiled DP.
//
// Single mode: one seam per DP pass (exact, like seam_carve_tiled_pf).
// Batch  mode: STRIP_K seams per DP pass.
//   Each strip backtracks its own local-minimum cost path → K approximate seams
//   extracted in one DP pass. Seam columns sorted per row on GPU, then removed
//   in a single batch-remove kernel (one read pass over the image).
//
//   Per-batch cost ≈ DP + sort + batch_remove   (vs K × DP in single mode).
//   Speedup is driven by amortising the dominant DP cost across K seams.
//
// Compile:
//   nvcc -O3 -arch=sm_70 seam_carve_batch.cu -o seam_carve_batch
// Run:
//   srun -N1 -n1 --gpus-per-node 1 -A ACD115083 -t 10 \
//       ./seam_carve_batch in.png <num_seams> [out.png] [single|batch]

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
                     cudaGetErrorString(_e)); exit(1); } } while (0)

#ifndef TILE_T
#define TILE_T  64
#endif
#ifndef STRIP_K
#define STRIP_K 60
#endif
#ifndef NT_TILE
#define NT_TILE 256
#endif

// ---------------------------------------------------------------------------
// Standard kernels (identical to seam_carve_tiled_pf.cu)
// ---------------------------------------------------------------------------

__global__ void grayscale_kernel(const float* __restrict__ img,
                                  float* __restrict__ gray, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        const float* p = img + ((size_t)y * W + x) * 3;
        gray[(size_t)y * W + x] = 0.299f*p[0] + 0.587f*p[1] + 0.114f*p[2];
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

__global__ void remove_seam_kernel(const float* __restrict__ img,
                                    float* __restrict__ out,
                                    const int* __restrict__ seam, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < H && x < W-1) {
        int s = seam[y], src = x < s ? x : x+1;
        const float* sp = img  + ((size_t)y * W      + src) * 3;
        float*       op = out  + ((size_t)y * (W-1)  + x  ) * 3;
        op[0]=sp[0]; op[1]=sp[1]; op[2]=sp[2];
    }
}

__global__ void init_dp_row_kernel(const float* __restrict__ energy,
                                    float* __restrict__ prev, int W) {
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < W;
         c += gridDim.x * blockDim.x)
        prev[c] = energy[c];
}

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

    for (int i = threadIdx.x; i < ext_w; i += blockDim.x)
        prev_sh[i] = d_prev[ext_s + i];
    __syncthreads();

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
            int la = (i > 0     && gc > 0  ) ? i-1 : i;
            int ra = (i < ext_w-1 && gc < W-1) ? i+1 : i;
            float best = prev_sh[la]; int arg = la;
            float m = prev_sh[i];  if (m < best) { best=m; arg=i;  }
            m = prev_sh[ra]; if (m < best) { best=m; arg=ra; }
            curr_sh[i] = ecur[n] + best;
            const int ci = i - halo_l;
            if (ci >= 0 && ci < S)
                brow[gc] = (signed char)((ext_s + arg) - gc);
        }
        __syncthreads();
        float* tmp = prev_sh; prev_sh = curr_sh; curr_sh = tmp;
    }

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
    d_seam[H-1] = best;
    for (int i = H-1; i > 0; --i) {
        best += d_back[(size_t)i * W + best];
        d_seam[i-1] = best;
    }
}

// ---------------------------------------------------------------------------
// Batch kernels
// ---------------------------------------------------------------------------

// batch_backtrack_kernel — one thread per strip.
// Each thread finds the strip-local minimum in d_prev, then backtracks through
// d_back to recover a full H-pixel seam path.
//
// Output: d_seams[k * H + row] = seam column for strip k at image row `row`.
//         Layout is strip-major so that strip k's seam is contiguous in memory.
__global__ void batch_backtrack_kernel(
        const float* __restrict__ d_prev,
        const signed char* __restrict__ d_back,
        int* __restrict__ d_seams,   // STRIP_K × H, strip-major
        int H, int W, int batch_sz)  // batch_sz ≤ STRIP_K
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= batch_sz) return;

    const int col_start = (long long)k       * W / STRIP_K;
    const int col_end   = (long long)(k + 1) * W / STRIP_K;

    // Find strip-local minimum
    int best = col_start; float bestv = d_prev[col_start];
    for (int j = col_start + 1; j < col_end; ++j)
        if (d_prev[j] < bestv) { bestv = d_prev[j]; best = j; }

    d_seams[k * H + (H-1)] = best;
    for (int i = H-1; i > 0; --i) {
        best += d_back[(size_t)i * W + best];
        d_seams[k * H + (i-1)] = best;
    }
}

// sort_seams_kernel — one thread per image row.
// Transposes from strip-major (k × H) to row-major (H × batch_sz),
// then insertion-sorts the batch_sz column values for this row.
// K=60 elements, nearly pre-sorted by strip index → O(K) average.
__global__ void sort_seams_kernel(
        const int* __restrict__ d_seams,   // STRIP_K × H, strip-major
        int* __restrict__ d_sorted,         // H × batch_sz, row-major, sorted
        int H, int batch_sz)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) return;

    // Gather
    int tmp[STRIP_K];
    for (int k = 0; k < batch_sz; ++k) tmp[k] = d_seams[k * H + row];

    // Insertion sort (K≤60, strip order already approximately sorted)
    for (int i = 1; i < batch_sz; ++i) {
        int v = tmp[i], j = i-1;
        while (j >= 0 && tmp[j] > v) { tmp[j+1] = tmp[j]; --j; }
        tmp[j+1] = v;
    }

    // Write sorted row
    for (int k = 0; k < batch_sz; ++k) d_sorted[row * batch_sz + k] = tmp[k];
}

// batch_remove_kernel — remove batch_sz seams in one read/write pass.
// For each output pixel (row, dst_col), find source column by counting how many
// sorted seam columns lie ≤ source col (linear scan, O(batch_sz) = O(60)).
// Total memory traffic: read W×H once, write (W-batch_sz)×H once — independent of K.
__global__ void batch_remove_kernel(
        const float* __restrict__ img,      // H × W × 3
        float* __restrict__ out,            // H × (W-K) × 3
        const int* __restrict__ sorted_seams, // H × batch_sz, sorted per row
        int H, int W, int batch_sz)
{
    int dst_col = blockIdx.x * blockDim.x + threadIdx.x;
    int row     = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= H || dst_col >= W - batch_sz) return;

    const int* row_seams = sorted_seams + row * batch_sz;

    // Map dst_col → src_col: skip over removed seam columns.
    // Sorted seams: iterate and bump src for each seam ≤ current src.
    int src = dst_col;
    for (int j = 0; j < batch_sz; ++j) {
        if (row_seams[j] <= src) src++;
        else break;  // seams are sorted; no further adjustment needed
    }

    const float* sp = img + ((size_t)row * W         + src    ) * 3;
    float*       op = out + ((size_t)row * (W - batch_sz) + dst_col) * 3;
    op[0]=sp[0]; op[1]=sp[1]; op[2]=sp[2];
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

static size_t tile_shmem(int w) {
    int s = (w + STRIP_K - 1) / STRIP_K;
    return 2 * (size_t)(s + 2 * TILE_T) * sizeof(float);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.png> <num_seams> [out.png] [single|batch]\n",
                argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    int num_seams        = atoi(argv[2]);
    const char* out_path = argc >= 4 ? argv[3] : "carved_batch.png";
    bool batch_mode      = argc >= 5 && strcmp(argv[4], "batch") == 0;

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }

    printf("image: %dx%d   removing %d seams   mode: %s\n",
           W0, H, num_seams, batch_mode ? "batch" : "single");
    printf("kernel params: STRIP_K=%d  TILE_T=%d  NT_TILE=%d\n",
           STRIP_K, TILE_T, NT_TILE);

    // Convert to float
    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    // GPU buffers
    float *d_img, *d_img2, *d_gray, *d_energy;
    signed char* d_back;
    int* d_seam;               // single-mode: H ints
    int* d_seams_flat;         // batch-mode: STRIP_K × H ints (strip-major)
    int* d_sorted_seams;       // batch-mode: H × STRIP_K ints (row-major, sorted)
    float *d_prev, *d_next;

    CUDA_CHECK(cudaMalloc(&d_img,         npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,        npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,        npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy,      npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back,        npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_seam,        (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seams_flat,  (size_t)STRIP_K * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_seams,(size_t)H * STRIP_K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,        (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next,        (size_t)W0 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_img);

    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int w = W0;

    if (!batch_mode) {
        // ---- Single mode: one seam per DP pass ----
        for (int s = 0; s < num_seams; ++s) {
            grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
            energy_kernel   <<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);
            init_dp_row_kernel<<<(w+255)/256, 256>>>(d_energy, d_prev, w);

            const size_t sh = tile_shmem(w);
            int row = 1;
            while (row < H) {
                int tr = (row + TILE_T <= H) ? TILE_T : (H - row);
                seam_dp_tile_pf_kernel<<<STRIP_K, NT_TILE, sh>>>(
                    d_energy, d_back, d_prev, d_next, H, w, row, tr);
                float* tmp = d_prev; d_prev = d_next; d_next = tmp;
                row += tr;
            }

            seam_backtrack_kernel<<<1,1>>>(d_prev, d_back, d_seam, H, w);
            remove_seam_kernel<<<grid2d(w-1), block2d>>>(d_img, d_img2, d_seam, H, w);
            float* tmp = d_img; d_img = d_img2; d_img2 = tmp;
            --w;
        }
    } else {
        // ---- Batch mode: STRIP_K seams per DP pass ----
        int seams_done = 0;
        while (seams_done < num_seams) {
            int batch_sz = (num_seams - seams_done < STRIP_K)
                           ? (num_seams - seams_done) : STRIP_K;

            // Energy pipeline (once per batch)
            grayscale_kernel<<<grid2d(w), block2d>>>(d_img, d_gray, H, w);
            energy_kernel   <<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);
            init_dp_row_kernel<<<(w+255)/256, 256>>>(d_energy, d_prev, w);

            // DP (once per batch)
            const size_t sh = tile_shmem(w);
            int row = 1;
            while (row < H) {
                int tr = (row + TILE_T <= H) ? TILE_T : (H - row);
                seam_dp_tile_pf_kernel<<<STRIP_K, NT_TILE, sh>>>(
                    d_energy, d_back, d_prev, d_next, H, w, row, tr);
                float* tmp = d_prev; d_prev = d_next; d_next = tmp;
                row += tr;
            }

            // Batch backtrack: batch_sz threads, one per strip
            {
                int blk = ((batch_sz + 31) / 32) * 32;  // round up to warp
                batch_backtrack_kernel<<<1, blk>>>(
                    d_prev, d_back, d_seams_flat, H, w, batch_sz);
            }

            // Sort seam columns per row: one thread per row
            {
                int blk = 128;
                int grd = (H + blk - 1) / blk;
                sort_seams_kernel<<<grd, blk>>>(
                    d_seams_flat, d_sorted_seams, H, batch_sz);
            }

            // Batch remove: 2D grid over output pixels
            {
                dim3 bk(32, 8);
                dim3 gd((w - batch_sz + bk.x - 1) / bk.x,
                        (H + bk.y - 1) / bk.y);
                batch_remove_kernel<<<gd, bk>>>(
                    d_img, d_img2, d_sorted_seams, H, w, batch_sz);
                float* tmp = d_img; d_img = d_img2; d_img2 = tmp;
            }

            w -= batch_sz;
            seams_done += batch_sz;
        }
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("GPU carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
           ms, ms / num_seams, w, H);

    // Write output
    const size_t out_pix = (size_t)w * H;
    float* h_out_f = (float*)malloc(out_pix * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out_f, d_img, out_pix * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    unsigned char* h_out = (unsigned char*)malloc(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = h_out_f[i] * 255.0f + 0.5f;
        h_out[i] = (v < 0 ? 0 : v > 255 ? 255 : (unsigned char)v);
    }
    if (!stbi_write_png(out_path, w, H, 3, h_out, w*3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    free(h_out_f); free(h_out);
    cudaFree(d_img); cudaFree(d_img2); cudaFree(d_gray);
    cudaFree(d_energy); cudaFree(d_back); cudaFree(d_seam);
    cudaFree(d_seams_flat); cudaFree(d_sorted_seams);
    cudaFree(d_prev); cudaFree(d_next);
    return 0;
}
