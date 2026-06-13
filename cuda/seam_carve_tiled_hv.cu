// seam_carve_tiled_hv.cu — Tiled multi-SM DP (prefetch) + horizontal seams via
// a coalesced shared-memory VRAM transpose.
//
// The DP / energy / grayscale / remove kernels are byte-for-byte the same as
// seam_carve_tiled_pf.cu. The only new kernel is transpose_kernel: a classic
// 32x32 tiled, bank-conflict-padded transpose over float3 (RGB) pixels.
//
// Horizontal seam carving = transpose ONCE -> run the vertical pipeline for all
// N seams in transposed space -> transpose ONCE back. The transpose cost is a
// fixed 2-pass overhead amortized over N removals, NOT an N-times penalty.
//
//   ./seam_carve_tiled_hv <in.png> <num_seams> [out.png] [v|h|selftest]
//     v        : vertical seams   (default; identical to seam_carve_tiled_pf)
//     h        : horizontal seams (transpose -> vertical carve -> transpose)
//     selftest : verify transpose(transpose(img)) == img, then exit
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

// Transpose tile geometry (independent of the DP tiling above).
#define TR_DIM   32
#define TR_ROWS  8

// ---------------------------------------------------------------------------
// Coalesced shared-memory transpose of an H x W image of float3 (RGB) pixels.
// Output is a W x H image. Element (y,x) of `in` maps to (x,y) of `out`.
// The shared tile is padded ([TR_DIM][TR_DIM+1]) so the +1 offsets the
// stride-TR_DIM access pattern off the 32 shared-memory banks.
// ---------------------------------------------------------------------------
__global__ void transpose_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int H, int W) {
    __shared__ float3 tile[TR_DIM][TR_DIM + 1];

    int x = blockIdx.x * TR_DIM + threadIdx.x;   // input column [0, W)
    int y = blockIdx.y * TR_DIM + threadIdx.y;   // input row    [0, H)

    // Coalesced read: consecutive threadIdx.x -> consecutive input columns.
    #pragma unroll
    for (int j = 0; j < TR_DIM; j += TR_ROWS) {
        if (x < W && (y + j) < H) {
            const float* p = in + ((size_t)(y + j) * W + x) * 3;
            tile[threadIdx.y + j][threadIdx.x] = make_float3(p[0], p[1], p[2]);
        }
    }
    __syncthreads();

    // Transposed block origin: swap the block's x/y so the write is coalesced.
    int tx = blockIdx.y * TR_DIM + threadIdx.x;  // output column [0, H)
    int ty = blockIdx.x * TR_DIM + threadIdx.y;  // output row    [0, W)

    // Output image is W x H (row-major, stride H), so out[(ty+j)*H + tx].
    #pragma unroll
    for (int j = 0; j < TR_DIM; j += TR_ROWS) {
        if (tx < H && (ty + j) < W) {
            float3 v = tile[threadIdx.x][threadIdx.y + j];
            float* q = out + ((size_t)(ty + j) * H + tx) * 3;
            q[0] = v.x; q[1] = v.y; q[2] = v.z;
        }
    }
}

// ---------------------------------------------------------------------------
// Standard kernels (identical to seam_carve_tiled_pf.cu)
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
    d_seam[H - 1] = best;
    for (int i = H - 1; i > 0; --i) {
        best += d_back[(size_t)i * W + best];
        d_seam[i - 1] = best;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Buffers {
    float *d_gray, *d_energy, *d_prev, *d_next;
    signed char *d_back;
    int *d_seam;
};

// Remove `num_seams` vertical seams from a (H x W) image held in *d_img,
// using *d_img2 as scratch. On return *d_img holds the (H x (W-num_seams))
// result and the function returns the final width. Pointers are swapped to
// always leave the live image in *d_img.
static int carve_vertical(float** d_img, float** d_img2, Buffers& b,
                          int H, int W, int num_seams) {
    const dim3 block2d(32, 8);
    auto grid2d = [&](int w) {
        return dim3((w + block2d.x-1)/block2d.x, (H + block2d.y-1)/block2d.y);
    };
    auto tile_shmem = [&](int w) -> size_t {
        int s = (w + STRIP_K - 1) / STRIP_K;
        return 2 * (size_t)(s + 2 * TILE_T) * sizeof(float);
    };

    int w = W;
    for (int s = 0; s < num_seams; ++s) {
        grayscale_kernel<<<grid2d(w), block2d>>>(*d_img, b.d_gray, H, w);
        energy_kernel   <<<grid2d(w), block2d>>>(b.d_gray, b.d_energy, H, w);
        init_dp_row_kernel<<<(w + 255)/256, 256>>>(b.d_energy, b.d_prev, w);

        const size_t sh = tile_shmem(w);
        int row = 1;
        while (row < H) {
            int tr = (row + TILE_T <= H) ? TILE_T : (H - row);
            seam_dp_tile_pf_kernel<<<STRIP_K, NT_TILE, sh>>>(
                b.d_energy, b.d_back, b.d_prev, b.d_next, H, w, row, tr);
            float* tmp = b.d_prev; b.d_prev = b.d_next; b.d_next = tmp;
            row += tr;
        }

        seam_backtrack_kernel<<<1, 1>>>(b.d_prev, b.d_back, b.d_seam, H, w);
        remove_seam_kernel<<<grid2d(w-1), block2d>>>(*d_img, *d_img2, b.d_seam, H, w);
        float* tmp = *d_img; *d_img = *d_img2; *d_img2 = tmp;
        --w;
    }
    return w;
}

static void transpose(const float* d_in, float* d_out, int H, int W) {
    dim3 block(TR_DIM, TR_ROWS);
    dim3 grid((W + TR_DIM - 1) / TR_DIM, (H + TR_DIM - 1) / TR_DIM);
    transpose_kernel<<<grid, block>>>(d_in, d_out, H, W);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.png> <num_seams> [output.png] [v|h|selftest]\n",
                argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    int num_seams        = atoi(argv[2]);
    const char* out_path = argc >= 4 ? argv[3] : "carved.png";
    const char* mode     = argc >= 5 ? argv[4] : "v";
    const bool horizontal = (strcmp(mode, "h") == 0);
    const bool selftest   = (strcmp(mode, "selftest") == 0);

    int W0, H0, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H0, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }

    const size_t npix = (size_t)W0 * H0;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    printf("loaded %s : %dx%d  mode=%s  removing %d seams\n",
           in_path, W0, H0, mode, num_seams);
    printf("tiled-hv: STRIP_K=%d TILE_T=%d NT_TILE=%d  transpose=%dx%d\n",
           STRIP_K, TILE_T, NT_TILE, TR_DIM, TR_ROWS);

    const int maxdim = W0 > H0 ? W0 : H0;
    float *d_img, *d_img2, *d_imgT;
    Buffers b;
    CUDA_CHECK(cudaMalloc(&d_img,     npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_imgT,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_gray,  npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_energy,npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_back,  npix * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&b.d_seam,  (size_t)maxdim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b.d_prev,  (size_t)maxdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_next,  (size_t)maxdim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // --- self-test: transpose(transpose(img)) must equal img bit-for-bit ---
    if (selftest) {
        transpose(d_img, d_imgT, H0, W0);    // H0xW0 -> W0xH0
        transpose(d_imgT, d_img2, W0, H0);   // W0xH0 -> H0xW0 (round trip)
        CUDA_CHECK(cudaDeviceSynchronize());
        float* h_rt = (float*)malloc(npix * 3 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_rt, d_img2, npix * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        size_t mism = 0; double maxabs = 0.0;
        for (size_t i = 0; i < npix * 3; ++i) {
            double d = (double)h_rt[i] - (double)h_img[i];
            if (d != 0.0) { mism++; if (d < 0) d = -d; if (d > maxabs) maxabs = d; }
        }
        printf("selftest round-trip: %zu / %zu mismatched, max|diff|=%g  -> %s\n",
               mism, npix * 3, maxabs, mism == 0 ? "PASS" : "FAIL");
        free(h_rt); free(h_img);
        return mism == 0 ? 0 : 2;
    }

    if (num_seams <= 0) { fprintf(stderr, "num_seams must be > 0\n"); return 1; }

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    int outW, outH;
    float ms_tr_in = 0.0f, ms_carve = 0.0f, ms_tr_out = 0.0f;
    float* live = nullptr;   // tracks the live buffer across pointer swaps

    if (horizontal) {
        // Horizontal seams: a vertical carve in transposed space.
        if (num_seams >= H0) {
            fprintf(stderr, "num_seams must be in [1, %d) for horizontal\n", H0);
            return 1;
        }
        // transpose H0xW0 -> cH x cW  with cH=W0, cW=H0
        const int cH = W0, cW = H0;
        CUDA_CHECK(cudaEventRecord(e0));
        transpose(d_img, d_imgT, H0, W0);
        CUDA_CHECK(cudaEventRecord(e1));
        // carve in transposed space: live image starts in d_imgT, scratch d_img2
        live = d_imgT;
        int fw = carve_vertical(&live, &d_img2, b, cH, cW, num_seams);  // cH x fw
        CUDA_CHECK(cudaEventRecord(e2));
        // transpose back cH x fw -> fw x cH = (H0-num_seams) x W0
        transpose(live, d_img, cH, fw);
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tr_in,  e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_carve,  e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tr_out, e2, e3));
        outW = W0; outH = fw;   // final original-space image: outH x outW
        // result now in d_img, dimensions outH x outW
    } else {
        if (num_seams >= W0) {
            fprintf(stderr, "num_seams must be in [1, %d) for vertical\n", W0);
            return 1;
        }
        CUDA_CHECK(cudaEventRecord(e1));
        live = d_img;
        int fw = carve_vertical(&live, &d_img2, b, H0, W0, num_seams);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaEventSynchronize(e2));
        CUDA_CHECK(cudaEventElapsedTime(&ms_carve, e1, e2));
        d_img = live;
        outW = fw; outH = H0;
    }

    float ms_total = ms_tr_in + ms_carve + ms_tr_out;
    if (horizontal) {
        printf("transpose-in: %.3f ms   carve: %.3f ms (%.4f ms/seam)   "
               "transpose-out: %.3f ms\n",
               ms_tr_in, ms_carve, ms_carve / num_seams, ms_tr_out);
        printf("HORIZONTAL total: %.3f ms  (transpose overhead %.3f ms = %.2f%%)  "
               "final size %dx%d\n",
               ms_total, ms_tr_in + ms_tr_out,
               100.0 * (ms_tr_in + ms_tr_out) / ms_total, outW, outH);
    } else {
        printf("GPU carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
               ms_carve, ms_carve / num_seams, outW, outH);
    }

    const size_t out_pix = (size_t)outW * outH;
    float* h_out_f = (float*)malloc(out_pix * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out_f, d_img, out_pix * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned char* h_out = (unsigned char*)malloc(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = h_out_f[i] * 255.0f + 0.5f;
        h_out[i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    if (!stbi_write_png(out_path, outW, outH, 3, h_out, outW * 3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    free(h_img); free(h_out_f); free(h_out);
    // The three image allocations may have been permuted across pointer swaps;
    // free the unique set of {d_img, d_img2, d_imgT, live} to avoid double-free.
    float* uniq[4]; int nu = 0;
    float* cand[4] = {d_img, d_img2, d_imgT, live};
    for (int i = 0; i < 4; ++i) {
        if (!cand[i]) continue;
        bool seen = false;
        for (int j = 0; j < nu; ++j) if (uniq[j] == cand[i]) { seen = true; break; }
        if (!seen) uniq[nu++] = cand[i];
    }
    for (int i = 0; i < nu; ++i) cudaFree(uniq[i]);
    cudaFree(b.d_gray); cudaFree(b.d_energy); cudaFree(b.d_back);
    cudaFree(b.d_seam); cudaFree(b.d_prev); cudaFree(b.d_next);
    return 0;
}
