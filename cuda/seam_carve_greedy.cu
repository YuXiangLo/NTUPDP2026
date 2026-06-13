// seam_carve_greedy.cu — NON-DP seam carving via greedy multi-start descent.
//
// Motivation: the exact cumulative DP has an irreducible serial chain of H steps
// (the hard bound of the whole DP study). This prototype drops the cumulative
// dependency entirely. Seam selection becomes W independent greedy walks:
//
//   for each start column j0 (one GPU thread):
//       col = j0; total = E[0][j0]
//       for i = 1..H-1:  col <- argmin E[i][{col-1,col,col+1}];  total += that E
//   best start = argmin_j0 total[j0];  re-walk it to recover the seam.
//
// No cost matrix, no back-pointer array, no per-row barrier, no serial chain
// across the image: the W walks are fully independent. The remaining cost is the
// embarrassingly-parallel energy map. Output is APPROXIMATE (greedy, locally
// optimal) — the speed/quality trade-off vs. exact Tiled is the point.
//
//   ./seam_carve_greedy <in.png> <num_seams> [out.png]
//
// Target: NVIDIA V100 (sm_70).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) { \
             fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(_e)); \
             exit(1); } } while (0)

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

// One thread per start column: greedy descent, accumulate raw path energy.
// Tie-break left->center->right (strict <), matching the exact pipeline.
__global__ void greedy_score_kernel(const float* __restrict__ E,
                                    float* __restrict__ total, int H, int W) {
    int j0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (j0 >= W) return;
    int col = j0;
    float t = __ldg(&E[j0]);
    for (int i = 1; i < H; ++i) {
        const float* row = E + (size_t)i * W;
        int cl = col > 0   ? col - 1 : 0;
        int cr = col < W-1 ? col + 1 : W - 1;
        float best = __ldg(&row[cl]); int bc = cl;
        float m = __ldg(&row[col]); if (m < best) { best = m; bc = col; }
        m = __ldg(&row[cr]);        if (m < best) { best = m; bc = cr; }
        col = bc;
        t += best;
    }
    total[j0] = t;
}

// One thread: pick the min-total start, then re-walk it to recover the seam.
__global__ void greedy_pick_trace_kernel(const float* __restrict__ E,
                                         const float* __restrict__ total,
                                         int* __restrict__ seam, int H, int W) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int best0 = 0; float bv = total[0];
    for (int j = 1; j < W; ++j) if (total[j] < bv) { bv = total[j]; best0 = j; }
    int col = best0; seam[0] = col;
    for (int i = 1; i < H; ++i) {
        const float* row = E + (size_t)i * W;
        int cl = col > 0   ? col - 1 : 0;
        int cr = col < W-1 ? col + 1 : W - 1;
        float best = row[cl]; int bc = cl;
        float m = row[col]; if (m < best) { best = m; bc = col; }
        m = row[cr];        if (m < best) { best = m; bc = cr; }
        col = bc; seam[i] = col;
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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.png> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    int num_seams        = atoi(argv[2]);
    const char* out_path = argc >= 4 ? argv[3] : "carved_greedy.png";

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "failed to load %s\n", in_path); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s : %dx%d  removing %d seams (GREEDY non-DP)\n",
           in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    float* h_img = (float*)malloc(npix * 3 * sizeof(float));
    for (size_t i = 0; i < npix * 3; ++i) h_img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *d_img, *d_img2, *d_gray, *d_energy, *d_total;
    int* d_seam;
    CUDA_CHECK(cudaMalloc(&d_img,    npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2,   npix * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gray,   npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_total,  (size_t)W0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seam,   (size_t)H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, npix * 3 * sizeof(float), cudaMemcpyHostToDevice));

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
        energy_kernel   <<<grid2d(w), block2d>>>(d_gray, d_energy, H, w);
        greedy_score_kernel<<<(w + 255)/256, 256>>>(d_energy, d_total, H, w);
        greedy_pick_trace_kernel<<<1, 1>>>(d_energy, d_total, d_seam, H, w);
        remove_seam_kernel<<<grid2d(w-1), block2d>>>(d_img, d_img2, d_seam, H, w);
        float* tmp = d_img; d_img = d_img2; d_img2 = tmp;
        --w;
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaGetLastError());
    printf("GREEDY carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
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
    cudaFree(d_energy); cudaFree(d_total); cudaFree(d_seam);
    return 0;
}
