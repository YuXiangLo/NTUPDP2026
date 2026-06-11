// seam_carve_cpu.cpp - Standalone CPU vertical seam carving (the host-side
// reference point for the GPU versions). Same toolchain, same stb_image I/O,
// same energy + DP + tie-break as seam_carve.cu, so the output is bit-comparable
// and the ms/seam number is directly comparable to the CUDA binaries.
//
//   ./seam_carve_cpu     <input.jpg> <num_seams> [output.png]   # single thread
//   ./seam_carve_cpu_omp <input.jpg> <num_seams> [output.png]   # OpenMP (OMP_NUM_THREADS)
//
// The whole pipeline (grayscale, energy = gradient magnitude, cumulative-cost
// DP, backtrack, seam removal) is plain C++. With -fopenmp the embarrassingly
// parallel passes and each DP row's cells are split across cores; the row->row
// dependency stays serial (irreducible for an exact seam), exactly like the GPU.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// stb implementation lives in stb_impl.o; here we only need the declarations.
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

// gray = 0.299 R + 0.587 G + 0.114 B  (img row-major, channel-last, [0,1])
static void grayscale(const float* img, float* gray, int H, int W) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float* p = img + ((size_t)y * W + x) * 3;
            gray[(size_t)y * W + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        }
    }
}

// energy = |gray[x+1]-gray[x-1]| + |gray[y+1]-gray[y-1]|, edges clamped.
static void energy(const float* gray, float* en, int H, int W) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; ++y) {
        int u = y > 0 ? y - 1 : 0;
        int d = y < H - 1 ? y + 1 : H - 1;
        for (int x = 0; x < W; ++x) {
            int l = x > 0 ? x - 1 : 0;
            int r = x < W - 1 ? x + 1 : W - 1;
            float dx = fabsf(gray[(size_t)y * W + r] - gray[(size_t)y * W + l]);
            float dy = fabsf(gray[(size_t)d * W + x] - gray[(size_t)u * W + x]);
            en[(size_t)y * W + x] = dx + dy;
        }
    }
}

// Cumulative-cost DP over the energy map, then argmin + backtrack.
// prev/curr cost rows ping-pong; back[i*W+j] = chosen parent column (absolute),
// tie-break left->center->right with strict '<' to match the CUDA kernels.
static void seam_dp(const float* en, int* back, int* seam, int H, int W,
                    std::vector<float>& a, std::vector<float>& b) {
    float* prev = a.data();
    float* curr = b.data();
    for (int j = 0; j < W; ++j) prev[j] = en[j];

    for (int i = 1; i < H; ++i) {
        const float* erow = en + (size_t)i * W;
        int* brow = back + (size_t)i * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < W; ++j) {
            int la = j > 0 ? j - 1 : 0;
            int ra = j < W - 1 ? j + 1 : W - 1;
            float c = prev[la];
            int arg = la;
            float m = prev[j];
            if (m < c) { c = m; arg = j; }
            m = prev[ra];
            if (m < c) { c = m; arg = ra; }
            curr[j] = erow[j] + c;
            brow[j] = arg;
        }
        float* tmp = prev; prev = curr; curr = tmp;
    }

    // prev now holds the last row's cumulative costs.
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

// Copy each row skipping the seam column -> width W-1, in place into `out`.
static void remove_seam(const float* img, float* out, const int* seam, int H, int W) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; ++y) {
        int s = seam[y];
        for (int x = 0; x < W - 1; ++x) {
            int src = x < s ? x : x + 1;
            const float* sp = img + ((size_t)y * W + src) * 3;
            float* op = out + ((size_t)y * (W - 1) + x) * 3;
            op[0] = sp[0]; op[1] = sp[1]; op[2] = sp[2];
        }
    }
}

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

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif
    printf("loaded %s : %dx%d, removing %d seams (CPU, %d thread%s)\n",
           in_path, W0, H, num_seams, nthreads, nthreads == 1 ? "" : "s");

    const size_t npix = (size_t)W0 * H;
    std::vector<float> img(npix * 3), img2(npix * 3), gray(npix), en(npix);
    std::vector<int> back(npix), seam(H);
    std::vector<float> dpa(W0), dpb(W0);  // DP ping-pong rows
    for (size_t i = 0; i < npix * 3; ++i) img[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float* cur = img.data();
    float* nxt = img2.data();

    auto t0 = std::chrono::high_resolution_clock::now();

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale(cur, gray.data(), H, w);
        energy(gray.data(), en.data(), H, w);
        seam_dp(en.data(), back.data(), seam.data(), H, w, dpa, dpb);
        remove_seam(cur, nxt, seam.data(), H, w);
        float* tmp = cur; cur = nxt; nxt = tmp;
        --w;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
           ms, ms / num_seams, w, H);

    const size_t out_pix = (size_t)w * H;
    std::vector<unsigned char> h_out(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = cur[i] * 255.0f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        h_out[i] = (unsigned char)(v + 0.5f);
    }
    if (!stbi_write_png(out_path, w, H, 3, h_out.data(), w * 3))
        fprintf(stderr, "failed to write %s\n", out_path);
    else
        printf("wrote %s\n", out_path);

    return 0;
}
