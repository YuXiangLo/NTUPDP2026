// seam_carve_omp_v2.c — OpenMP v2: parallel energy map + parallel column DP + parallel remove.
//
//   OMP_NUM_THREADS=8 ./seam_carve_omp_v2 <input> <num_seams> [output.png]
//
// DP strategy: parallelize the inner column loop within a sequential row loop.
// A single `#pragma omp for` (with implicit barrier) after each row ensures that
// cost[i-1][*] is fully written before any thread reads it for row i.
//
// This matches the conceptual GPU approach (one kernel per row with a global barrier)
// but uses OpenMP's intra-team barrier instead of CUDA kernel launch.
// The row serial dependency is unavoidable for the exact algorithm; bandwidth
// saturation past ~8–16 threads is expected (profile with LIKWID or perf).
//
// Thread team stays alive for the entire seam-removal loop (outer parallel region).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../cuda/third_party/stb_image.h"
#include "../cuda/third_party/stb_image_write.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    int num_seams = atoi(argv[2]);
    const char *out_path = argc >= 4 ? argv[3] : "carved_omp_v2.png";

    int W0, H, comp;
    unsigned char *pixels = stbi_load(argv[1], &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s: %dx%d, removing %d seams  threads=%d\n",
           argv[1], W0, H, num_seams, omp_get_max_threads());

    size_t npix = (size_t)W0 * H;
    float *buf1   = malloc(npix * 3 * sizeof(float));
    float *buf2   = malloc(npix * 3 * sizeof(float));
    float *gray   = malloc(npix * sizeof(float));
    float *energy = malloc(npix * sizeof(float));
    float *cost   = malloc(npix * sizeof(float));
    int   *back   = malloc(npix * sizeof(int));
    int   *seam_v = malloc(H * sizeof(int));

    for (size_t i = 0; i < npix * 3; i++) buf1[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);

    float *cur = buf1, *nxt = buf2;
    int w = W0;

    double t0 = now_ms();

    #pragma omp parallel default(none) \
        shared(cur, nxt, gray, energy, cost, back, seam_v, num_seams, H, w)
    for (int s = 0; s < num_seams; s++) {

        // --- grayscale (parallel) ---
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < w; x++) {
                const float *p = cur + ((size_t)y * w + x) * 3;
                gray[(size_t)y * w + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
            }

        // --- energy (parallel) ---
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < w; x++) {
                int l = x > 0 ? x-1 : 0,     r = x < w-1 ? x+1 : w-1;
                int u = y > 0 ? y-1 : 0,     d = y < H-1 ? y+1 : H-1;
                float dx = fabsf(gray[(size_t)y*w+r] - gray[(size_t)y*w+l]);
                float dy = fabsf(gray[(size_t)d*w+x] - gray[(size_t)u*w+x]);
                energy[(size_t)y * w + x] = dx + dy;
            }

        // --- DP row 0 init (parallel) ---
        #pragma omp for schedule(static)
        for (int j = 0; j < w; j++) cost[j] = energy[j];

        // --- DP rows 1..H-1: parallel columns, global barrier per row ---
        // Each `omp for` has an implicit barrier at exit: all threads finish
        // writing cost[i][*] before any thread reads it for row i+1.
        for (int i = 1; i < H; i++) {
            #pragma omp for schedule(static)
            for (int j = 0; j < w; j++) {
                int la = j > 0 ? j-1 : 0, ra = j < w-1 ? j+1 : w-1;
                float c = cost[(size_t)(i-1)*w+la]; int arg = la;
                float m = cost[(size_t)(i-1)*w+j];
                if (m < c) { c = m; arg = j; }
                m = cost[(size_t)(i-1)*w+ra];
                if (m < c) { c = m; arg = ra; }
                cost[(size_t)i*w+j] = energy[(size_t)i*w+j] + c;
                back[(size_t)i*w+j] = arg;
            }
        }

        // --- argmin of last row + backtrack: single thread ---
        #pragma omp single
        {
            int best = 0;
            float bestv = cost[(size_t)(H-1)*w];
            for (int j = 1; j < w; j++)
                if (cost[(size_t)(H-1)*w+j] < bestv) { bestv = cost[(size_t)(H-1)*w+j]; best = j; }
            seam_v[H-1] = best;
            for (int i = H-2; i >= 0; i--)
                seam_v[i] = back[(size_t)(i+1)*w + seam_v[i+1]];
        } // implicit barrier

        // --- remove seam (parallel per row) ---
        #pragma omp for schedule(static)
        for (int y = 0; y < H; y++) {
            int sc = seam_v[y];
            memcpy(nxt + (size_t)y*(w-1)*3,
                   cur + (size_t)y*w*3,
                   (size_t)sc * 3 * sizeof(float));
            if (sc < w-1)
                memcpy(nxt + ((size_t)y*(w-1) + sc)*3,
                       cur + ((size_t)y*w + sc+1)*3,
                       (size_t)(w-1-sc) * 3 * sizeof(float));
        }

        // --- swap buffers, shrink width ---
        #pragma omp single
        {
            float *tmp = cur; cur = nxt; nxt = tmp;
            w--;
        }
    }

    double elapsed = now_ms() - t0;
    printf("OMP_v2 carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
           elapsed, elapsed / num_seams, w, H);

    size_t out_pix = (size_t)w * H;
    unsigned char *out_uc = malloc(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; i++) {
        float v = cur[i] * 255.0f;
        out_uc[i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
    }
    if (!stbi_write_png(out_path, w, H, 3, out_uc, w * 3))
        fprintf(stderr, "write failed\n");
    else
        printf("wrote %s\n", out_path);

    free(buf1); free(buf2); free(gray); free(energy); free(cost); free(back); free(seam_v); free(out_uc);
    return 0;
}
