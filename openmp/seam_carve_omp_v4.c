// seam_carve_omp_v4.c — OpenMP v4: v2 + NUMA-aware parallel first-touch.
//
//   OMP_NUM_THREADS=40 OMP_PROC_BIND=close OMP_PLACES=cores
//       ./seam_carve_omp_v4 <input> <num_seams> [output.png]
//
// Identical algorithm to v2 (persistent parallel region; parallel grayscale,
// energy, column-DP, and remove; one implicit barrier per DP row). The only
// change is buffer initialization: v2 normalizes the image serially on the
// master thread, so every page of `buf1` is first-touched on the master's NUMA
// node. On a 2-socket node, threads on the remote socket then pay remote-memory
// latency for the whole pipeline, capping scaling near one socket's core count.
// v4 first-touches every working buffer in parallel with the SAME static row
// partition the compute loops use, so pages land on the socket that reads them.
// This is the standard NUMA first-touch fix and gives the fairest (strongest)
// multi-core CPU baseline.
//
// Output is bit-identical to v2 / the sequential reference (first-touch changes
// page placement, not values). The row serial dependency is still irreducible.

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
    const char *out_path = argc >= 4 ? argv[3] : "carved_omp_v4.png";

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
    float *cost   = malloc(2 * (size_t)W0 * sizeof(float)); // ping-pong: 2 rows
    int   *back   = malloc(npix * sizeof(int));
    int   *seam_v = malloc(H * sizeof(int));

    // NUMA-aware first-touch: each thread touches the rows it will later compute
    // (static schedule over rows), so pages land on the local socket. buf1 is
    // also where the normalized image is produced, so this both initializes and
    // places it. pixels[] stays on the master node but is read only once here.
    #pragma omp parallel default(none) \
        shared(buf1, buf2, gray, energy, cost, back, pixels, H, W0, npix)
    {
        #pragma omp for schedule(static) nowait
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W0; x++) {
                size_t p = (size_t)y * W0 + x;
                buf1[p*3+0] = pixels[p*3+0] / 255.0f;
                buf1[p*3+1] = pixels[p*3+1] / 255.0f;
                buf1[p*3+2] = pixels[p*3+2] / 255.0f;
                buf2[p*3+0] = buf2[p*3+1] = buf2[p*3+2] = 0.0f;
                gray[p] = 0.0f; energy[p] = 0.0f; back[p] = 0;
            }
        }
        // cost is only two rows (ping-pong); touch it separately.
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < 2 * W0; j++) cost[j] = 0.0f;
    }
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

        // --- DP row 0 init -> parity-0 cost row (parallel) ---
        #pragma omp for schedule(static)
        for (int j = 0; j < w; j++) cost[j] = energy[j];

        // --- DP rows 1..H-1: parallel columns, one barrier per row ---
        // cost is a 2-row ping-pong indexed by row parity (i&1): row i reads the
        // previous parity row and writes its own. This keeps the working cost set
        // (2*w floats) in cache and roughly halves DP memory traffic vs a full
        // H*W cost matrix. The full back[] array is still needed for backtrack.
        // The implicit barrier after each `omp for` orders row i-1 before row i.
        for (int i = 1; i < H; i++) {
            const size_t pbase = (size_t)((i - 1) & 1) * w;  // previous row
            const size_t cbase = (size_t)(i & 1) * w;        // current row
            #pragma omp for schedule(static)
            for (int j = 0; j < w; j++) {
                int la = j > 0 ? j-1 : 0, ra = j < w-1 ? j+1 : w-1;
                float c = cost[pbase+la]; int arg = la;
                float m = cost[pbase+j];
                if (m < c) { c = m; arg = j; }
                m = cost[pbase+ra];
                if (m < c) { c = m; arg = ra; }
                cost[cbase+j] = energy[(size_t)i*w+j] + c;
                back[(size_t)i*w+j] = arg;
            }
        }

        // --- argmin of last row + backtrack: single thread ---
        #pragma omp single
        {
            const size_t lbase = (size_t)((H - 1) & 1) * w;  // last row's parity
            int best = 0;
            float bestv = cost[lbase];
            for (int j = 1; j < w; j++)
                if (cost[lbase+j] < bestv) { bestv = cost[lbase+j]; best = j; }
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
    printf("OMP_v4 carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
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
