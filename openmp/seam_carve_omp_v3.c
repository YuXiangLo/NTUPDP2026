// seam_carve_omp_v3.c — OpenMP v3: wavefront (pipeline) DP.
//
//   OMP_NUM_THREADS=8 ./seam_carve_omp_v3 <input> <num_seams> [output.png]
//
// DP strategy: divide columns into T contiguous chunks (one per thread).
// Thread t can advance to row i as soon as thread t-1 has finished row i-1,
// i.e., threads overlap diagonally instead of waiting at a global barrier each row.
//
//   t=0: row 1, row 2, row 3, ...
//   t=1:      row 1, row 2, ...   (after t=0 done with row 0 → starts row 1)
//   ...
//
// This replaces H global barriers with a lightweight per-thread spin on an
// atomic frontier counter, reducing synchronization overhead at the cost of
// some startup latency (T extra "diagonal" steps to fill the pipeline).
//
// For seam carving's DP, the benefit grows with H/T.  At small H or large T
// the pipeline drain eats the gains — compare with v2 for your image size.
//
// Implementation note: `done[t]` is padded to one cache line (64 bytes)
// to prevent false sharing between adjacent threads.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

#include "../cuda/third_party/stb_image.h"
#include "../cuda/third_party/stb_image_write.h"

#define CACHE_LINE 64

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// Per-thread frontier: highest row that thread t has *completed*.
// Padded to avoid false sharing.
typedef struct { volatile int row; char _pad[CACHE_LINE - sizeof(int)]; } frontier_t;

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    int num_seams = atoi(argv[2]);
    const char *out_path = argc >= 4 ? argv[3] : "carved_omp_v3.png";

    int W0, H, comp;
    unsigned char *pixels = stbi_load(argv[1], &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }

    int T = omp_get_max_threads();
    printf("loaded %s: %dx%d, removing %d seams  threads=%d\n",
           argv[1], W0, H, num_seams, T);

    frontier_t *done = aligned_alloc(CACHE_LINE, (size_t)T * sizeof(frontier_t));
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
        shared(cur, nxt, gray, energy, cost, back, seam_v, done, num_seams, H, w, T)
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

        // --- wavefront DP ---
        // Assign contiguous column chunks; each thread spins on its left
        // neighbour's done[t-1] instead of waiting at a global barrier.
        #pragma omp parallel num_threads(T)  // nested parallel, launches T threads
        {
            int t  = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (w + nt - 1) / nt;
            int lo = t * chunk;
            int hi = lo + chunk < w ? lo + chunk : w;

            // Init frontier: row 0 is trivially ready for all threads.
            done[t].row = 0;

            // Seed row 0 for this thread's columns.
            for (int j = lo; j < hi; j++) cost[j] = energy[j];
            // Ensure all threads have written their row-0 chunk before anyone advances.
            #pragma omp barrier

            for (int i = 1; i < H; i++) {
                // Wait until the left neighbour has finished row i-1.
                // Thread 0 has no left neighbour.
                if (t > 0)
                    while (done[t-1].row < i - 1)
                        ; // spin — CPU stays in cache with this short busy-wait

                for (int j = lo; j < hi; j++) {
                    int la = j > 0 ? j-1 : 0, ra = j < w-1 ? j+1 : w-1;
                    float c = cost[(size_t)(i-1)*w+la]; int arg = la;
                    float m = cost[(size_t)(i-1)*w+j];
                    if (m < c) { c = m; arg = j; }
                    m = cost[(size_t)(i-1)*w+ra];
                    if (m < c) { c = m; arg = ra; }
                    cost[(size_t)i*w+j] = energy[(size_t)i*w+j] + c;
                    back[(size_t)i*w+j] = arg;
                }

                // Publish this row as complete. Memory fence prevents reordering.
                __atomic_store_n(&done[t].row, i, __ATOMIC_RELEASE);
            }
            // No explicit barrier needed here: the outer omp for loops provide it.
        } // end nested parallel

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
        }

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
    printf("OMP_v3 carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
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

    free(buf1); free(buf2); free(gray); free(energy); free(cost); free(back);
    free(seam_v); free(done); free(out_uc);
    return 0;
}
