// seam_carve_seq.c — Sequential C seam carving reference.
//
//   ./seam_carve_seq <input> <num_seams> [output.png]
//
// Correctness oracle and single-thread speedup denominator.
// Tie-break matches numpy argmin (leftmost minimum wins), consistent with
// cuda/seam_carve.cu and seam_carving/cpu.py.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cuda/third_party/stb_image.h"
#include "../cuda/third_party/stb_image_write.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void grayscale(const float *img, float *gray, int H, int W) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float *p = img + ((size_t)y * W + x) * 3;
            gray[(size_t)y * W + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        }
}

static void compute_energy(const float *gray, float *energy, int H, int W) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int l = x > 0 ? x - 1 : 0,       r = x < W - 1 ? x + 1 : W - 1;
            int u = y > 0 ? y - 1 : 0,       d = y < H - 1 ? y + 1 : H - 1;
            float dx = fabsf(gray[(size_t)y*W+r] - gray[(size_t)y*W+l]);
            float dy = fabsf(gray[(size_t)d*W+x] - gray[(size_t)u*W+x]);
            energy[(size_t)y * W + x] = dx + dy;
        }
}

// Fills cost[] and back[], then writes the optimal seam column indices to seam[].
// Tie-break: strict `<` checking left→center→right — leftmost minimum wins.
static void dp_seam(const float *energy, float *cost, int *back, int *seam, int H, int W) {
    for (int j = 0; j < W; j++) cost[j] = energy[j];

    for (int i = 1; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int la = j > 0 ? j - 1 : 0;
            int ra = j < W - 1 ? j + 1 : W - 1;
            float c = cost[(size_t)(i-1)*W + la]; int arg = la;
            float m = cost[(size_t)(i-1)*W + j];
            if (m < c) { c = m; arg = j; }
            m = cost[(size_t)(i-1)*W + ra];
            if (m < c) { c = m; arg = ra; }
            cost[(size_t)i*W + j] = energy[(size_t)i*W + j] + c;
            back[(size_t)i*W + j] = arg;
        }
    }

    int best = 0;
    float bestv = cost[(size_t)(H-1)*W];
    for (int j = 1; j < W; j++)
        if (cost[(size_t)(H-1)*W + j] < bestv) { bestv = cost[(size_t)(H-1)*W + j]; best = j; }
    seam[H-1] = best;
    for (int i = H - 2; i >= 0; i--)
        seam[i] = back[(size_t)(i+1)*W + seam[i+1]];
}

static void remove_seam(const float *src, float *dst, const int *seam, int H, int W) {
    for (int y = 0; y < H; y++) {
        int s = seam[y];
        memcpy(dst + (size_t)y*(W-1)*3,
               src + (size_t)y*W*3,
               (size_t)s * 3 * sizeof(float));
        if (s < W - 1)
            memcpy(dst + ((size_t)y*(W-1) + s)*3,
                   src + ((size_t)y*W + s + 1)*3,
                   (size_t)(W-1-s) * 3 * sizeof(float));
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <num_seams> [output.png]\n", argv[0]);
        return 1;
    }
    int num_seams = atoi(argv[2]);
    const char *out_path = argc >= 4 ? argv[3] : "carved_seq.png";

    int W0, H, comp;
    unsigned char *pixels = stbi_load(argv[1], &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s: %dx%d, removing %d seams\n", argv[1], W0, H, num_seams);

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
    for (int s = 0; s < num_seams; s++) {
        grayscale(cur, gray, H, w);
        compute_energy(gray, energy, H, w);
        dp_seam(energy, cost, back, seam_v, H, w);
        remove_seam(cur, nxt, seam_v, H, w);
        float *tmp = cur; cur = nxt; nxt = tmp;
        w--;
    }
    double elapsed = now_ms() - t0;

    printf("SEQ carving time: %.3f ms  (%.4f ms/seam)  final size %dx%d\n",
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
