// seam_vis.cpp - generate the "before / seams / after" figure assets for the
// paper. CPU only (reuses the same energy + DP + tie-break as the CUDA path).
//
//   ./seam_vis <input.jpg> <num_seams> <seams_out.png> <carved_out.png>
//
// It removes `num_seams` vertical seams one at a time, and while doing so keeps
// an index map from each working-image column back to its ORIGINAL column, so
// the removed seams can be painted (in red) onto a copy of the original image.
// Two PNGs are written:
//   seams_out  = the original image with the removed seams drawn in red
//   carved_out = the final narrowed image
// This matches seam_carve.cu's algorithm, so the carved result is the same one
// the GPU produces (the figure is faithful to the measured pipeline).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

static void grayscale(const float* img, float* gray, int H, int W) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float* p = img + ((size_t)y * W + x) * 3;
            gray[(size_t)y * W + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        }
}

static void energy(const float* gray, float* en, int H, int W) {
    for (int y = 0; y < H; ++y) {
        int u = y > 0 ? y - 1 : 0, d = y < H - 1 ? y + 1 : H - 1;
        for (int x = 0; x < W; ++x) {
            int l = x > 0 ? x - 1 : 0, r = x < W - 1 ? x + 1 : W - 1;
            float dx = fabsf(gray[(size_t)y * W + r] - gray[(size_t)y * W + l]);
            float dy = fabsf(gray[(size_t)d * W + x] - gray[(size_t)u * W + x]);
            en[(size_t)y * W + x] = dx + dy;
        }
    }
}

// Find the minimum vertical seam over the current w-wide image; seam[y] = column.
static void find_seam(const float* en, int* back, int* seam, int H, int W,
                      std::vector<float>& a, std::vector<float>& b) {
    float* prev = a.data(); float* curr = b.data();
    for (int j = 0; j < W; ++j) prev[j] = en[j];
    for (int i = 1; i < H; ++i) {
        const float* erow = en + (size_t)i * W;
        int* brow = back + (size_t)i * W;
        for (int j = 0; j < W; ++j) {
            int la = j > 0 ? j - 1 : 0, ra = j < W - 1 ? j + 1 : W - 1;
            float c = prev[la]; int arg = la;
            float m = prev[j];  if (m < c) { c = m; arg = j; }
            m = prev[ra];       if (m < c) { c = m; arg = ra; }
            curr[j] = erow[j] + c; brow[j] = arg;
        }
        float* tmp = prev; prev = curr; curr = tmp;
    }
    int best = 0; float bestv = prev[0];
    for (int j = 1; j < W; ++j) if (prev[j] < bestv) { bestv = prev[j]; best = j; }
    seam[H - 1] = best;
    for (int i = H - 1; i > 0; --i) { best = back[(size_t)i * W + best]; seam[i - 1] = best; }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <input.jpg> <num_seams> <seams_out.png> <carved_out.png>\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    int num_seams = atoi(argv[2]);
    const char* seams_out = argv[3];
    const char* carved_out = argv[4];

    int W0, H, comp;
    unsigned char* pixels = stbi_load(in_path, &W0, &H, &comp, 3);
    if (!pixels) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }
    if (num_seams <= 0 || num_seams >= W0) {
        fprintf(stderr, "num_seams must be in [1, %d)\n", W0); return 1;
    }
    printf("loaded %s : %dx%d, drawing/removing %d seams\n", in_path, W0, H, num_seams);

    const size_t npix = (size_t)W0 * H;
    std::vector<float> work(npix * 3), work2(npix * 3), gray(npix), en(npix);
    std::vector<int> back(npix), seam(H);
    std::vector<float> dpa(W0), dpb(W0);
    for (size_t i = 0; i < npix * 3; ++i) work[i] = pixels[i] / 255.0f;

    // idx[y*W0 + x] = original column of working column x in row y.
    std::vector<int> idx(npix), idx2(npix);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W0; ++x) idx[(size_t)y * W0 + x] = x;

    // mark[y*W0 + origcol] = this original pixel was on a removed seam.
    std::vector<unsigned char> mark(npix, 0);

    float* cur = work.data(); float* nxt = work2.data();
    int* icur = idx.data();   int* inxt = idx2.data();

    int w = W0;
    for (int s = 0; s < num_seams; ++s) {
        grayscale(cur, gray.data(), H, w);
        energy(gray.data(), en.data(), H, w);
        find_seam(en.data(), back.data(), seam.data(), H, w, dpa, dpb);
        // record removed columns in original coordinates, then compact row.
        for (int y = 0; y < H; ++y) {
            int sc = seam[y];
            mark[(size_t)y * W0 + icur[(size_t)y * w + sc]] = 1;
            for (int x = 0, o = 0; x < w; ++x) {
                if (x == sc) continue;
                const float* sp = cur + ((size_t)y * w + x) * 3;
                float* op = nxt + ((size_t)y * (w - 1) + o) * 3;
                op[0] = sp[0]; op[1] = sp[1]; op[2] = sp[2];
                inxt[(size_t)y * (w - 1) + o] = icur[(size_t)y * w + x];
                ++o;
            }
        }
        float* tf = cur; cur = nxt; nxt = tf;
        int* ti = icur; icur = inxt; inxt = ti;
        --w;
    }

    // (1) original with removed seams painted red.
    std::vector<unsigned char> seam_img(npix * 3);
    for (size_t i = 0; i < npix; ++i) {
        if (mark[i]) { seam_img[i*3+0] = 255; seam_img[i*3+1] = 0; seam_img[i*3+2] = 0; }
        else { seam_img[i*3+0] = pixels[i*3+0]; seam_img[i*3+1] = pixels[i*3+1]; seam_img[i*3+2] = pixels[i*3+2]; }
    }
    if (!stbi_write_png(seams_out, W0, H, 3, seam_img.data(), W0 * 3))
        fprintf(stderr, "failed to write %s\n", seams_out);
    else printf("wrote %s (%dx%d)\n", seams_out, W0, H);

    // (2) final carved image.
    const size_t out_pix = (size_t)w * H;
    std::vector<unsigned char> carved(out_pix * 3);
    for (size_t i = 0; i < out_pix * 3; ++i) {
        float v = cur[i] * 255.0f; v = v < 0 ? 0 : (v > 255 ? 255 : v);
        carved[i] = (unsigned char)(v + 0.5f);
    }
    if (!stbi_write_png(carved_out, w, H, 3, carved.data(), w * 3))
        fprintf(stderr, "failed to write %s\n", carved_out);
    else printf("wrote %s (%dx%d)\n", carved_out, w, H);

    stbi_image_free(pixels);
    return 0;
}
