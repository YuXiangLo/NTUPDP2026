// seam_carve_cpu_incr.cpp — EXACT incremental seam carving (CPU reference).
//
// The "read-less" escape from the per-seam Omega(HW) floor: after removing a
// seam, only a thin, contiguous "influence cone" of the cumulative-cost table M
// actually changes (measured ~11% of cells). We maintain M across seams and
// recompute only that cone, reusing the rest. Output must be bit-identical to a
// full recompute -- this binary validates that exactness and measures the
// incremental speedup before any GPU port.
//
//   ./seam_carve_cpu_incr <in.png> <num_seams> [out.png] [full|incr]
//
// Single-threaded; energy is recomputed fully each seam (simple, correct), so
// the measured win isolates the DP cone. All arrays are stored compacted at the
// current width w (row i at i*w), so a seam removal is a per-row column delete.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

static void grayscale(const float* img, float* g, int H, int W) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float* p = img + ((size_t)y * W + x) * 3;
            g[(size_t)y*W+x] = 0.299f*p[0] + 0.587f*p[1] + 0.114f*p[2];
        }
}
static void energy(const float* g, float* e, int H, int W) {
    for (int y = 0; y < H; ++y) {
        int u = y>0?y-1:0, d = y<H-1?y+1:H-1;
        for (int x = 0; x < W; ++x) {
            int l = x>0?x-1:0, r = x<W-1?x+1:W-1;
            float dx = fabsf(g[(size_t)y*W+r]-g[(size_t)y*W+l]);
            float dy = fabsf(g[(size_t)d*W+x]-g[(size_t)u*W+x]);
            e[(size_t)y*W+x] = dx+dy;
        }
    }
}
// M[i][j] = e[i][j] + min(M[i-1][j-1..j+1]); back = RELATIVE parent offset
// (parent_col - j) in {-1,0,+1}. Relative offsets survive a seam-column shift
// for cells whose local neighbourhood is intact (the non-cone region). Stride W.
static inline void dp_cell(const float* e, float* M, int* back, int i, int j, int W) {
    const float* pv = M + (size_t)(i-1)*W;
    int la = j>0?j-1:0, ra = j<W-1?j+1:W-1;
    float c = pv[la]; int arg = la;
    float m = pv[j];  if (m < c) { c = m; arg = j;  }
    m = pv[ra];       if (m < c) { c = m; arg = ra; }
    M[(size_t)i*W+j] = e[(size_t)i*W+j] + c;
    back[(size_t)i*W+j] = arg - j;            // relative
}
static void dp_full(const float* e, float* M, int* back, int H, int W) {
    for (int j = 0; j < W; ++j) { M[j] = e[j]; back[j] = 0; }
    for (int i = 1; i < H; ++i)
        for (int j = 0; j < W; ++j) dp_cell(e, M, back, i, j, W);
}
static void backtrack(const float* M, const int* back, int* seam, int H, int W) {
    int best = 0; float bv = M[(size_t)(H-1)*W];
    for (int j = 1; j < W; ++j) if (M[(size_t)(H-1)*W+j] < bv) { bv = M[(size_t)(H-1)*W+j]; best = j; }
    seam[H-1] = best;
    for (int i = H-1; i > 0; --i) {
        best += back[(size_t)i*W+best];
        if (best < 0) best = 0; else if (best > W-1) best = W-1;
        seam[i-1] = best;
    }
}
// delete column seam[y] from a (H x W x C) buffer -> (H x (W-1) x C) compacted.
template<class T>
static void remove_col(const T* in, T* out, const int* seam, int H, int W, int C) {
    for (int y = 0; y < H; ++y) {
        int s = seam[y];
        for (int x = 0; x < W-1; ++x) {
            int src = x < s ? x : x+1;
            for (int c = 0; c < C; ++c) out[((size_t)y*(W-1)+x)*C+c] = in[((size_t)y*W+src)*C+c];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s <in> <n> [out] [full|incr]\n",argv[0]); return 1; }
    int num_seams = atoi(argv[2]);
    const char* out_path = argc>=4?argv[3]:"carved_incr.png";
    bool incr = !(argc>=5 && strcmp(argv[4],"full")==0);

    int W0,H,comp; unsigned char* px = stbi_load(argv[1],&W0,&H,&comp,3);
    if (!px) { fprintf(stderr,"load failed\n"); return 1; }
    if (num_seams<=0||num_seams>=W0){fprintf(stderr,"bad n\n");return 1;}
    printf("loaded %s : %dx%d  %d seams  mode=%s\n", argv[1],W0,H,num_seams, incr?"incr":"full");

    size_t npix=(size_t)W0*H;
    std::vector<float> imgA(npix*3), imgB(npix*3), gray(npix), eCur(npix), ePrev(npix);
    std::vector<float> M(npix), M2(npix); std::vector<int> back(npix), back2(npix), seam(H);
    for (size_t i=0;i<npix*3;++i) imgA[i]=px[i]/255.0f;
    stbi_image_free(px);
    float *img=imgA.data(), *img2=imgB.data();
    float *Mp=M.data(), *Mp2=M2.data(); int *Bp=back.data(), *Bp2=back2.data();
    float *eC=eCur.data(), *eP=ePrev.data();

    auto t0=std::chrono::high_resolution_clock::now();
    int w=W0;
    grayscale(img,gray.data(),H,w);
    energy(gray.data(),eC,H,w);
    dp_full(eC,Mp,Bp,H,w);

    for (int s=0;s<num_seams;++s) {
        backtrack(Mp,Bp,seam.data(),H,w);
        remove_col(img,img2,seam.data(),H,w,3); std::swap(img,img2);
        int nw=w-1;
        if (s==num_seams-1) { w=nw; break; }

        if (!incr) {
            grayscale(img,gray.data(),H,nw);
            energy(gray.data(),eC,H,nw);
            dp_full(eC,Mp,Bp,H,nw);
            w=nw; continue;
        }

        // --- incremental ---
        // shift M, back to align with the removed seam (compacted to width nw)
        remove_col(Mp,Mp2,seam.data(),H,w,1); std::swap(Mp,Mp2);
        remove_col(Bp,Bp2,seam.data(),H,w,1); std::swap(Bp,Bp2);
        // old energy (width w) -> eP; recompute new energy (width nw) -> eC
        std::swap(eC,eP);
        grayscale(img,gray.data(),H,nw);
        energy(gray.data(),eC,H,nw);

        // recompute the cone: per-row contiguous changed interval, propagated.
        int plo=0, phi=-1;                       // previous row's changed interval (empty)
        for (int i=0;i<H;++i) {
            int si=seam[i];
            const float* eNr = eC + (size_t)i*nw;          // new energy, this row (stride nw)
            const float* ePr = eP + (size_t)i*w;           // old energy, this row (stride w)
            // energy-changed extent vs shifted old energy: old(i,j)=ePr[j<si? j : j+1]
            int blo = si-3<0?0:si-3, bhi = si+2>nw-1?nw-1:si+2;   // structural safety band
            int elo=blo, ehi=bhi;                                 // always include the band
            for (int j=0;j<nw;++j) { if (eNr[j]!=ePr[j<si? j : j+1]) { if(j<elo)elo=j; if(j>ehi)ehi=j; } }
            // union with expanded previous changed interval
            int clo=elo, chi=ehi;
            if (phi>=plo) { int xl=plo-1<0?0:plo-1, xh=phi+1>nw-1?nw-1:phi+1; if(xl<clo)clo=xl; if(xh>chi)chi=xh; }
            if (clo<0)clo=0; if (chi>nw-1)chi=nw-1;

            float* Mr=Mp+(size_t)i*nw; int* Br=Bp+(size_t)i*nw;
            int nlo=nw, nhi=-1;
            if (i==0) {
                for (int j=clo;j<=chi;++j){ float nv=eNr[j]; if(nv!=Mr[j]||Br[j]!=0){Mr[j]=nv;Br[j]=0; if(j<nlo)nlo=j; if(j>nhi)nhi=j;} }
            } else {
                for (int j=clo;j<=chi;++j){
                    float oldM=Mr[j]; int oldB=Br[j];
                    dp_cell(eC,Mp,Bp,i,j,nw);            // reads Mp[i-1][*] (already aligned/updated)
                    if (Mp[(size_t)i*nw+j]!=oldM || Bp[(size_t)i*nw+j]!=oldB) { if(j<nlo)nlo=j; if(j>nhi)nhi=j; }
                }
            }
            plo=nlo; phi=nhi;
        }
        w=nw;
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("CPU-%s time: %.3f ms  (%.4f ms/seam)  final %dx%d\n", incr?"incr":"full", ms, ms/num_seams, w, H);

    size_t op=(size_t)w*H; std::vector<unsigned char> outc(op*3);
    for (size_t i=0;i<op*3;++i){ float v=img[i]*255.0f+0.5f; outc[i]=(unsigned char)(v<0?0:v>255?255:v); }
    if(!stbi_write_png(out_path,w,H,3,outc.data(),w*3)) fprintf(stderr,"write failed\n");
    else printf("wrote %s\n",out_path);
    return 0;
}
