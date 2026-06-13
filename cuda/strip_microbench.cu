// strip_microbench.cu — does the tiled DP wall-time depend on the number of
// active strips? This decides whether an incremental (cone-restricted) DP can
// speed up single-seam carving on the V100.
//
// We run the exact seam_dp_tile_pf tile-loop over a fixed 8K energy map, but
// launch only `nstrips` of the STRIP_K=60 blocks per tile (the rest idle, as in
// an incremental cone touching ~24% of columns). If wall-time is flat in
// nstrips, the DP is latency-bound (serial T-row chain) and incremental cannot
// help; if it scales down, incremental could win.
//
//   ./strip_microbench <in.png>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#define CUDA_CHECK(c) do{cudaError_t e=(c); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define TILE_T 64
#define STRIP_K 60
#define NT_TILE 256

__global__ void grayscale_kernel(const float* __restrict__ img,float* __restrict__ g,int H,int W){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<W&&y<H){const float*p=img+((size_t)y*W+x)*3; g[(size_t)y*W+x]=0.299f*p[0]+0.587f*p[1]+0.114f*p[2];}
}
__global__ void energy_kernel(const float* __restrict__ g,float* __restrict__ e,int H,int W){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<W&&y<H){int l=x>0?x-1:0,r=x<W-1?x+1:W-1,u=y>0?y-1:0,d=y<H-1?y+1:H-1;
        e[(size_t)y*W+x]=fabsf(g[(size_t)y*W+r]-g[(size_t)y*W+l])+fabsf(g[(size_t)d*W+x]-g[(size_t)u*W+x]);}
}
__global__ void init_dp_row(const float* __restrict__ e,float* __restrict__ prev,int W){
    for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<W;c+=gridDim.x*blockDim.x) prev[c]=e[c];
}
// identical body to seam_carve_tiled_pf.cu's kernel
__global__ void seam_dp_tile_pf_kernel(const float* __restrict__ d_energy,signed char* __restrict__ d_back,
        const float* __restrict__ d_prev,float* __restrict__ d_next,int H,int W,int row_start,int tile_rows){
    const int k=blockIdx.x;
    const int col_start=(long long)k*W/STRIP_K, col_end=(long long)(k+1)*W/STRIP_K, S=col_end-col_start;
    const int halo_l=(col_start>=TILE_T)?TILE_T:col_start, halo_r=(col_end+TILE_T<=W)?TILE_T:(W-col_end);
    const int ext_s=col_start-halo_l, ext_w=S+halo_l+halo_r;
    extern __shared__ float sh[]; float* prev_sh=sh; float* curr_sh=sh+ext_w;
    for(int i=threadIdx.x;i<ext_w;i+=blockDim.x) prev_sh[i]=d_prev[ext_s+i];
    __syncthreads();
    float epf[2]={0,0};
    #pragma unroll
    for(int n=0;n<2;++n){int i=threadIdx.x+n*NT_TILE; if(i<ext_w) epf[n]=__ldg(&d_energy[(size_t)row_start*W+(ext_s+i)]);}
    for(int t=0;t<tile_rows;++t){
        const int row=row_start+t; signed char* brow=d_back+(size_t)row*W;
        float ecur[2];
        #pragma unroll
        for(int n=0;n<2;++n) ecur[n]=epf[n];
        const int nr=row+1;
        if(t+1<tile_rows&&nr<H){
            #pragma unroll
            for(int n=0;n<2;++n){int i=threadIdx.x+n*NT_TILE; if(i<ext_w) epf[n]=__ldg(&d_energy[(size_t)nr*W+(ext_s+i)]);}
        }
        #pragma unroll
        for(int n=0;n<2;++n){int i=threadIdx.x+n*NT_TILE; if(i>=ext_w)break; const int gc=ext_s+i;
            int la=(i>0&&gc>0)?i-1:i, ra=(i<ext_w-1&&gc<W-1)?i+1:i;
            float best=prev_sh[la]; int arg=la; float m=prev_sh[i]; if(m<best){best=m;arg=i;} m=prev_sh[ra]; if(m<best){best=m;arg=ra;}
            curr_sh[i]=ecur[n]+best; const int ci=i-halo_l; if(ci>=0&&ci<S) brow[gc]=(signed char)((ext_s+arg)-gc);}
        __syncthreads();
        float* tmp=prev_sh; prev_sh=curr_sh; curr_sh=tmp;
    }
    for(int i=threadIdx.x;i<S;i+=blockDim.x) d_next[col_start+i]=prev_sh[halo_l+i];
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <in.png>\n",argv[0]);return 1;}
    int W,H,c; unsigned char* px=stbi_load(argv[1],&W,&H,&c,3); if(!px){fprintf(stderr,"load fail\n");return 1;}
    size_t npix=(size_t)W*H; float* h=(float*)malloc(npix*3*sizeof(float));
    for(size_t i=0;i<npix*3;++i) h[i]=px[i]/255.f; stbi_image_free(px);
    printf("image %dx%d, STRIP_K=%d TILE_T=%d\n",W,H,STRIP_K,TILE_T);
    float *d_img,*d_gray,*d_e,*d_prev,*d_next; signed char* d_back;
    CUDA_CHECK(cudaMalloc(&d_img,npix*3*4)); CUDA_CHECK(cudaMalloc(&d_gray,npix*4));
    CUDA_CHECK(cudaMalloc(&d_e,npix*4)); CUDA_CHECK(cudaMalloc(&d_back,npix));
    CUDA_CHECK(cudaMalloc(&d_prev,(size_t)W*4)); CUDA_CHECK(cudaMalloc(&d_next,(size_t)W*4));
    CUDA_CHECK(cudaMemcpy(d_img,h,npix*3*4,cudaMemcpyHostToDevice));
    dim3 b(32,8), g((W+31)/32,(H+7)/8);
    grayscale_kernel<<<g,b>>>(d_img,d_gray,H,W); energy_kernel<<<g,b>>>(d_gray,d_e,H,W);
    size_t sh=2*(size_t)((W+STRIP_K-1)/STRIP_K+2*TILE_T)*sizeof(float);
    int reps=30;
    int strip_counts[]={60,40,30,20,10,5};
    printf("\nnstrips | DP-loop ms (median-ish, best of %d) | cols covered\n",reps);
    for(int sc=0; sc<6; ++sc){
        int ns=strip_counts[sc];
        // warmup
        { int row=1; while(row<H){int tr=(row+TILE_T<=H)?TILE_T:(H-row); seam_dp_tile_pf_kernel<<<ns,NT_TILE,sh>>>(d_e,d_back,d_prev,d_next,H,W,row,tr); float*t=d_prev;d_prev=d_next;d_next=t; row+=tr;} }
        CUDA_CHECK(cudaDeviceSynchronize());
        float best=1e30f; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for(int r=0;r<reps;++r){
            init_dp_row<<<(W+255)/256,256>>>(d_e,d_prev,W);
            cudaEventRecord(e0);
            int row=1; while(row<H){int tr=(row+TILE_T<=H)?TILE_T:(H-row); seam_dp_tile_pf_kernel<<<ns,NT_TILE,sh>>>(d_e,d_back,d_prev,d_next,H,W,row,tr); float*t=d_prev;d_prev=d_next;d_next=t; row+=tr;}
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms,e0,e1); if(ms<best)best=ms;
        }
        printf("%6d  | %8.3f ms                          | %5.1f%%\n", ns, best, 100.0*ns/STRIP_K);
    }
    return 0;
}
