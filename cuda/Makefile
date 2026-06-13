# Standalone CUDA seam carving. No torch / python needed.
#
#   module load cuda
#   make                      # builds all CUDA variants
#   make ARCH=sm_80           # override arch (A100)
#   make clean
#
# Run (on the SLURM cluster):
#   srun -N1 -n1 --gpus-per-node 1 -A ACD115083 -t 5 \
#       ./seam_carve_v5    data/ctrl/broadway_tower_ctrl_960x540.png 96 out.png
#   srun -N1 -n1 --gpus-per-node 1 -A ACD115083 -t 5 \
#       ./seam_carve_wide  data/8k/forest_pano_8k_7680x4320.png 768 out.png
#
# Variants:
#   seam_carve        (v2) — baseline single-block shared-mem DP
#   seam_carve_v0          — naive baseline, one kernel launch per DP row
#   seam_carve_v3          — cooperative multi-SM (grid.sync), experimental
#   seam_carve_v4          — v2 + software-pipelined energy prefetch
#   seam_carve_v5          — v4 + int8 relative back-pointers  [width ≤ 2048]
#   seam_carve_v6          — v5 generalized to arbitrary width (templated cols/thread)
#   seam_carve_naive       — per-seam H2D/D2H transfer baseline [width ≤ 2048]
#   seam_carve_wide        — v5-style DP, grid-stride, no width limit
#   seam_carve_fused       — fused gray+energy+DP kernel        [width ≤ 2048]
#   seam_carve_cpu         — single-thread C++ reference baseline
#   seam_carve_cpu_omp     — CPU reference built with OpenMP
#   seam_vis               — CPU tool rendering before/seams/after figures

NVCC ?= nvcc
CXX  ?= g++
ARCH ?= sm_70

NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -lineinfo
COOPFLAGS := $(NVCCFLAGS) -rdc=true
HOSTFLAGS := -O3 -std=c++14
# CPU baseline: aggressive host optimization, native vectorization.
CPUFLAGS  := -O3 -std=c++14 -march=native -funroll-loops

TARGETS := seam_carve seam_carve_v0 seam_carve_v3 seam_carve_v4 seam_carve_v5 seam_carve_v6 \
           seam_carve_naive seam_carve_wide seam_carve_fused seam_carve_tiled \
           seam_carve_persistent seam_carve_batch seam_carve_tiled_hv \
           seam_carve_cpu seam_carve_cpu_omp seam_vis

all: $(TARGETS)

# --- v2: baseline single-block shared-memory DP ---
seam_carve.o: seam_carve.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve.cu -o seam_carve.o

seam_carve: seam_carve.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve.o stb_impl.o -o seam_carve

# --- v0: naive baseline, one kernel launch per DP row ---
seam_carve_v0.o: seam_carve_v0.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v0.cu -o seam_carve_v0.o

seam_carve_v0: seam_carve_v0.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v0.o stb_impl.o -o seam_carve_v0

# --- v3: cooperative multi-SM DP (experimental) ---
seam_carve_v3.o: seam_carve_v3.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(COOPFLAGS) -c seam_carve_v3.cu -o seam_carve_v3.o

seam_carve_v3: seam_carve_v3.o stb_impl.o
	$(NVCC) $(COOPFLAGS) seam_carve_v3.o stb_impl.o -o seam_carve_v3

# --- v4: v2 + software-pipelined energy prefetch ---
seam_carve_v4.o: seam_carve_v4.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v4.cu -o seam_carve_v4.o

seam_carve_v4: seam_carve_v4.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v4.o stb_impl.o -o seam_carve_v4

# --- v5: v4 + int8 back-pointers [width ≤ 2048] ---
seam_carve_v5.o: seam_carve_v5.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v5.cu -o seam_carve_v5.o

seam_carve_v5: seam_carve_v5.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v5.o stb_impl.o -o seam_carve_v5

# --- v6: v5 generalized to arbitrary width (templated cols/thread) ---
seam_carve_v6.o: seam_carve_v6.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v6.cu -o seam_carve_v6.o

seam_carve_v6: seam_carve_v6.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v6.o stb_impl.o -o seam_carve_v6

# --- naive: per-seam H2D/D2H transfer baseline [width ≤ 2048] ---
seam_carve_naive.o: seam_carve_naive.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_naive.cu -o seam_carve_naive.o

seam_carve_naive: seam_carve_naive.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_naive.o stb_impl.o -o seam_carve_naive

# --- wide: grid-stride DP, no width limit (4K/8K support) ---
seam_carve_wide.o: seam_carve_wide.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_wide.cu -o seam_carve_wide.o

seam_carve_wide: seam_carve_wide.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_wide.o stb_impl.o -o seam_carve_wide

# --- tiled: multi-SM tiled DP, arbitrary width ---
seam_carve_tiled.o: seam_carve_tiled.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_tiled.cu -o seam_carve_tiled.o

seam_carve_tiled: seam_carve_tiled.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_tiled.o stb_impl.o -o seam_carve_tiled

# --- batch: approximate multi-seam removal (K seams per DP pass) ---
seam_carve_batch: seam_carve_batch.cu stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_batch.cu stb_impl.o -o seam_carve_batch

# --- tiled-hv: tiled DP + VRAM transpose for horizontal seams ---
seam_carve_tiled_hv.o: seam_carve_tiled_hv.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_tiled_hv.cu -o seam_carve_tiled_hv.o

seam_carve_tiled_hv: seam_carve_tiled_hv.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_tiled_hv.o stb_impl.o -o seam_carve_tiled_hv

# --- greedy: NON-DP seam carving (greedy multi-start, no cost matrix) ---
seam_carve_greedy: seam_carve_greedy.cu stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_greedy.cu stb_impl.o -o seam_carve_greedy

# --- strip_microbench: tiled DP wall-time vs active-strip count (latency probe) ---
strip_microbench: strip_microbench.cu stb_impl.o
	$(NVCC) $(NVCCFLAGS) strip_microbench.cu stb_impl.o -o strip_microbench

# --- cpu_incr: exact incremental seam carving CPU reference (host C++) ---
seam_carve_cpu_incr: seam_carve_cpu_incr.cpp stb_impl.o
	$(CXX) $(CPUFLAGS) seam_carve_cpu_incr.cpp stb_impl.o -o seam_carve_cpu_incr

# --- persistent: tiled DP fused into one cooperative launch per seam ---
seam_carve_persistent.o: seam_carve_persistent.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_persistent.cu -o seam_carve_persistent.o

seam_carve_persistent: seam_carve_persistent.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_persistent.o stb_impl.o -o seam_carve_persistent

# --- fused: single-kernel gray+energy+DP [width ≤ 2048] ---
seam_carve_fused.o: seam_carve_fused.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_fused.cu -o seam_carve_fused.o

seam_carve_fused: seam_carve_fused.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_fused.o stb_impl.o -o seam_carve_fused

# --- CPU baseline: single-thread C++ reference (the host-side comparison point) ---
seam_carve_cpu: seam_carve_cpu.cpp stb_impl.o third_party/stb_image.h third_party/stb_image_write.h
	$(CXX) $(CPUFLAGS) seam_carve_cpu.cpp stb_impl.o -o seam_carve_cpu

# --- CPU baseline: same source built with OpenMP (set OMP_NUM_THREADS at run time) ---
seam_carve_cpu_omp: seam_carve_cpu.cpp stb_impl.o third_party/stb_image.h third_party/stb_image_write.h
	$(CXX) $(CPUFLAGS) -fopenmp seam_carve_cpu.cpp stb_impl.o -o seam_carve_cpu_omp

# --- seam_vis: CPU tool that renders the paper's before/seams/after figures ---
seam_vis: seam_vis.cpp stb_impl.o third_party/stb_image.h third_party/stb_image_write.h
	$(CXX) $(HOSTFLAGS) seam_vis.cpp stb_impl.o -o seam_vis

# --- shared image I/O (host-only translation unit) ---
stb_impl.o: stb_impl.cpp third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(HOSTFLAGS) -c stb_impl.cpp -o stb_impl.o

clean:
	rm -f $(TARGETS) \
	      seam_carve.o seam_carve_v0.o seam_carve_v3.o seam_carve_v4.o seam_carve_v5.o seam_carve_v6.o \
	      seam_carve_naive.o seam_carve_wide.o seam_carve_fused.o seam_carve_tiled.o \
	      seam_carve_tiled_hv.o seam_carve_persistent.o stb_impl.o

.PHONY: all clean
