# Standalone CUDA seam carving. No torch / python needed.
#
#   module load cuda
#   make                      # builds ./seam_carve (v2) and ./seam_carve_v3
#   make ARCH=sm_80           # override arch
#   make clean
#
# Run (on the SLURM cluster):
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 \
#       ./seam_carve    ../Broadway_tower_edit.jpg 10 out2.png   # v2: single-block DP
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 \
#       ./seam_carve_v3 ../Broadway_tower_edit.jpg 10 out3.png   # v3: cooperative multi-SM DP

NVCC ?= nvcc
ARCH ?= sm_70

NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -lineinfo
# v3 needs relocatable device code for cooperative-groups grid.sync().
COOPFLAGS := $(NVCCFLAGS) -rdc=true
HOSTFLAGS := -O3 -std=c++14

TARGETS := seam_carve seam_carve_v3 seam_carve_v4 seam_carve_v5 seam_carve_v6

all: $(TARGETS)

# --- v2: single-block shared-memory DP ---
seam_carve.o: seam_carve.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve.cu -o seam_carve.o

seam_carve: seam_carve.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve.o stb_impl.o -o seam_carve

# --- v4: single-block DP with software-pipelined energy prefetch ---
seam_carve_v4.o: seam_carve_v4.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v4.cu -o seam_carve_v4.o

seam_carve_v4: seam_carve_v4.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v4.o stb_impl.o -o seam_carve_v4

# --- v5: v4 + int8 relative back-pointers ---
seam_carve_v5.o: seam_carve_v5.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v5.cu -o seam_carve_v5.o

seam_carve_v5: seam_carve_v5.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v5.o stb_impl.o -o seam_carve_v5

# --- v6: v5 generalized to arbitrary width (templated cols/thread) ---
seam_carve_v6.o: seam_carve_v6.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve_v6.cu -o seam_carve_v6.o

seam_carve_v6: seam_carve_v6.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve_v6.o stb_impl.o -o seam_carve_v6

# --- v3: cooperative multi-SM DP ---
seam_carve_v3.o: seam_carve_v3.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(COOPFLAGS) -c seam_carve_v3.cu -o seam_carve_v3.o

seam_carve_v3: seam_carve_v3.o stb_impl.o
	$(NVCC) $(COOPFLAGS) seam_carve_v3.o stb_impl.o -o seam_carve_v3

# --- shared image I/O (host-only translation unit) ---
stb_impl.o: stb_impl.cpp third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(HOSTFLAGS) -c stb_impl.cpp -o stb_impl.o

clean:
	rm -f $(TARGETS) seam_carve.o seam_carve_v3.o seam_carve_v4.o seam_carve_v5.o seam_carve_v6.o stb_impl.o

.PHONY: all clean
