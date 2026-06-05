# Standalone CUDA seam carving. No torch / python needed.
#
#   module load cuda
#   make                      # builds ./seam_carve  (default ARCH=sm_70, V100)
#   make ARCH=sm_80           # override arch
#   make clean
#
# Run (on the SLURM cluster):
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 1 \
#       ./seam_carve ../Broadway_tower_edit.jpg 10 carved.png

NVCC ?= nvcc
ARCH ?= sm_70

NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -lineinfo
HOSTFLAGS := -O3 -std=c++14

TARGET := seam_carve

all: $(TARGET)

seam_carve.o: seam_carve.cu third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(NVCCFLAGS) -c seam_carve.cu -o seam_carve.o

stb_impl.o: stb_impl.cpp third_party/stb_image.h third_party/stb_image_write.h
	$(NVCC) $(HOSTFLAGS) -c stb_impl.cpp -o stb_impl.o

$(TARGET): seam_carve.o stb_impl.o
	$(NVCC) $(NVCCFLAGS) seam_carve.o stb_impl.o -o $(TARGET)

clean:
	rm -f $(TARGET) seam_carve.o stb_impl.o

.PHONY: all clean
