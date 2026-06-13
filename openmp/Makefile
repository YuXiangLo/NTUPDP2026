# Standalone OpenMP seam carving.
#
#   module load gcc  (needs gcc >= 9 for _Atomic / OpenMP 4.5)
#   make
#   make ARCH=-march=native   # enable AVX2/AVX-512 auto-vectorisation
#   make clean
#
# Run (SLURM):
#   srun -N 1 -n 1 -c 8 -A ACD115083 -t 5 \
#       env OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
#       ./seam_carve_seq    ../Broadway_tower_edit.jpg 10 out_seq.png
#   srun ... env OMP_NUM_THREADS=8 ... \
#       ./seam_carve_omp_v1 ../Broadway_tower_edit.jpg 10 out_v1.png
#   srun ... env OMP_NUM_THREADS=8 ... \
#       ./seam_carve_omp_v2 ../Broadway_tower_edit.jpg 10 out_v2.png
#   srun ... env OMP_NUM_THREADS=8 ... \
#       ./seam_carve_omp_v3 ../Broadway_tower_edit.jpg 10 out_v3.png
#
# Correctness check (outputs should be pixel-identical to seq):
#   diff <(xxd out_seq.png) <(xxd out_v1.png)

CC     ?= gcc
ARCH   ?=
CFLAGS := -O3 -std=c11 $(ARCH) -Wall -Wextra -I../cuda/third_party -D_POSIX_C_SOURCE=200809L

STB_OBJ := stb_impl.o
TARGETS  := seam_carve_seq seam_carve_omp_v1 seam_carve_omp_v2 seam_carve_omp_v3 seam_carve_omp_v4

all: $(TARGETS)

$(STB_OBJ): stb_impl.c ../cuda/third_party/stb_image.h ../cuda/third_party/stb_image_write.h
	$(CC) $(CFLAGS) -c stb_impl.c -o $(STB_OBJ)

seam_carve_seq: seam_carve_seq.c $(STB_OBJ)
	$(CC) $(CFLAGS) seam_carve_seq.c $(STB_OBJ) -o $@ -lm

seam_carve_omp_v1: seam_carve_omp_v1.c $(STB_OBJ)
	$(CC) $(CFLAGS) -fopenmp seam_carve_omp_v1.c $(STB_OBJ) -o $@ -lm

seam_carve_omp_v2: seam_carve_omp_v2.c $(STB_OBJ)
	$(CC) $(CFLAGS) -fopenmp seam_carve_omp_v2.c $(STB_OBJ) -o $@ -lm

seam_carve_omp_v3: seam_carve_omp_v3.c $(STB_OBJ)
	$(CC) $(CFLAGS) -fopenmp seam_carve_omp_v3.c $(STB_OBJ) -o $@ -lm

seam_carve_omp_v4: seam_carve_omp_v4.c $(STB_OBJ)
	$(CC) $(CFLAGS) -fopenmp seam_carve_omp_v4.c $(STB_OBJ) -o $@ -lm

clean:
	rm -f $(TARGETS) $(STB_OBJ)

.PHONY: all clean
