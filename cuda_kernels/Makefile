# Build the seam_cuda extension (.so) for the V100 (sm_70).
#
#   make            # build seam_cuda.so in this directory
#   make clean
#   make PYTHON=python3.10   # override the interpreter if needed
#
# The resulting seam_cuda.so is imported by cuda_versions/cuda_v2.py.

PYTHON ?= python
NVCC   ?= nvcc
ARCH   ?= sm_70

TORCH_INC := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
TORCH_LIB := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))")
PY_INC    := $(shell $(PYTHON) -c "import sysconfig; print('-I'+sysconfig.get_path('include'))")
ABI       := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

CXXFLAGS := -O3 -std=c++17 -arch=$(ARCH) --compiler-options '-fPIC' \
            -D_GLIBCXX_USE_CXX11_ABI=$(ABI) \
            $(TORCH_INC) $(PY_INC)

LDFLAGS  := $(TORCH_LIB) -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -ltorch_python

TARGET := seam_cuda.so
SRC    := seam_dp.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -shared $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
