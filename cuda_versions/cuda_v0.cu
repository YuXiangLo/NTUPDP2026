#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t _status = (call);                                                       \
        if (_status != cudaSuccess) {                                                       \
            throw std::runtime_error(cudaGetErrorString(_status));                          \
        }                                                                                   \
    } while (0)

#define CUDA_CHECK_LAUNCH()                                                                 \
    do {                                                                                    \
        cudaError_t _status = cudaGetLastError();                                           \
        if (_status != cudaSuccess) {                                                       \
            throw std::runtime_error(cudaGetErrorString(_status));                          \
        }                                                                                   \
    } while (0)

namespace {

struct TensorData {
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;
    std::vector<float> values;
};

__device__ __forceinline__ float grayscale_at(const float* image, int y, int x, int width, int channels) {
    const int offset = (y * width + x) * channels;
    if (channels == 1) {
        return image[offset];
    }
    if (channels >= 3) {
        return 0.299f * image[offset] + 0.587f * image[offset + 1] + 0.114f * image[offset + 2];
    }
    float total = 0.0f;
    for (int c = 0; c < channels; ++c) {
        total += image[offset + c];
    }
    return total / static_cast<float>(channels);
}

__global__ void compute_energy_kernel(const float* image, float* energy, int height, int width, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int left_x = x > 0 ? x - 1 : 0;
    const int right_x = x + 1 < width ? x + 1 : width - 1;
    const int up_y = y > 0 ? y - 1 : 0;
    const int down_y = y + 1 < height ? y + 1 : height - 1;

    const float left = grayscale_at(image, y, left_x, width, channels);
    const float right = grayscale_at(image, y, right_x, width, channels);
    const float up = grayscale_at(image, up_y, x, width, channels);
    const float down = grayscale_at(image, down_y, x, width, channels);
    energy[y * width + x] = fabsf(right - left) + fabsf(down - up);
}

__global__ void remove_vertical_seam_kernel(
    const float* input,
    float* output,
    const int32_t* seam,
    int height,
    int width,
    int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width - 1 || y >= height) {
        return;
    }

    const int seam_x = seam[y];
    const int src_x = x < seam_x ? x : x + 1;
    const int input_offset = (y * width + src_x) * channels;
    const int output_offset = (y * (width - 1) + x) * channels;
    for (int c = 0; c < channels; ++c) {
        output[output_offset + c] = input[input_offset + c];
    }
}

std::vector<int32_t> find_vertical_seam(const std::vector<float>& energy, int height, int width) {
    std::vector<float> previous(energy.begin(), energy.begin() + width);
    std::vector<float> current(width, 0.0f);
    std::vector<int32_t> backtrack(static_cast<size_t>(height) * width, 0);

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int left = std::max(x - 1, 0);
            const int right = std::min(x + 1, width - 1);

            int best = left;
            float best_cost = previous[left];
            for (int candidate = left + 1; candidate <= right; ++candidate) {
                if (previous[candidate] < best_cost) {
                    best = candidate;
                    best_cost = previous[candidate];
                }
            }

            backtrack[static_cast<size_t>(y) * width + x] = static_cast<int32_t>(best);
            current[x] = energy[static_cast<size_t>(y) * width + x] + best_cost;
        }
        previous.swap(current);
    }

    std::vector<int32_t> seam(height, 0);
    seam[height - 1] = static_cast<int32_t>(std::min_element(previous.begin(), previous.end()) - previous.begin());
    for (int y = height - 2; y >= 0; --y) {
        seam[y] = backtrack[static_cast<size_t>(y + 1) * width + seam[y + 1]];
    }
    return seam;
}

TensorData read_tensor_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + path);
    }

    TensorData tensor;
    input.read(reinterpret_cast<char*>(&tensor.height), sizeof(tensor.height));
    input.read(reinterpret_cast<char*>(&tensor.width), sizeof(tensor.width));
    input.read(reinterpret_cast<char*>(&tensor.channels), sizeof(tensor.channels));
    if (!input) {
        throw std::runtime_error("Failed to read tensor header: " + path);
    }
    if (tensor.height <= 0 || tensor.width <= 0 || tensor.channels <= 0) {
        throw std::runtime_error("Invalid tensor dimensions in input file");
    }

    const size_t count = static_cast<size_t>(tensor.height) * tensor.width * tensor.channels;
    tensor.values.resize(count);
    input.read(reinterpret_cast<char*>(tensor.values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!input) {
        throw std::runtime_error("Failed to read tensor payload: " + path);
    }
    return tensor;
}

void write_tensor_file(const std::string& path, const TensorData& tensor) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    output.write(reinterpret_cast<const char*>(&tensor.height), sizeof(tensor.height));
    output.write(reinterpret_cast<const char*>(&tensor.width), sizeof(tensor.width));
    output.write(reinterpret_cast<const char*>(&tensor.channels), sizeof(tensor.channels));
    output.write(reinterpret_cast<const char*>(tensor.values.data()),
                 static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("Failed to write tensor payload: " + path);
    }
}

TensorData carve_seams_cuda_v0(const TensorData& input, int num_seams, bool show_progress) {
    if (num_seams <= 0) {
        return input;
    }
    if (num_seams >= input.width) {
        throw std::runtime_error("num_seams must be smaller than image width");
    }

    TensorData current;
    current.height = input.height;
    current.width = input.width;
    current.channels = input.channels;

    float* d_current = nullptr;
    const size_t current_bytes = input.values.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_current), current_bytes));
    CUDA_CHECK(cudaMemcpy(d_current, input.values.data(), current_bytes, cudaMemcpyHostToDevice));

    int height = input.height;
    int width = input.width;
    const int channels = input.channels;

    for (int seam_idx = 0; seam_idx < num_seams; ++seam_idx) {
        float* d_energy = nullptr;
        const size_t energy_count = static_cast<size_t>(height) * width;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_energy), energy_count * sizeof(float)));

        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        compute_energy_kernel<<<grid, block>>>(d_current, d_energy, height, width, channels);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> energy(energy_count);
        CUDA_CHECK(cudaMemcpy(energy.data(), d_energy, energy_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_energy));

        const std::vector<int32_t> seam = find_vertical_seam(energy, height, width);

        int32_t* d_seam = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_seam), static_cast<size_t>(height) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_seam, seam.data(), static_cast<size_t>(height) * sizeof(int32_t), cudaMemcpyHostToDevice));

        float* d_next = nullptr;
        const size_t next_count = static_cast<size_t>(height) * (width - 1) * channels;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next), next_count * sizeof(float)));

        const dim3 remove_grid(((width - 1) + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        remove_vertical_seam_kernel<<<remove_grid, block>>>(d_current, d_next, d_seam, height, width, channels);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_seam));
        d_current = d_next;
        --width;

        if (show_progress) {
            std::cerr << "\rCUDA_v0 seams " << (seam_idx + 1) << "/" << num_seams << std::flush;
        }
    }

    if (show_progress) {
        std::cerr << std::endl;
    }

    current.width = width;
    current.values.resize(static_cast<size_t>(height) * width * channels);
    CUDA_CHECK(cudaMemcpy(current.values.data(), d_current, current.values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    return current;
}

struct Options {
    std::string input_path;
    std::string output_path;
    int seams = 10;
    bool progress = false;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --input <input.bin> --output <output.bin> --seams <count> [--progress]\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (arg == "--seams" && i + 1 < argc) {
            options.seams = std::stoi(argv[++i]);
        } else if (arg == "--progress") {
            options.progress = true;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (options.input_path.empty() || options.output_path.empty()) {
        throw std::runtime_error("Both --input and --output are required");
    }
    if (options.seams < 0) {
        throw std::runtime_error("--seams must be non-negative");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const TensorData input = read_tensor_file(options.input_path);
        const TensorData output = carve_seams_cuda_v0(input, options.seams, options.progress);
        write_tensor_file(options.output_path, output);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "CUDA_v0 error: " << exc.what() << std::endl;
        return 1;
    }
}
