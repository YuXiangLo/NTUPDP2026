// Single translation unit holding the stb_image implementations.
// Compiled with the host compiler; kept separate from the CUDA TU so the
// big stb bodies aren't pulled through nvcc's device path.
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
