#pragma once
#include <cuda_runtime.h>
#include <cuda_surface_types.h>

#define FILTER_KERNEL_LAUNCH(W, H) \
    dim3 threads(16, 16); \
    dim3 blocks((W + 15) / 16, (H + 15) / 16);

