#pragma once
#include <cuda_runtime.h>
#include <string>

enum class FilterType {
    None,
    Sobel,
    Prewitt,
    Blur
};

void applyFilter(FilterType type, cudaArray_t dest, int width, int height);
