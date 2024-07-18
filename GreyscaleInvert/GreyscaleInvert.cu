#include "filter_common.cuh"

__global__ void grayscaleInvertKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 pixel;
    surf2Dread(&pixel, surf, x * sizeof(uchar4), y);

    int gray = (int)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    int inverted = 255 - gray;

    uchar4 out = make_uchar4(inverted, inverted, inverted, 255);
    surf2Dwrite(out, surf, x * sizeof(uchar4), y);
}

extern "C"
void applyGrayscaleInvert(cudaSurfaceObject_t surface, int width, int height) {
    FILTER_KERNEL_LAUNCH(width, height);
    grayscaleInvertKernel<<<blocks, threads>>>(surface, width, height);
}
