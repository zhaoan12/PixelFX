#include "filter_common.cuh"

__global__ void meanKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int r = 0, g = 0, b = 0;
    uchar4 px;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            surf2Dread(&px, surf, (x + dx) * sizeof(uchar4), y + dy);
            r += px.x;
            g += px.y;
            b += px.z;
        }
    }

    uchar4 out = make_uchar4(r / 9, g / 9, b / 9, 255);
    surf2Dwrite(out, surf, x * sizeof(uchar4), y);
}

extern "C"
void applyMean(cudaSurfaceObject_t surface, int width, int height) {
    FILTER_KERNEL_LAUNCH(width, height);
    meanKernel<<<blocks, threads>>>(surface, width, height);
}
