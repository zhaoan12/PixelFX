#include "filter_common.cuh"

__global__ void sepiaKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 pixel;
    surf2Dread(&pixel, surf, x * sizeof(uchar4), y);

    float r = pixel.x;
    float g = pixel.y;
    float b = pixel.z;

    int tr = (int)(0.393f * r + 0.769f * g + 0.189f * b);
    int tg = (int)(0.349f * r + 0.686f * g + 0.168f * b);
    int tb = (int)(0.272f * r + 0.534f * g + 0.131f * b);

    uchar4 out = make_uchar4(
        min(255, tr),
        min(255, tg),
        min(255, tb),
        255
    );

    surf2Dwrite(out, surf, x * sizeof(uchar4), y);
}

extern "C"
void applySepia(cudaSurfaceObject_t surface, int width, int height) {
    FILTER_KERNEL_LAUNCH(width, height);
    sepiaKernel<<<blocks, threads>>>(surface, width, height);
}
