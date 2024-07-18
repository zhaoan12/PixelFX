#include "filter_common.cuh"

__global__ void prewittKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    uchar4 px;
    int gx = 0, gy = 0;

    // Horizontal kernel
    int H[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };
    // Vertical kernel
    int V[3][3] = {
        { 1,  1,  1},
        { 0,  0,  0},
        {-1, -1, -1}
    };

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            surf2Dread(&px, surf, (x + dx) * sizeof(uchar4), y + dy);
            int gray = (int)(0.299f * px.x + 0.587f * px.y + 0.114f * px.z);
            gx += gray * H[dy + 1][dx + 1];
            gy += gray * V[dy + 1][dx + 1];
        }
    }

    int mag = min(255, abs(gx) + abs(gy));
    uchar4 out = make_uchar4(mag, mag, mag, 255);
    surf2Dwrite(out, surf, x * sizeof(uchar4), y);
}

extern "C"
void applyPrewitt(cudaSurfaceObject_t surface, int width, int height) {
    FILTER_KERNEL_LAUNCH(width, height);
    prewittKernel<<<blocks, threads>>>(surface, width, height);
}
