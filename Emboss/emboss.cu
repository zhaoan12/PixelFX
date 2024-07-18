#include "filter_common.cuh"

// Emboss kernel: highlights edges in stylized way
__global__ void embossKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    uchar4 p1, p2;
    surf2Dread(&p1, surf, (x - 1) * sizeof(uchar4), y - 1);
    surf2Dread(&p2, surf, (x + 1) * sizeof(uchar4), y + 1);

    int r = clamp((int)p1.x - (int)p2.x + 128, 0, 255);
    int g = clamp((int)p1.y - (int)p2.y + 128, 0, 255);
    int b = clamp((int)p1.z - (int)p2.z + 128, 0, 255);

    uchar4 out = make_uchar4(r, g, b, 255);
    surf2Dwrite(out, surf, x * sizeof(uchar4), y);
}

extern "C"
void applyEmboss(cudaSurfaceObject_t surface, int width, int height) {
    FILTER_KERNEL_LAUNCH(width, height);
    embossKernel<<<blocks, threads>>>(surface, width, height);
}
