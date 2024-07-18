#pragma once
#include <cuda_gl_interop.h>

class GuiApp {
public:
    bool init(const char* title, int width, int height);
    void run();
    void shutdown();

private:
    void renderUI();
    void renderImage();
    void updateCudaTexture();

    void* window = nullptr;
    int width = 1280;
    int height = 720;

    // OpenGL texture
    unsigned int glTex = 0;

    // CUDA interop
    cudaGraphicsResource* cudaTexResource = nullptr;
    int texWidth = 512;
    int texHeight = 512;

    int selectedFilterIdx = 2; // Prewitt
    const char* filterNames[11] = {
    "None",
    "Sobel",
    "Prewitt",
    "Blur",
    "Laplacian",
    "GrayscaleInvert",
    "Invert",
    "Sharpen",
    "Sepia",
    "Mean",
    "Emboss"
};
int selectedFilterIdx = 2; // default = Prewitt

};
