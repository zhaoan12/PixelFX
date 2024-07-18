#include "filter_dispatch.h"
#include "filter_common.cuh"

// External filter functions
extern void applyPrewitt(cudaSurfaceObject_t, int, int);
extern void applyBlur(cudaSurfaceObject_t, int, int);
extern void applyLaplacian(cudaSurfaceObject_t, int, int);
extern void applyGrayscaleInvert(cudaSurfaceObject_t, int, int);
extern void applySharpen(cudaSurfaceObject_t, int, int);
extern void applySepia(cudaSurfaceObject_t, int, int);
extern void applyMean(cudaSurfaceObject_t, int, int);
extern void applyEmboss(cudaSurfaceObject_t, int, int);
//...

void applyFilter(FilterType type, cudaArray_t destArray, int width, int height) {
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = destArray;
    
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &desc);

    switch (type) {
        enum class FilterType {
            None,
            Sobel,
            Prewitt,
            Blur,
            Laplacian,
            GrayscaleInvert,
            Invert,
            Sharpen,
            Sepia,
            Mean,
            Emboss
        };

    cudaDestroySurfaceObject(surface);
}
