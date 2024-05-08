#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

__global__ void rgbToIpm(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);
extern "C" void warpImageK(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height);