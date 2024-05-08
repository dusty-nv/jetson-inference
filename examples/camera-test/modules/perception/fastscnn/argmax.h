#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define IMG_WIDTH 1024
#define IMG_HEIGHT 512
#define IMG_HW 524288
#define NUM_CLASSES 6

__global__ void generateClassMapKernel(float *output_1D, uint8_t *class_map);
extern "C" void generateClassMap(float *output_1D, uint8_t *class_map);
