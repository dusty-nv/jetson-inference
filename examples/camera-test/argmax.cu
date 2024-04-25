#include "argmax.h"

__global__ void generateClassMapKernel(float *output_1D, uint8_t *class_map)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // equivalent of i in your original function
    int j = blockIdx.x * blockDim.x + threadIdx.x; // equivalent of j in your original function

    if (i < IMG_HEIGHT && j < IMG_WIDTH)
    {
        float max_value = -10000.0f;
        int max_class = 0;
        int i_width = i * IMG_WIDTH;
        int i_width_j = i_width + j;

        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            int index = c * IMG_HW + i_width_j;
            if (output_1D[index] > max_value)
            {
                max_value = output_1D[index];
                max_class = c;
            }
        }
        class_map[i_width_j] = max_class;
    }
}
void generateClassMap(float *output_1D, uint8_t *class_map)
{

    dim3 blockDim(16, 16); // You might need to tune these numbers
    dim3 gridDim((IMG_WIDTH + blockDim.x - 1) / blockDim.x, (IMG_HEIGHT + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    generateClassMapKernel<<<gridDim, blockDim>>>(output_1D, class_map);
}