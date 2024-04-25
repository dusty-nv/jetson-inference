#include <cuda_runtime_api.h>
__global__ void preprocess_img_kernel_to_float_NCHW(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset);
void preprocessImgK(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset);

__global__ void resizeAndCenterKernel(uchar3 *input, float *output, int width, int height, int region_size, float scale_x, float scale_y);
void preprocessROIImgK(uchar3 *input_data, float *output_data);