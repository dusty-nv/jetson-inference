#include <cuda_runtime_api.h>
__global__ void preprocess_img_kernel_to_float_NCHW(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset);
void preprocessImgK(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset);

__global__ void resizeAndCenterKernel(uchar3 *input, float *output, int width, int height, int region_size, float scale_x, float scale_y);
void preprocessROIImgK(uchar3 *input_data, int region_size, float *output_data);

void drawBoundingBox(uchar3** image, int roi_width, int roi_height, int roi_pos_x, int roi_pos_y, int box_x, int box_y, int box_width, int box_height, uchar3 color);