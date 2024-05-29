#include <cuda_runtime_api.h>
void preprocessROIImgK(uchar3 *input_data, int region_size, float *output_data);
void drawBoundingBox(uchar3* image, int image_width, int image_height, int box_x, int box_y, int box_width, int box_height, uchar3 color);
void getROIOfImage(uchar3* image, uchar3* roi, int image_width, int image_height, int roi_width, int roi_height);