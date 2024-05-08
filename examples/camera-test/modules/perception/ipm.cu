#include "ipm.h"

#define UV_GRID_COLS 524288
#define OUT_IMAGE_WIDTH 1024
#define OUT_IMAGE_HEIGHT 512

__global__ void rgbToIpm(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height - 1)
        return;

    int uvIndex = y * width + x;

    int ui = uGrid[uvIndex];
    int vi = vGrid[uvIndex];

    int inIndex = vi * 1920 + ui;
    int outIndex = y * width + x;

    if (ui >= 0 && ui < 1920 && vi >= 0 && vi < 1080)
    {

        output[outIndex] = (input[inIndex].x / 255.0f - 0.485f) / 0.229f;
        output[outIndex + 524288] = (input[inIndex].y / 255.0f - 0.456f) / 0.224f;
        output[outIndex + 1048576] = (input[inIndex].z / 255.0f - 0.406f) / 0.225f;
    }
    else
    {
        output[outIndex] = -2.11790393f;
        output[outIndex + 524288] = -2.035714286f;
        output[outIndex + 1048576] = -1.804444444f;
    }
}
void warpImageK(uchar3 *input, float *output, int *uGrid, int *vGrid, int width, int height)
{
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((OUT_IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (OUT_IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    rgbToIpm<<<gridDim, blockDim>>>(input, output, uGrid, vGrid, 1024, 512);
}