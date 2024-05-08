__global__ void preprocess_img_kernel_to_float_NCHW(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= newWidth || y >= newHeight)
        return;

    // Calculate the index for output tensor in NCHW format
    int channelSize = newWidth * newHeight;
    int red_channel_idx = y * newWidth + x;
    int green_channel_idx = channelSize + y * newWidth + x;
    int blue_channel_idx = 2 * channelSize + y * newWidth + x;

    // If the calculated position falls outside the resized image, set it to gray
    if (x < x_offset || x >= (newWidth - x_offset) || y < y_offset || y >= (newHeight - y_offset))
    {
        output[red_channel_idx] = 0.501960784f;
        output[green_channel_idx] = 0.501960784f;
        output[blue_channel_idx] = 0.501960784f;
        return;
    }

    // Map the positions back to the original image coordinates
    int orig_x = (x - x_offset) * width / (newWidth - 2 * x_offset);
    int orig_y = (y - y_offset) * height / (newHeight - 2 * y_offset);

    uchar3 pixel = input[orig_y * width + orig_x];

    // Perform the copy and normalization to [0,1]
    output[red_channel_idx] = pixel.x * 0.003921569f;
    output[green_channel_idx] = pixel.y * 0.003921569f;
    output[blue_channel_idx] = pixel.z * 0.003921569f;
}
void preprocessImgK(uchar3 *input, float *output, int width, int height, int newWidth, int newHeight, int x_offset, int y_offset)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((newWidth + blockSize.x - 1) / blockSize.x, (newHeight + blockSize.y - 1) / blockSize.y);

    preprocess_img_kernel_to_float_NCHW<<<gridSize, blockSize>>>(input, output, width, height, newWidth, newHeight, x_offset, y_offset);
}

__global__ void resizeAndCenterKernel(uchar3 *input, float *output, int width, int height,
                                      int region_size, float scale_x, float scale_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 320 && y < 320)
    {
        int in_x = __float2int_rd(scale_x * x) + (width - region_size); // Start from top-right corner
        int in_y = __float2int_rd(scale_y * y);                         // Top part

        // Ensure coordinates are within the defined region
        in_x = min(max(0, in_x), width - 1);
        in_y = min(max(0, in_y), region_size - 1); // Make sure it does not go beyond region_size

        uchar3 pixel = input[in_y * width + in_x];

        // Output format: NCHW
        output[y * 320 + x] = pixel.x / 255.0f;                 // Channel 0 (R)
        output[y * 320 + x + 320 * 320] = pixel.y / 255.0f;     // Channel 1 (G)
        output[y * 320 + x + 2 * 320 * 320] = pixel.z / 255.0f; // Channel 2 (B)
    }
}

void preprocessROIImgK(uchar3 *input_data, float *output_data)
{
    // Define the size of the region you want to crop (n x n)
    int region_size = 540; // Replace with the size you want

    // Calculate the scaling factors
    float scale_x = (float)region_size / 320.0f;
    float scale_y = (float)region_size / 320.0f;

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(320 / threadsPerBlock.x, 320 / threadsPerBlock.y);

    resizeAndCenterKernel<<<blocks, threadsPerBlock>>>(input_data, output_data, 1920, 1080,
                                                       region_size, scale_x, scale_y);
}
