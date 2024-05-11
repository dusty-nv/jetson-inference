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

void preprocessROIImgK(uchar3 *input_data, int region_size, float *output_data)
{
    // Calculate the scaling factors
    float scale_x = (float)region_size / 320.0f;
    float scale_y = (float)region_size / 320.0f;

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(320 / threadsPerBlock.x, 320 / threadsPerBlock.y);

    resizeAndCenterKernel<<<blocks, threadsPerBlock>>>(input_data, output_data, 1920, 1080,
                                                       region_size, scale_x, scale_y);
}

/*
 * roi_pos_x,y - position of the left down point of the roi 
 */
__global__ void drawBoundingBoxKernelROI(uchar3** image, int roi_width, int roi_height, int roi_pos_x, int roi_pos_y, int box_x, int box_y, int box_width, int box_height, uchar3 color)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < roi_width * roi_height; i += stride) {
        int row = i / roi_width;
        int col = i % roi_width;

        // Check if the current pixel is within the ROI and the bounding box
        if (col >= roi_pos_x && col < roi_pos_x + roi_width && row >= roi_pos_y && row < roi_pos_y + roi_height &&
            col >= box_x && col < box_x + box_width && row >= box_y && row < box_y + box_height) {
            // Check if the current pixel is on the border of the bounding box
            if (col == box_x || col == box_x + box_width - 1 || row == box_y || row == box_y + box_height - 1) {
                *image[i] = color;
            }
        }
    }
}

void drawBoundingBox(uchar3** image, int roi_width, int roi_height, int roi_pos_x, int roi_pos_y, int box_x, int box_y, int box_width, int box_height, uchar3 color)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(320 / threadsPerBlock.x, 320 / threadsPerBlock.y);

    drawBoundingBoxKernelROI<<<blocks, threadsPerBlock>>>(image, roi_width, roi_height, roi_pos_x, roi_pos_y, box_x,
                                                       box_y, box_width, box_height, color);
}