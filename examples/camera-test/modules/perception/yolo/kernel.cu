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

__global__ void drawBoundingBoxKernel(uchar3* image, int image_width, int image_height, int box_x, int box_y, int box_width, int box_height, uchar3 color)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float alpha = 0.5;
    if (col < image_width && row < image_height)
    {
        // Check if the current pixel is on the border of the bounding box
        if ((col > box_x && col < box_x + box_width)&&
            (row > box_y && row < box_y + box_height))
        {
            int index = row * image_width + col;
            image[index].x = (1-alpha) * image[index].x + alpha * color.x;
            image[index].y = (1-alpha) * image[index].y + alpha * color.y;
            image[index].z = (1-alpha) * image[index].z + alpha * color.z;
        }
    }
}

void drawBoundingBox(uchar3* image, int image_width, int image_height, int box_x, int box_y, int box_width, int box_height, uchar3 color)
{
    // Define the dimensions of the thread block and the grid
    dim3 threadsPerBlock(16, 16);
    //dim3 numBlocks((image_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (image_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 numBlocks(540 / threadsPerBlock.x, 540 / threadsPerBlock.y);
    // Launch the kernel to draw the bounding box
    drawBoundingBoxKernel<<<numBlocks, threadsPerBlock>>>(image, image_width, image_height, box_x, box_y, box_width, box_height, color);
}

__global__ void getROIOfImageKernel(uchar3* image, uchar3* roi, int image_width, int image_height, int roi_width, int roi_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < roi_width && y < roi_height)
    {
        int in_x = __float2int_rd(x + image_width - roi_width); // Start from top-right corner
        int in_y = __float2int_rd(y);                         // Top part

        // Ensure coordinates are within the defined region
        in_x = min(max(0, in_x), image_width - 1);
        in_y = min(max(0, in_y), roi_width - 1); // Make sure it does not go beyond region_size

        uchar3 pixel = image[in_y * image_width + in_x];

        roi[y * roi_width + x] = pixel;                          
    }
}

void getROIOfImage(uchar3* image, uchar3* roi, int image_width, int image_height, int roi_width, int roi_height){
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(roi_width / threadsPerBlock.x, roi_height / threadsPerBlock.y);

    getROIOfImageKernel<<<blocks, threadsPerBlock>>>(image, roi, image_width, image_height,
                                                       roi_width, roi_height);
}