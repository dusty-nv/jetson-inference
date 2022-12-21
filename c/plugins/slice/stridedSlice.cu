#include "stridedSlice.h"
#include <iostream>
#include <cassert>

// gpu operation for nearest neighbor upsampling
template <typename T>
__global__ void gpuSlice( T* input, int nChannels, int iHeight, int iWidth, int oHeight, int oWidth, T* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    //printf("fill %d %d %d\n", x, y, z);

    if( x >= nChannels || y >= oHeight || z >= oWidth )
        return;

    const T px = input[x * iWidth * iHeight + y * iWidth + z];

    output[x * oWidth * oHeight + y * oWidth + z] = px;
}


// nearest neighbor upsampling
template <typename T>
cudaError_t cudaSlice( T* input, int nChannels, int inputHeight, int inputWidth,
                        T* output, cudaStream_t stream )
{
    if( !input || !output )
    {
        std::cout << "No input or no output" << std::endl;
        return cudaErrorInvalidDevicePointer;
    }

    // launch kernel
    //std::cout << "input2: " << input2[0] << std::endl;
    const int outputHeight = inputHeight - 1;
    const int outputWidth = inputWidth;
    const dim3 blockDim(1, 16, 16);
    const dim3 gridDim(iDivUp(nChannels, blockDim.x), iDivUp(outputHeight, blockDim.y), iDivUp(outputWidth, blockDim.z));

    gpuSlice<T><<<gridDim, blockDim, 0, stream>>>(input, nChannels, inputHeight, inputWidth, outputHeight, outputWidth, output);
    return CUDA(cudaGetLastError());
}

template cudaError_t cudaSlice<float>(float*, int, int, int, float*, cudaStream_t);
template cudaError_t cudaSlice<__half>(__half*, int, int, int, __half*, cudaStream_t);
