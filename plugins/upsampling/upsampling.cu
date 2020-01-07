#include "upsampling.h"
#include <iostream>
#include <cassert>

// gpu operation for nearest neighbor upsampling
template <typename T>
__global__ void gpuResizeNearestNeighbor( T* input, int nChannels, int iHeight, int iWidth, int oHeight, int oWidth, T* output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if( x >= nChannels || y >= oHeight || z >= oWidth )
        return;

    const int dy = y / 2;
    const int dz = z / 2;

    const T px = input[x * iWidth * iHeight + dy * iWidth + dz];

    output[x * oWidth * oHeight + y * oWidth + z] = px;
}


// nearest neighbor upsampling
template <typename T>
cudaError_t cudaResizeNearestNeighbor( T* input, int nChannels, int inputHeight, int inputWidth,
                        T* output, cudaStream_t stream )
{
    if( !input || !output )
    {
        std::cout << "No input or no output" << std::endl;
        return cudaErrorInvalidDevicePointer;
    }

    if( inputWidth == 0 || inputHeight == 0 )
    {
        std::cout << "Width or height is 0" << std::endl;
        return cudaErrorInvalidValue;
    }

    // launch kernel
    const dim3 blockDim(1, 16, 16);
    const size_t outputWidth = 2 * inputWidth;
    const size_t outputHeight = 2 * inputHeight;
    const dim3 gridDim(iDivUp(nChannels, blockDim.x), iDivUp(outputHeight, blockDim.y), iDivUp(outputWidth, blockDim.z));

    gpuResizeNearestNeighbor<T><<<gridDim, blockDim, 0, stream>>>(input, nChannels, inputHeight, inputWidth, outputHeight, outputWidth, output);

    return CUDA(cudaGetLastError());
}

//gpu operation for bilinear upsampling
template <typename T>
// TODO
__global__ void gpuResizeBilinear( float2 scale, T* input, int iWidth, T* output, int oWidth, int oHeight ) { }

// bilinear upsampling
cudaError_t cudaResizeBilinear( float* input, size_t inputWidth, size_t inputHeight,
                        float* output, size_t outputWidth, size_t outputHeight )
{
    // TODO
    return CUDA(cudaGetLastError());
}

template cudaError_t cudaResizeNearestNeighbor<float>(float*, int, int, int, float*, cudaStream_t);
template cudaError_t cudaResizeNearestNeighbor<__half>(__half*, int, int, int, __half*, cudaStream_t);
