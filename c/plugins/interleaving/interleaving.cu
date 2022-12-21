#include "interleaving.h"
#include <iostream>
#include <cassert>

// Define the CUDA kernel.
template <typename T>
__global__ void gpuInterleave( int out_size, int N, int C, int H, int W, T* in1, T* in2, T* in3, T* in4, T* out) 
{ 
  //int index = blockIdx.x * blockDim.x + threadIdx.x;  
  for( int index = blockIdx.x * blockDim.x + threadIdx.x; index < out_size; index += blockDim.x * gridDim.x )
  {
    int n = index / (C * H * W);   
    int c = (index % (C * H * W)) / (H * W);
    int h = (index % (H * W)) / W;
    int w = index % W;

    int is_h_even = h % 2;
    int is_w_even = w % 2;

    int index_in_input = n * C * H * W / 4 + c * H * W / 4 + (h / 2) * W / 2 + (w / 2);
    //printf("Fill output index: %d %d %d\n index in input is: %d\n", c, h, w, index_in_input);
    
    if( !is_h_even )
    {
        if( !is_w_even )
        {
            out[index] = in1[index_in_input];
        }
        else
        {
            out[index] = in2[index_in_input];
        }
    }
    else
    {
        if( !is_w_even )
        {
            out[index] = in3[index_in_input];
        }
        else
        {
            out[index] = in4[index_in_input];
        }
    }
  }  
}  

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
cudaError_t cudaInterleave(T* in1, T* in2, T* in3, T* in4,
                     int inputChannels, int inputHeight, int inputWidth, T* out, cudaStream_t stream) 
{
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  int size = inputChannels * inputWidth * inputHeight * 4;
  int outputChannels = inputChannels;
  int outputWidth = inputWidth * 2;
  int outputHeight = inputHeight * 2;
  gpuInterleave<T>
      <<<block_count, thread_per_block, 0, stream>>>( size, 1, outputChannels, outputHeight, outputWidth, in1, in2, in3, in4, out);
  return CUDA(cudaGetLastError());
}

template cudaError_t cudaInterleave<float>(float*, float*, float*, float*, int, int, int, float*, cudaStream_t);
template cudaError_t cudaInterleave<__half>(__half*, __half*, __half*, __half*, int, int, int, __half*, cudaStream_t);