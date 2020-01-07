#ifndef __STRIDED_SLICE_H__
#define __STRIDED_SLICE_H__

#include <cublas_v2.h>
#include "cudaUtility.h"

/**
 * Function for upsampling image or feature map using bilinear interpolation
 * @ingroup util
 */
template<typename T>
cudaError_t cudaSlice( T* input, int inputChannels, int inputHeight, int inputWidth,
                        T* output, cudaStream_t stream );

#endif