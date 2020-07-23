// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef REDTAIL_CONV_UTILS_H
#define REDTAIL_CONV_UTILS_H

#include <NvInfer.h>
#include <cudnn.h>
#include "redtail_tensorrt_plugins.h"

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Contains various helper methods that set cudNN convolution descriptors.
//
// Notes on 3D convolution implementation in cuDNN:
// 1. Convolution input is represnted as 4D tensor (ignoring batch size dimension):
//    (C, Di, Hi, Wi), where C is number of 3D feature maps, Di, Hi, Wi - 
//    depth, height and width of the input over which convolution is performed.
//    Convolution is performed by the filter (described later in details) over
//    (Di, Hi, Wi) dimensions. This is similar to 2D case where convolution
//    is performed over (Hi,Wi) dimensions over input that has C feature maps.
// 2. Convolution filter is represented as 5D tensor (cuDNN-like notation):
//    (K, C, V, R, S), where K is number of output feature maps (similar to 2D convo),
//    C - must be equal to C of the input (similar to 2D convo), and V, R, S - 
//    dimensions of the filter that go along Di, Hi, Wi dimensions of the input.
// 3. Convolution output is represented as 4D tensor (ignoring batch dimension):
//    (K, Do, Ho, Wo), where K is number of output feature maps (equal to K dim
//    of the filter), and Do, Ho, Wo are dimensions of the output that are 
//    computed according to usual rules of convolution dimensions, taking into
//    account padding and strides.
//
// This is different from 3D convolution implementation in some other
// DL toolkits, for example, TensorFlow conv3d operator computes it differently
// REVIEW alexeyk: finish documenting Conv3DType.
// 
// We also have to abuse TensorRT DimsNCHW type in few places to represent 3D convolutions
// as TRT (as of v3.0) does not support generic 4D/5D tensors - it fails with assert
// somewhere in the guts of TRT when using generic Dims type.
// -----------------------------------------------------------------
class ConvUtils
{
public:
    // -----------------------------------------------------------------
    // Sets descriptor for 3D convolution input/output.
    // This method is used in:
    // - Conv3DPlugin: 
    //      sets plugin input descriptor which is a convolution input.
    //      sets plugin output descriptor which is a convolution output.
    // - Conv3DTransposePlugin:
    //      sets plugin output descriptor which is a transposed convolution output.
    //      sets plugin input descriptor which is a transposed convolution input.
    //
    // dims      : tensor dimensions, must have rank 4.
    // batch_size: batch size, must be positive.
    // -----------------------------------------------------------------
    static void setConv3DTensorDescriptor(Conv3DType conv_type, Dims dims, int batch_size,
                                          cudnnDataType_t data_type, cudnnTensorDescriptor_t& desc,
                                          ILogger& log);

    // -----------------------------------------------------------------
    // Sets convolution filter and operation descriptors for 3D convolution.
    // This method is used in:
    // - Conv3DPlugin: 
    // - Conv3DTransposePlugin:
    //
    // w_dims:
    //   Filter dimensions, must have rank 5. The filter tensor is assumed to be in (K, C, V, R, S) format.
    // pad_dims:
    //   Input padding dimensions, must have rank 3. The padding is for (Di, Hi, Wi) dimensions of the
    //   convolution input (transposed conovlution output).
    // stride_dims:
    //   Filter stride dimensions, must have rank 3. The stride is for (Di, Hi, Wi) dimensions of the
    //   convolution input (transposed conovlution output).
    // -----------------------------------------------------------------
    static void setConv3DOperationDescriptors(Conv3DType conv_type, Dims w_dims, Dims stride_dims, Dims pad_dims,
                                              cudnnDataType_t data_type,
                                              cudnnFilterDescriptor_t& w_desc, cudnnConvolutionDescriptor_t& c_desc,
                                              ILogger& log);

    // -----------------------------------------------------------------
    // Returns convolution output dimensions given input, filter and
    // convolution descriptors.
    // -----------------------------------------------------------------
    static Dims getConv3DOutputDims(cudnnConvolutionDescriptor_t c_desc, cudnnTensorDescriptor_t x_desc,
                                    cudnnFilterDescriptor_t w_desc, ILogger& log);

    static void setConv3DBiasDescriptor(Dims dims, cudnnDataType_t data_type,
                                        cudnnTensorDescriptor_t& desc, ILogger& log);

public:
    ConvUtils(ConvUtils&&) = delete;
};

} }

#endif