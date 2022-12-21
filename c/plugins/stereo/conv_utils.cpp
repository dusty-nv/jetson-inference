// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "conv_utils.h"
#include <array>
#include <cassert>
#include "internal_utils.h"

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

void ConvUtils::setConv3DTensorDescriptor(Conv3DType conv_type, Dims dims, int batch_size,
                                          cudnnDataType_t data_type, cudnnTensorDescriptor_t& desc,
                                          ILogger& log)
{
    assert(dims.nbDims == 4);
    assert(batch_size > 0);
    assert(desc != nullptr);

    int c = dims.d[0];
    int d = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];
    
    // Reshape input if needed.
    if (conv_type == Conv3DType::kTensorFlow)
    {
        c = 1;
        d = dims.d[0] * dims.d[1];
    }

    std::array<int, 5> full_dims = { batch_size, c, d, h, w };
    std::array<int, 5> strides   = { c * d * h * w,
                                     d * h * w,
                                     h * w,
                                     w,
                                     1};
    // Sanity check.
    assert((int64_t)c * d * h * w == c * d * h * w);

    CHECKL(cudnnSetTensorNdDescriptor(desc, data_type, full_dims.size(), full_dims.data(), strides.data()), log);
}

void ConvUtils::setConv3DOperationDescriptors(Conv3DType conv_type, Dims w_dims, Dims stride_dims, Dims pad_dims,
                                              cudnnDataType_t data_type,
                                              cudnnFilterDescriptor_t& w_desc, cudnnConvolutionDescriptor_t& c_desc,
                                              ILogger& log)
{
    assert(w_dims.nbDims      == 5);
    assert(stride_dims.nbDims == 3);
    assert(pad_dims.nbDims    == 3);
    assert(w_desc != nullptr);
    assert(c_desc != nullptr);

    // Reshape/update if needed.
    if (conv_type == Conv3DType::kTensorFlow)
    {
        int c = w_dims.d[1];
        int v = w_dims.d[2];

        // Reshape filter.
        w_dims.d[1] = 1;
        w_dims.d[2] = c * v;

        // Update stride.
        stride_dims.d[0] *= v;

        // Update padding.
        pad_dims.d[0] *= v;
    }

    // Set filter descriptor.
    CHECKL(cudnnSetFilterNdDescriptor(w_desc, data_type, CUDNN_TENSOR_NCHW, w_dims.nbDims, w_dims.d), log);

    // Set convolution descriptor.
    std::array<int, 3> dilation = { 1, 1, 1 };
    CHECKL(cudnnSetConvolutionNdDescriptor(c_desc, pad_dims.nbDims, pad_dims.d, stride_dims.d, dilation.data(),
                                           CUDNN_CROSS_CORRELATION, data_type), log);
}

Dims ConvUtils::getConv3DOutputDims(cudnnConvolutionDescriptor_t c_desc, cudnnTensorDescriptor_t x_desc,
                                    cudnnFilterDescriptor_t w_desc, ILogger& log)
{
    assert(c_desc != nullptr);
    assert(x_desc != nullptr);
    assert(w_desc != nullptr);

    // 5D tensor in (N, K, D, H, W) format.
    Dims y_dims;
    y_dims.nbDims = 5;
    CHECKL(cudnnGetConvolutionNdForwardOutputDim(c_desc, x_desc, w_desc, y_dims.nbDims, y_dims.d), log);
    return y_dims;
}

void ConvUtils::setConv3DBiasDescriptor(Dims dims, cudnnDataType_t data_type,
                                        cudnnTensorDescriptor_t& desc, ILogger& log)
{
    assert(dims.nbDims == 5);
    assert(desc != nullptr);

    // REVIEW alexeyk: see the comment in tensorrt_model_builder.py re: the stride issue in Conv3D.
    std::array<int, 5> strides = {dims.d[1] * dims.d[2] * dims.d[3] * dims.d[4],
                                  dims.d[2] * dims.d[3] * dims.d[4],
                                  dims.d[3] * dims.d[4],
                                  dims.d[4],
                                  1};
    CHECKL(cudnnSetTensorNdDescriptor(desc, data_type, dims.nbDims, dims.d, strides.data()), log);
}

} }