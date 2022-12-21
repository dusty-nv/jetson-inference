// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef REDTAIL_INTERNAL_UTILS_H
#define REDTAIL_INTERNAL_UTILS_H

#include <cassert>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <cudnn.h>

#include "internal_macros.h"
#include "redtail_tensorrt_plugins.h"

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Constants.
// REVIEW alexeyk: float ony for now, refactor to support FP16/Int8.
// -----------------------------------------------------------------
struct Consts
{
    static const float kZero;
    static const float kOne;

public:
    Consts(Consts&&) = delete;
};

// -----------------------------------------------------------------
// Various Dims helper functions.
// -----------------------------------------------------------------
class DimsUtils
{
public:
    static size_t getTensorSize(Dims dims);

    static Dims   getStrides(Dims dims);

    static bool   areEqual(Dims d1, Dims d2);

    static std::string toString(Dims dims);

public:
    DimsUtils(DimsUtils&&) = delete;
};

// -----------------------------------------------------------------
// String helpers.
// -----------------------------------------------------------------
class StrUtils
{
public:
    static std::string toString(DataType type);

    static std::string toString(PluginFormat format);

public:
    StrUtils(DimsUtils&&) = delete;
};

// -----------------------------------------------------------------
// Conversion helpers.
// -----------------------------------------------------------------
cudnnDataType_t trtToCudnnDataType(DataType trt_data_type);

// -----------------------------------------------------------------
// CUDA kernels interface.
// -----------------------------------------------------------------
class CudaKernels
{
public:
    template<typename T>
    static cudaError_t computeCostVolume(DataType data_type, const T* left, const T* right, Dims in_dims, T* cost_vol, Dims out_dims,
                                         cudaStream_t stream);

    template<typename T>
    static cudaError_t computeCorrCostVolume(DataType data_type, const T* left, const T* right, Dims in_dims, T* cost_vol, Dims out_dims,
                                             cudaStream_t stream);

    template<typename T>
    static cudaError_t addDBiasTo3DConv(const T* bias, Dims bias_dims, T* conv, Dims conv_dims, cudaStream_t stream);

    static cudaError_t fp32Tofp16(const float* src,    uint16_t* dst, size_t size, cudaStream_t stream);
    static cudaError_t fp16Tofp32(const uint16_t* src, float* dst,    size_t size, cudaStream_t stream);

public:
    CudaKernels(CudaKernels&&) = delete;
};

// Template instantiation.
template<>
cudaError_t CudaKernels::computeCostVolume(DataType data_type, const float*, const float*, Dims, float*, Dims, cudaStream_t);

template<>
cudaError_t CudaKernels::computeCorrCostVolume(DataType data_type, const float* left, const float* right, Dims in_dims, 
                                               float* cost_vol, Dims out_dims, cudaStream_t stream);

template<>
cudaError_t CudaKernels::addDBiasTo3DConv(const float*, Dims, float*, Dims, cudaStream_t);

// -----------------------------------------------------------------
// Simple implementation of IPluginContainer.
// The container keeps pointers to created plugins.
// -----------------------------------------------------------------
class PluginContainer: public IPluginContainer
{
public:
    PluginContainer(ILogger& log):
        log_(log)
    {
    }

    ~PluginContainer() noexcept = default;

    // Also disables copy ctor/assign ops.
    PluginContainer(PluginContainer&&) = delete;

    // ELU plugin.
    IPlugin* createEluPlugin(DataType data_type, std::string name) override;
    IPlugin* deserializeEluPlugin(const char* name, const void* data, size_t size) override;

    // Cost volume plugin.
    IPlugin* createCostVolumePlugin(DataType data_type, CostVolumeType cv_type, int max_disparity,
                                    std::string name) override;
    IPlugin* deserializeCostVolumePlugin(const char* name, const void* data, size_t size) override;

    // 3D convolution.
    IPlugin* createConv3DPlugin(Conv3DType conv_type, Dims kernel_dims,
                                Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                Weights kernel_weights, Weights bias_weights,
                                std::string name) override;

    // Transposed 3D convolution.
    IPlugin* createConv3DTransposePlugin(Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                                         Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                         Weights kernel_weights, Weights bias_weights,
                                         std::string name) override;

    // Transform.
    IPlugin* createTransformPlugin(Permutation permutation, std::string name) override;

    // Padding.
    IPlugin* createPaddingPlugin(DimsNCHW pad_start, DimsNCHW pad_end,
                                 std::string name) override;

    IPlugin* createSlicePlugin(Dims dims, Dims slice_start, Dims slice_end,
                               std::string name) override;

    IPlugin* createSoftargmaxPlugin(DataType data_type, SoftargmaxType sm_type, std::string name) override;
    IPlugin* deserializeSoftargmaxPlugin(const char* name, const void* data, size_t size) override;

private:
    // TensorRT IPlugin interface has protected dtor so cannot use unique_ptr
    // or delete plugins in the container dtor.
    std::vector<IPlugin*> plugins_;
    std::mutex            lock_;

    ILogger& log_;
};

void reportError(cudaError_t status, const char* file, int line, const char* func, ILogger& log);
void reportError(cudnnStatus_t status, const char* file, int line, const char* func, ILogger& log);

template<typename S, typename T=S>
void write_stream(S val, std::ostringstream& ss)
{
    auto dst = (T)val;
    ss.write((const char*)&dst, sizeof(dst));
}

template<typename T>
T read_stream(std::istringstream& ss)
{
    T res;
    ss.read((char *)&res, sizeof(res));
    return res;
}

} }

#endif