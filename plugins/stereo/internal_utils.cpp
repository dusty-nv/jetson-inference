// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>

namespace redtail { namespace tensorrt
{

const float Consts::kZero = 0.0f;
const float Consts::kOne  = 1.0f;

// -----------------------------------------------------------------
// DimsUtils implementation.
// -----------------------------------------------------------------
size_t DimsUtils::getTensorSize(Dims dims)
{
    assert(dims.nbDims >= 1);
    size_t res = std::accumulate(dims.d, dims.d + dims.nbDims, (size_t)1, std::multiplies<size_t>());
    // Dims.d is int, so overflow check.
    assert(res == (size_t)std::accumulate(dims.d, dims.d + dims.nbDims, (int)1, std::multiplies<int>()));
    return res;
}

Dims DimsUtils::getStrides(Dims dims)
{
    Dims strides;
    strides.nbDims = dims.nbDims;
    strides.d[strides.nbDims - 1] = 1;
    for (int i = strides.nbDims - 2; i >= 0; i--)
    {
        strides.d[i]    = strides.d[i + 1] * dims.d[i + 1];
        strides.type[i] = DimensionType::kSPATIAL;
    }
    return strides;
}

// Equality check. The function checks for shape and size,
// but not for dimension types.
bool DimsUtils::areEqual(Dims d1, Dims d2)
{
    if (d1.nbDims != d2.nbDims)
        return false;
    return std::equal(d1.d, d1.d + d1.nbDims, d2.d);
}

std::string DimsUtils::toString(Dims dims)
{
    std::ostringstream str;
    str << "{";
    for (int i = 0; i < dims.nbDims; i++)
    {
        str << std::fixed << std::setw(4) << dims.d[i];
        if (i < dims.nbDims - 1)
            str << ",";
    }
    str << "}";
    return str.str();
}

// -----------------------------------------------------------------
// String helpers.
// -----------------------------------------------------------------
std::string StrUtils::toString(DataType type)
{
    switch (type)
    {
    case DataType::kFLOAT:
        return "Float";
    case DataType::kHALF:
        return "Half";
    case DataType::kINT8:
        return "INT8";
    case DataType::kINT32:
        return "INT32";
    default:
        return "Unknown (" + std::to_string((int)type) + ")";
    }
}

std::string StrUtils::toString(PluginFormat format)
{
    switch (format)
    {
    case PluginFormat::kNCHW:
        return "NCHW";
    case PluginFormat::kNC2HW2:
        return "NC2HW2";
    case PluginFormat::kNHWC8:
        return "NHWC8";
    default:
        return "Unknown (" + std::to_string((int)format) + ")";
    }
}

// -----------------------------------------------------------------
// Conversion helpers.
// -----------------------------------------------------------------
cudnnDataType_t trtToCudnnDataType(DataType trt_data_type)
{
    // REVIEW alexeyk: fp32/16 only for now.
    assert(trt_data_type == DataType::kFLOAT || trt_data_type == DataType::kHALF);

    if (trt_data_type == DataType::kFLOAT)
        return CUDNN_DATA_FLOAT;
    if (trt_data_type == DataType::kHALF)
        return CUDNN_DATA_HALF;
    return (cudnnDataType_t)-1;
}

// -----------------------------------------------------------------
// PluginContainer implementation.
// -----------------------------------------------------------------

/*std::unique_ptr<IPluginContainer> IPluginContainer::create(ILogger& log)
{
    return std::make_unique<PluginContainer>(log);
}*/

IPluginContainer* IPluginContainer::create(ILogger& log)
{
    return new PluginContainer(log);
}

// -----------------------------------------------------------------
// Plugins helper functions.
// -----------------------------------------------------------------
IPluginLayer* addPlugin(INetworkDefinition& network, ITensor* const* inputs, int num_inputs, IPlugin* plugin)
{
    auto plugin_ext   = dynamic_cast<IPluginExt*>(plugin);
    auto plugin_layer = plugin_ext != nullptr
                        ? network.addPluginExt(inputs, num_inputs, *plugin_ext)
                        : network.addPlugin(inputs, num_inputs, *plugin);
    return plugin_layer;
}

ILayer* addElu(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DataType data_type, const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createEluPlugin(data_type, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto     layer    = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addCostVolume(IPluginContainer& plugin_factory, INetworkDefinition& network,
                      ITensor& left_input, ITensor& right_input,
                      CostVolumeType cv_type, int max_disparity,
                      DataType data_type, const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createCostVolumePlugin(data_type, cv_type, max_disparity, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&left_input, &right_input};
    auto layer = addPlugin(network, inputs, 2, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addConv3D(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                  Conv3DType conv_type, Dims kernel_dims, Dims stride_dims,
                  Dims pad_start_dims, Dims pad_end_dims,
                  Weights kernel_weights, Weights bias_weights,
                  const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createConv3DPlugin(conv_type, kernel_dims,
                                                    stride_dims, pad_start_dims, pad_end_dims,
                                                    kernel_weights, bias_weights,
                                                    name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto layer = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addConv3DTranspose(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                           Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                           Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                           Weights kernel_weights, Weights bias_weights,
                           const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createConv3DTransposePlugin(conv_type, kernel_dims, out_dims,
                                                             stride_dims, pad_start_dims, pad_end_dims,
                                                             kernel_weights, bias_weights,
                                                             name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto layer = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addSlice(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                 Dims dims, Dims slice_start, Dims slice_end,
                 const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createSlicePlugin(dims, slice_start, slice_end, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto     layer    = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addTransform(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                     Permutation permutation,
                     const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createTransformPlugin(permutation, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto     layer    = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addPad(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DimsNCHW pad_start, DimsNCHW pad_end,
               const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createPaddingPlugin(pad_start, pad_end, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto     layer    = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

ILayer* addSoftargmax(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                      SoftargmaxType sm_type, DataType data_type, const std::string& name)
{
    // Create plugin.
    auto plugin = plugin_factory.createSoftargmaxPlugin(data_type, sm_type, name);
    assert(plugin != nullptr);
    // Add to the network.
    ITensor* inputs[] = {&input};
    auto     layer    = addPlugin(network, inputs, 1, plugin);
    assert(layer != nullptr);
    return layer;
}

// -----------------------------------------------------------------
// Error checking and reporting methods.
// -----------------------------------------------------------------
void reportError(cudaError_t status, const char* file, int line, const char* func, ILogger& log)
{
    if (status == cudaSuccess)
        return;
    std::ostringstream str;
    str << file << ":" << line << ": " << func << ": CUDA error "
        << status << " (" << cudaGetErrorName(status) << ": " << cudaGetErrorString(status) << ").";
    log.log(ILogger::Severity::kERROR, str.str().c_str());
    assert(status == cudaSuccess);
}

void reportError(cudnnStatus_t status, const char* file, int line, const char* func, ILogger& log)
{
    if (status == CUDNN_STATUS_SUCCESS)
        return;
    std::ostringstream str;
    str << file << ":" << line << ": " << func << ": cuDNN error " << status << " (" << cudnnGetErrorString(status) << ").";
    log.log(ILogger::Severity::kERROR, str.str().c_str());
    assert(status == CUDNN_STATUS_SUCCESS);
}

// -----------------------------------------------------------------
// Plugin factory used in (de)serialization.
// -----------------------------------------------------------------
StereoDnnPluginFactory::StereoDnnPluginFactory(IPluginContainer& container):
    container_(container)
{
}

IPlugin* StereoDnnPluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    assert(serialLength >= sizeof(int32_t));
    size_t bytes_read = 0;
    auto ptr          = (const uint8_t*)serialData;
    auto plugin_type  = (PluginType)*(int32_t*)ptr;
    bytes_read += sizeof(int32_t);
    ptr += bytes_read;

    IPlugin* plugin = nullptr;
    switch (plugin_type)
    {
    case PluginType::kElu:
            plugin = container_.deserializeEluPlugin(layerName, ptr, serialLength - bytes_read);
        break;
    case PluginType::kCostVolume:
            plugin = container_.deserializeCostVolumePlugin(layerName, ptr, serialLength - bytes_read);
        break;
    case PluginType::kSoftargmax:
            plugin = container_.deserializeSoftargmaxPlugin(layerName, ptr, serialLength - bytes_read);
        break;
    }

    return plugin;
}

} }
