// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef REDTAIL_TENSORRT_PLUGINS_H
#define REDTAIL_TENSORRT_PLUGINS_H

#include <memory>
#include <NvInfer.h>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// 3D convolution type used to setup convolution descriptors.
// -----------------------------------------------------------------
enum class Conv3DType
{
    kCuDnn      = 0,
    kTensorFlow = 1
};

// -----------------------------------------------------------------
// Cost volume type.
// -----------------------------------------------------------------
enum class CostVolumeType
{
    kDefault     = 0, // Default    : 2 3D inputs are transformed into one 4D output.
    kCorrelation = 1  // Correlation: 2 3D inputs are transformed into one 3D output.
};

// -----------------------------------------------------------------
// Softargmax type.
// -----------------------------------------------------------------
enum class SoftargmaxType
{
    kMax = 0, // Computes softargmax.
    kMin = 1  // Computes softargmin.
};

// -----------------------------------------------------------------
// Plugin container/factory.
// TensorRT does not manage plugins and requires a plugin lifetime
// to be the same as any TRT engine.
// Each plugin create* function returns naked pointer as expected by TRT,
// with IPluginContainer managing the plugin's lifetime.
// -----------------------------------------------------------------
class IPluginContainer
{
public:
    virtual ~IPluginContainer() = default;

    // ELU plugin.
    virtual IPlugin* createEluPlugin(DataType data_type, std::string name) = 0;
    virtual IPlugin* deserializeEluPlugin(const char* name, const void* data, size_t size) = 0;

    // Cost volume plugin.
    virtual IPlugin* createCostVolumePlugin(DataType data_type, CostVolumeType cv_type, int max_disparity,
                                            std::string name) = 0;
    virtual IPlugin* deserializeCostVolumePlugin(const char* name, const void* data, size_t size) = 0;

    // 3D convolution.
    virtual IPlugin* createConv3DPlugin(Conv3DType conv_type, Dims kernel_dims,
                                        Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                        Weights kernel_weights, Weights bias_weights,
                                        std::string name) = 0;

    virtual IPlugin* createConv3DTransposePlugin(Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                                                 Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                                                 Weights kernel_weights, Weights bias_weights,
                                                 std::string name) = 0;

    virtual IPlugin* createTransformPlugin(Permutation permutation, std::string name) = 0;

    virtual IPlugin* createPaddingPlugin(DimsNCHW pad_start, DimsNCHW pad_end,
                                         std::string name) = 0;

    virtual IPlugin* createSlicePlugin(Dims dims, Dims slice_start, Dims slice_end,
                                       std::string name) = 0;

    virtual IPlugin* createSoftargmaxPlugin(DataType data_type, SoftargmaxType sm_type, std::string name) = 0;
    virtual IPlugin* deserializeSoftargmaxPlugin(const char* name, const void* data, size_t size) = 0;

    //static std::unique_ptr<IPluginContainer> create(ILogger& log);
    static IPluginContainer* create(ILogger& log);
};

// -----------------------------------------------------------------
// Plugins helper functions.
// -----------------------------------------------------------------
ILayer* addElu(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DataType data_type, const std::string& name);

ILayer* addCostVolume(IPluginContainer& plugin_factory, INetworkDefinition& network,
                      ITensor& left_input, ITensor& right_input,
                      CostVolumeType cv_type, int max_disparity,
                      DataType data_type, const std::string& name);

ILayer* addConv3D(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                  Conv3DType conv_type, Dims kernel_dims, Dims stride_dims,
                  Dims pad_start_dims, Dims pad_end_dims,
                  Weights kernel_weights, Weights bias_weights,
                  const std::string& name);

ILayer* addConv3DTranspose(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                           Conv3DType conv_type, Dims kernel_dims, Dims out_dims,
                           Dims stride_dims, Dims pad_start_dims, Dims pad_end_dims,
                           Weights kernel_weights, Weights bias_weights,
                           const std::string& name);

ILayer* addSlice(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                 Dims dims, Dims slice_start, Dims slice_end,
                 const std::string& name);

ILayer* addTransform(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                     Permutation permutation,
                     const std::string& name);

ILayer* addPad(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
               DimsNCHW pad_start, DimsNCHW pad_end,
               const std::string& name);

ILayer* addSoftargmax(IPluginContainer& plugin_factory, INetworkDefinition& network, ITensor& input,
                      SoftargmaxType sm_type, DataType data_type, const std::string& name);

// -----------------------------------------------------------------
// Plugin factory used in (de)serialization.
// -----------------------------------------------------------------
class StereoDnnPluginFactory: public IPluginFactory
{
public:
    enum class PluginType
    {
        kElu        = 0,
        kCostVolume = 1,
        kSoftargmax = 2
    };

public:
    StereoDnnPluginFactory(IPluginContainer& container);

    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

private:
    IPluginContainer& container_;
};

} }

#endif // REDTAIL_TENSORRT_PLUGINS_H
