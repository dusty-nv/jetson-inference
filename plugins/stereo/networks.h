
// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef NETWORKS_H
#define NETWORKS_H

#include <NvInfer.h>
#include <string>
#include <unordered_map>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

using weight_map = std::unordered_map<std::string, Weights>;

class IPluginContainer;

// NVSmall DNN: 1025x321 input, 96 max disparity.
INetworkDefinition* createNVSmall1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                 DimsCHW img_dims, const weight_map& weights, DataType data_type, ILogger& log);

// Tiny version of NVSmall DNN: 513x161 input, 48 max disparity.
INetworkDefinition* createNVTiny513x161Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                               DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                               ILogger& log);

// Baseline ResNet-18 DNN: 1025x321 input, 136 max disparity.
INetworkDefinition* createResNet18_1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                   DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                                   ILogger& log);

// ResNet18_2D DNN: 513x256 input, 96 max disparity.
INetworkDefinition* createResNet18_2D_513x257Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                                     DimsCHW img_dims, const weight_map& weights, DataType data_type, ILogger& log);
}}

#endif