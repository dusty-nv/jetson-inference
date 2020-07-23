// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

//-------------------------------------------------------------------
// !!! This file was automatically generated. Do not edit. !!!
//-------------------------------------------------------------------

#include <NvInfer.h>
#include <cassert>
#include <string>
#include <unordered_map>
#include "redtail_tensorrt_plugins.h"

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

using weight_map = std::unordered_map<std::string, Weights>;

INetworkDefinition* createResNet18_1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
                                     DimsCHW img_dims, const weight_map& weights, DataType data_type,
                                     ILogger& log)
{
    INetworkDefinition* network = builder.createNetwork();
    assert(network != nullptr);
    // Input tensor.
    auto left = network->addInput("left", DataType::kFLOAT, img_dims);
    assert(left != nullptr);

    // Input tensor.
    auto right = network->addInput("right", DataType::kFLOAT, img_dims);
    assert(right != nullptr);

    // Scaling op.
    auto left_scale = network->addScale(*left, ScaleMode::kUNIFORM,
                                 weights.at("left_scale_shift"), weights.at("left_scale_scale"), weights.at("left_scale_power"));
    assert(left_scale != nullptr);
    left_scale->setName("left_scale");

    // Scaling op.
    auto right_scale = network->addScale(*right, ScaleMode::kUNIFORM,
                                 weights.at("right_scale_shift"), weights.at("right_scale_scale"), weights.at("right_scale_power"));
    assert(right_scale != nullptr);
    right_scale->setName("right_scale");

    // left_conv1 convolution op.
    auto left_conv1 = network->addConvolution(*left_scale->getOutput(0), 32, DimsHW {5, 5},
                                       weights.at("left_conv1_k"), weights.at("left_conv1_b"));
    assert(left_conv1 != nullptr);
    left_conv1->setName("left_conv1");
    left_conv1->setStride( DimsHW {2, 2});
    left_conv1->setPadding(DimsHW {2, 2});

    // left_conv1_act ELU activation op.
    auto left_conv1_act = addElu(plugin_factory, *network, *left_conv1->getOutput(0), data_type, "left_conv1_act");
    assert(left_conv1_act != nullptr);
    left_conv1_act->setName("left_conv1_act");

    // right_conv1 convolution op.
    auto right_conv1 = network->addConvolution(*right_scale->getOutput(0), 32, DimsHW {5, 5},
                                       weights.at("right_conv1_k"), weights.at("right_conv1_b"));
    assert(right_conv1 != nullptr);
    right_conv1->setName("right_conv1");
    right_conv1->setStride( DimsHW {2, 2});
    right_conv1->setPadding(DimsHW {2, 2});

    // right_conv1_act ELU activation op.
    auto right_conv1_act = addElu(plugin_factory, *network, *right_conv1->getOutput(0), data_type, "right_conv1_act");
    assert(right_conv1_act != nullptr);
    right_conv1_act->setName("right_conv1_act");

    // left_resblock1_conv1 convolution op.
    auto left_resblock1_conv1 = network->addConvolution(*left_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock1_conv1_k"), weights.at("left_resblock1_conv1_b"));
    assert(left_resblock1_conv1 != nullptr);
    left_resblock1_conv1->setName("left_resblock1_conv1");
    left_resblock1_conv1->setStride( DimsHW {1, 1});
    left_resblock1_conv1->setPadding(DimsHW {1, 1});

    // left_resblock1_conv1_act ELU activation op.
    auto left_resblock1_conv1_act = addElu(plugin_factory, *network, *left_resblock1_conv1->getOutput(0), data_type, "left_resblock1_conv1_act");
    assert(left_resblock1_conv1_act != nullptr);
    left_resblock1_conv1_act->setName("left_resblock1_conv1_act");

    // left_resblock1_conv2 convolution op.
    auto left_resblock1_conv2 = network->addConvolution(*left_resblock1_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock1_conv2_k"), weights.at("left_resblock1_conv2_b"));
    assert(left_resblock1_conv2 != nullptr);
    left_resblock1_conv2->setName("left_resblock1_conv2");
    left_resblock1_conv2->setStride( DimsHW {1, 1});
    left_resblock1_conv2->setPadding(DimsHW {1, 1});

    // left_resblock1_conv2_add tensor add op.
    auto left_resblock1_conv2_add = network->addElementWise(*(left_resblock1_conv2->getOutput(0)), *(left_conv1_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(left_resblock1_conv2_add != nullptr);
    left_resblock1_conv2_add->setName("left_resblock1_conv2_add");

    // left_resblock1_conv2_add_act ELU activation op.
    auto left_resblock1_conv2_add_act = addElu(plugin_factory, *network, *left_resblock1_conv2_add->getOutput(0), data_type, "left_resblock1_conv2_add_act");
    assert(left_resblock1_conv2_add_act != nullptr);
    left_resblock1_conv2_add_act->setName("left_resblock1_conv2_add_act");

    // right_resblock1_conv1 convolution op.
    auto right_resblock1_conv1 = network->addConvolution(*right_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock1_conv1_k"), weights.at("right_resblock1_conv1_b"));
    assert(right_resblock1_conv1 != nullptr);
    right_resblock1_conv1->setName("right_resblock1_conv1");
    right_resblock1_conv1->setStride( DimsHW {1, 1});
    right_resblock1_conv1->setPadding(DimsHW {1, 1});

    // right_resblock1_conv1_act ELU activation op.
    auto right_resblock1_conv1_act = addElu(plugin_factory, *network, *right_resblock1_conv1->getOutput(0), data_type, "right_resblock1_conv1_act");
    assert(right_resblock1_conv1_act != nullptr);
    right_resblock1_conv1_act->setName("right_resblock1_conv1_act");

    // right_resblock1_conv2 convolution op.
    auto right_resblock1_conv2 = network->addConvolution(*right_resblock1_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock1_conv2_k"), weights.at("right_resblock1_conv2_b"));
    assert(right_resblock1_conv2 != nullptr);
    right_resblock1_conv2->setName("right_resblock1_conv2");
    right_resblock1_conv2->setStride( DimsHW {1, 1});
    right_resblock1_conv2->setPadding(DimsHW {1, 1});

    // right_resblock1_conv2_add tensor add op.
    auto right_resblock1_conv2_add = network->addElementWise(*(right_resblock1_conv2->getOutput(0)), *(right_conv1_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock1_conv2_add != nullptr);
    right_resblock1_conv2_add->setName("right_resblock1_conv2_add");

    // right_resblock1_conv2_add_act ELU activation op.
    auto right_resblock1_conv2_add_act = addElu(plugin_factory, *network, *right_resblock1_conv2_add->getOutput(0), data_type, "right_resblock1_conv2_add_act");
    assert(right_resblock1_conv2_add_act != nullptr);
    right_resblock1_conv2_add_act->setName("right_resblock1_conv2_add_act");

    // left_resblock2_conv1 convolution op.
    auto left_resblock2_conv1 = network->addConvolution(*left_resblock1_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock2_conv1_k"), weights.at("left_resblock2_conv1_b"));
    assert(left_resblock2_conv1 != nullptr);
    left_resblock2_conv1->setName("left_resblock2_conv1");
    left_resblock2_conv1->setStride( DimsHW {1, 1});
    left_resblock2_conv1->setPadding(DimsHW {1, 1});

    // left_resblock2_conv1_act ELU activation op.
    auto left_resblock2_conv1_act = addElu(plugin_factory, *network, *left_resblock2_conv1->getOutput(0), data_type, "left_resblock2_conv1_act");
    assert(left_resblock2_conv1_act != nullptr);
    left_resblock2_conv1_act->setName("left_resblock2_conv1_act");

    // left_resblock2_conv2 convolution op.
    auto left_resblock2_conv2 = network->addConvolution(*left_resblock2_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock2_conv2_k"), weights.at("left_resblock2_conv2_b"));
    assert(left_resblock2_conv2 != nullptr);
    left_resblock2_conv2->setName("left_resblock2_conv2");
    left_resblock2_conv2->setStride( DimsHW {1, 1});
    left_resblock2_conv2->setPadding(DimsHW {1, 1});

    // left_resblock2_conv2_add tensor add op.
    auto left_resblock2_conv2_add = network->addElementWise(*(left_resblock2_conv2->getOutput(0)), *(left_resblock1_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock2_conv2_add != nullptr);
    left_resblock2_conv2_add->setName("left_resblock2_conv2_add");

    // left_resblock2_conv2_add_act ELU activation op.
    auto left_resblock2_conv2_add_act = addElu(plugin_factory, *network, *left_resblock2_conv2_add->getOutput(0), data_type, "left_resblock2_conv2_add_act");
    assert(left_resblock2_conv2_add_act != nullptr);
    left_resblock2_conv2_add_act->setName("left_resblock2_conv2_add_act");

    // right_resblock2_conv1 convolution op.
    auto right_resblock2_conv1 = network->addConvolution(*right_resblock1_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock2_conv1_k"), weights.at("right_resblock2_conv1_b"));
    assert(right_resblock2_conv1 != nullptr);
    right_resblock2_conv1->setName("right_resblock2_conv1");
    right_resblock2_conv1->setStride( DimsHW {1, 1});
    right_resblock2_conv1->setPadding(DimsHW {1, 1});

    // right_resblock2_conv1_act ELU activation op.
    auto right_resblock2_conv1_act = addElu(plugin_factory, *network, *right_resblock2_conv1->getOutput(0), data_type, "right_resblock2_conv1_act");
    assert(right_resblock2_conv1_act != nullptr);
    right_resblock2_conv1_act->setName("right_resblock2_conv1_act");

    // right_resblock2_conv2 convolution op.
    auto right_resblock2_conv2 = network->addConvolution(*right_resblock2_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock2_conv2_k"), weights.at("right_resblock2_conv2_b"));
    assert(right_resblock2_conv2 != nullptr);
    right_resblock2_conv2->setName("right_resblock2_conv2");
    right_resblock2_conv2->setStride( DimsHW {1, 1});
    right_resblock2_conv2->setPadding(DimsHW {1, 1});

    // right_resblock2_conv2_add tensor add op.
    auto right_resblock2_conv2_add = network->addElementWise(*(right_resblock2_conv2->getOutput(0)), *(right_resblock1_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock2_conv2_add != nullptr);
    right_resblock2_conv2_add->setName("right_resblock2_conv2_add");

    // right_resblock2_conv2_add_act ELU activation op.
    auto right_resblock2_conv2_add_act = addElu(plugin_factory, *network, *right_resblock2_conv2_add->getOutput(0), data_type, "right_resblock2_conv2_add_act");
    assert(right_resblock2_conv2_add_act != nullptr);
    right_resblock2_conv2_add_act->setName("right_resblock2_conv2_add_act");

    // left_resblock3_conv1 convolution op.
    auto left_resblock3_conv1 = network->addConvolution(*left_resblock2_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock3_conv1_k"), weights.at("left_resblock3_conv1_b"));
    assert(left_resblock3_conv1 != nullptr);
    left_resblock3_conv1->setName("left_resblock3_conv1");
    left_resblock3_conv1->setStride( DimsHW {1, 1});
    left_resblock3_conv1->setPadding(DimsHW {1, 1});

    // left_resblock3_conv1_act ELU activation op.
    auto left_resblock3_conv1_act = addElu(plugin_factory, *network, *left_resblock3_conv1->getOutput(0), data_type, "left_resblock3_conv1_act");
    assert(left_resblock3_conv1_act != nullptr);
    left_resblock3_conv1_act->setName("left_resblock3_conv1_act");

    // left_resblock3_conv2 convolution op.
    auto left_resblock3_conv2 = network->addConvolution(*left_resblock3_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock3_conv2_k"), weights.at("left_resblock3_conv2_b"));
    assert(left_resblock3_conv2 != nullptr);
    left_resblock3_conv2->setName("left_resblock3_conv2");
    left_resblock3_conv2->setStride( DimsHW {1, 1});
    left_resblock3_conv2->setPadding(DimsHW {1, 1});

    // left_resblock3_conv2_add tensor add op.
    auto left_resblock3_conv2_add = network->addElementWise(*(left_resblock3_conv2->getOutput(0)), *(left_resblock2_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock3_conv2_add != nullptr);
    left_resblock3_conv2_add->setName("left_resblock3_conv2_add");

    // left_resblock3_conv2_add_act ELU activation op.
    auto left_resblock3_conv2_add_act = addElu(plugin_factory, *network, *left_resblock3_conv2_add->getOutput(0), data_type, "left_resblock3_conv2_add_act");
    assert(left_resblock3_conv2_add_act != nullptr);
    left_resblock3_conv2_add_act->setName("left_resblock3_conv2_add_act");

    // right_resblock3_conv1 convolution op.
    auto right_resblock3_conv1 = network->addConvolution(*right_resblock2_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock3_conv1_k"), weights.at("right_resblock3_conv1_b"));
    assert(right_resblock3_conv1 != nullptr);
    right_resblock3_conv1->setName("right_resblock3_conv1");
    right_resblock3_conv1->setStride( DimsHW {1, 1});
    right_resblock3_conv1->setPadding(DimsHW {1, 1});

    // right_resblock3_conv1_act ELU activation op.
    auto right_resblock3_conv1_act = addElu(plugin_factory, *network, *right_resblock3_conv1->getOutput(0), data_type, "right_resblock3_conv1_act");
    assert(right_resblock3_conv1_act != nullptr);
    right_resblock3_conv1_act->setName("right_resblock3_conv1_act");

    // right_resblock3_conv2 convolution op.
    auto right_resblock3_conv2 = network->addConvolution(*right_resblock3_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock3_conv2_k"), weights.at("right_resblock3_conv2_b"));
    assert(right_resblock3_conv2 != nullptr);
    right_resblock3_conv2->setName("right_resblock3_conv2");
    right_resblock3_conv2->setStride( DimsHW {1, 1});
    right_resblock3_conv2->setPadding(DimsHW {1, 1});

    // right_resblock3_conv2_add tensor add op.
    auto right_resblock3_conv2_add = network->addElementWise(*(right_resblock3_conv2->getOutput(0)), *(right_resblock2_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock3_conv2_add != nullptr);
    right_resblock3_conv2_add->setName("right_resblock3_conv2_add");

    // right_resblock3_conv2_add_act ELU activation op.
    auto right_resblock3_conv2_add_act = addElu(plugin_factory, *network, *right_resblock3_conv2_add->getOutput(0), data_type, "right_resblock3_conv2_add_act");
    assert(right_resblock3_conv2_add_act != nullptr);
    right_resblock3_conv2_add_act->setName("right_resblock3_conv2_add_act");

    // left_resblock4_conv1 convolution op.
    auto left_resblock4_conv1 = network->addConvolution(*left_resblock3_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock4_conv1_k"), weights.at("left_resblock4_conv1_b"));
    assert(left_resblock4_conv1 != nullptr);
    left_resblock4_conv1->setName("left_resblock4_conv1");
    left_resblock4_conv1->setStride( DimsHW {1, 1});
    left_resblock4_conv1->setPadding(DimsHW {1, 1});

    // left_resblock4_conv1_act ELU activation op.
    auto left_resblock4_conv1_act = addElu(plugin_factory, *network, *left_resblock4_conv1->getOutput(0), data_type, "left_resblock4_conv1_act");
    assert(left_resblock4_conv1_act != nullptr);
    left_resblock4_conv1_act->setName("left_resblock4_conv1_act");

    // left_resblock4_conv2 convolution op.
    auto left_resblock4_conv2 = network->addConvolution(*left_resblock4_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock4_conv2_k"), weights.at("left_resblock4_conv2_b"));
    assert(left_resblock4_conv2 != nullptr);
    left_resblock4_conv2->setName("left_resblock4_conv2");
    left_resblock4_conv2->setStride( DimsHW {1, 1});
    left_resblock4_conv2->setPadding(DimsHW {1, 1});

    // left_resblock4_conv2_add tensor add op.
    auto left_resblock4_conv2_add = network->addElementWise(*(left_resblock4_conv2->getOutput(0)), *(left_resblock3_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock4_conv2_add != nullptr);
    left_resblock4_conv2_add->setName("left_resblock4_conv2_add");

    // left_resblock4_conv2_add_act ELU activation op.
    auto left_resblock4_conv2_add_act = addElu(plugin_factory, *network, *left_resblock4_conv2_add->getOutput(0), data_type, "left_resblock4_conv2_add_act");
    assert(left_resblock4_conv2_add_act != nullptr);
    left_resblock4_conv2_add_act->setName("left_resblock4_conv2_add_act");

    // right_resblock4_conv1 convolution op.
    auto right_resblock4_conv1 = network->addConvolution(*right_resblock3_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock4_conv1_k"), weights.at("right_resblock4_conv1_b"));
    assert(right_resblock4_conv1 != nullptr);
    right_resblock4_conv1->setName("right_resblock4_conv1");
    right_resblock4_conv1->setStride( DimsHW {1, 1});
    right_resblock4_conv1->setPadding(DimsHW {1, 1});

    // right_resblock4_conv1_act ELU activation op.
    auto right_resblock4_conv1_act = addElu(plugin_factory, *network, *right_resblock4_conv1->getOutput(0), data_type, "right_resblock4_conv1_act");
    assert(right_resblock4_conv1_act != nullptr);
    right_resblock4_conv1_act->setName("right_resblock4_conv1_act");

    // right_resblock4_conv2 convolution op.
    auto right_resblock4_conv2 = network->addConvolution(*right_resblock4_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock4_conv2_k"), weights.at("right_resblock4_conv2_b"));
    assert(right_resblock4_conv2 != nullptr);
    right_resblock4_conv2->setName("right_resblock4_conv2");
    right_resblock4_conv2->setStride( DimsHW {1, 1});
    right_resblock4_conv2->setPadding(DimsHW {1, 1});

    // right_resblock4_conv2_add tensor add op.
    auto right_resblock4_conv2_add = network->addElementWise(*(right_resblock4_conv2->getOutput(0)), *(right_resblock3_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock4_conv2_add != nullptr);
    right_resblock4_conv2_add->setName("right_resblock4_conv2_add");

    // right_resblock4_conv2_add_act ELU activation op.
    auto right_resblock4_conv2_add_act = addElu(plugin_factory, *network, *right_resblock4_conv2_add->getOutput(0), data_type, "right_resblock4_conv2_add_act");
    assert(right_resblock4_conv2_add_act != nullptr);
    right_resblock4_conv2_add_act->setName("right_resblock4_conv2_add_act");

    // left_resblock5_conv1 convolution op.
    auto left_resblock5_conv1 = network->addConvolution(*left_resblock4_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock5_conv1_k"), weights.at("left_resblock5_conv1_b"));
    assert(left_resblock5_conv1 != nullptr);
    left_resblock5_conv1->setName("left_resblock5_conv1");
    left_resblock5_conv1->setStride( DimsHW {1, 1});
    left_resblock5_conv1->setPadding(DimsHW {1, 1});

    // left_resblock5_conv1_act ELU activation op.
    auto left_resblock5_conv1_act = addElu(plugin_factory, *network, *left_resblock5_conv1->getOutput(0), data_type, "left_resblock5_conv1_act");
    assert(left_resblock5_conv1_act != nullptr);
    left_resblock5_conv1_act->setName("left_resblock5_conv1_act");

    // left_resblock5_conv2 convolution op.
    auto left_resblock5_conv2 = network->addConvolution(*left_resblock5_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock5_conv2_k"), weights.at("left_resblock5_conv2_b"));
    assert(left_resblock5_conv2 != nullptr);
    left_resblock5_conv2->setName("left_resblock5_conv2");
    left_resblock5_conv2->setStride( DimsHW {1, 1});
    left_resblock5_conv2->setPadding(DimsHW {1, 1});

    // left_resblock5_conv2_add tensor add op.
    auto left_resblock5_conv2_add = network->addElementWise(*(left_resblock5_conv2->getOutput(0)), *(left_resblock4_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock5_conv2_add != nullptr);
    left_resblock5_conv2_add->setName("left_resblock5_conv2_add");

    // left_resblock5_conv2_add_act ELU activation op.
    auto left_resblock5_conv2_add_act = addElu(plugin_factory, *network, *left_resblock5_conv2_add->getOutput(0), data_type, "left_resblock5_conv2_add_act");
    assert(left_resblock5_conv2_add_act != nullptr);
    left_resblock5_conv2_add_act->setName("left_resblock5_conv2_add_act");

    // right_resblock5_conv1 convolution op.
    auto right_resblock5_conv1 = network->addConvolution(*right_resblock4_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock5_conv1_k"), weights.at("right_resblock5_conv1_b"));
    assert(right_resblock5_conv1 != nullptr);
    right_resblock5_conv1->setName("right_resblock5_conv1");
    right_resblock5_conv1->setStride( DimsHW {1, 1});
    right_resblock5_conv1->setPadding(DimsHW {1, 1});

    // right_resblock5_conv1_act ELU activation op.
    auto right_resblock5_conv1_act = addElu(plugin_factory, *network, *right_resblock5_conv1->getOutput(0), data_type, "right_resblock5_conv1_act");
    assert(right_resblock5_conv1_act != nullptr);
    right_resblock5_conv1_act->setName("right_resblock5_conv1_act");

    // right_resblock5_conv2 convolution op.
    auto right_resblock5_conv2 = network->addConvolution(*right_resblock5_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock5_conv2_k"), weights.at("right_resblock5_conv2_b"));
    assert(right_resblock5_conv2 != nullptr);
    right_resblock5_conv2->setName("right_resblock5_conv2");
    right_resblock5_conv2->setStride( DimsHW {1, 1});
    right_resblock5_conv2->setPadding(DimsHW {1, 1});

    // right_resblock5_conv2_add tensor add op.
    auto right_resblock5_conv2_add = network->addElementWise(*(right_resblock5_conv2->getOutput(0)), *(right_resblock4_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock5_conv2_add != nullptr);
    right_resblock5_conv2_add->setName("right_resblock5_conv2_add");

    // right_resblock5_conv2_add_act ELU activation op.
    auto right_resblock5_conv2_add_act = addElu(plugin_factory, *network, *right_resblock5_conv2_add->getOutput(0), data_type, "right_resblock5_conv2_add_act");
    assert(right_resblock5_conv2_add_act != nullptr);
    right_resblock5_conv2_add_act->setName("right_resblock5_conv2_add_act");

    // left_resblock6_conv1 convolution op.
    auto left_resblock6_conv1 = network->addConvolution(*left_resblock5_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock6_conv1_k"), weights.at("left_resblock6_conv1_b"));
    assert(left_resblock6_conv1 != nullptr);
    left_resblock6_conv1->setName("left_resblock6_conv1");
    left_resblock6_conv1->setStride( DimsHW {1, 1});
    left_resblock6_conv1->setPadding(DimsHW {1, 1});

    // left_resblock6_conv1_act ELU activation op.
    auto left_resblock6_conv1_act = addElu(plugin_factory, *network, *left_resblock6_conv1->getOutput(0), data_type, "left_resblock6_conv1_act");
    assert(left_resblock6_conv1_act != nullptr);
    left_resblock6_conv1_act->setName("left_resblock6_conv1_act");

    // left_resblock6_conv2 convolution op.
    auto left_resblock6_conv2 = network->addConvolution(*left_resblock6_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock6_conv2_k"), weights.at("left_resblock6_conv2_b"));
    assert(left_resblock6_conv2 != nullptr);
    left_resblock6_conv2->setName("left_resblock6_conv2");
    left_resblock6_conv2->setStride( DimsHW {1, 1});
    left_resblock6_conv2->setPadding(DimsHW {1, 1});

    // left_resblock6_conv2_add tensor add op.
    auto left_resblock6_conv2_add = network->addElementWise(*(left_resblock6_conv2->getOutput(0)), *(left_resblock5_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock6_conv2_add != nullptr);
    left_resblock6_conv2_add->setName("left_resblock6_conv2_add");

    // left_resblock6_conv2_add_act ELU activation op.
    auto left_resblock6_conv2_add_act = addElu(plugin_factory, *network, *left_resblock6_conv2_add->getOutput(0), data_type, "left_resblock6_conv2_add_act");
    assert(left_resblock6_conv2_add_act != nullptr);
    left_resblock6_conv2_add_act->setName("left_resblock6_conv2_add_act");

    // right_resblock6_conv1 convolution op.
    auto right_resblock6_conv1 = network->addConvolution(*right_resblock5_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock6_conv1_k"), weights.at("right_resblock6_conv1_b"));
    assert(right_resblock6_conv1 != nullptr);
    right_resblock6_conv1->setName("right_resblock6_conv1");
    right_resblock6_conv1->setStride( DimsHW {1, 1});
    right_resblock6_conv1->setPadding(DimsHW {1, 1});

    // right_resblock6_conv1_act ELU activation op.
    auto right_resblock6_conv1_act = addElu(plugin_factory, *network, *right_resblock6_conv1->getOutput(0), data_type, "right_resblock6_conv1_act");
    assert(right_resblock6_conv1_act != nullptr);
    right_resblock6_conv1_act->setName("right_resblock6_conv1_act");

    // right_resblock6_conv2 convolution op.
    auto right_resblock6_conv2 = network->addConvolution(*right_resblock6_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock6_conv2_k"), weights.at("right_resblock6_conv2_b"));
    assert(right_resblock6_conv2 != nullptr);
    right_resblock6_conv2->setName("right_resblock6_conv2");
    right_resblock6_conv2->setStride( DimsHW {1, 1});
    right_resblock6_conv2->setPadding(DimsHW {1, 1});

    // right_resblock6_conv2_add tensor add op.
    auto right_resblock6_conv2_add = network->addElementWise(*(right_resblock6_conv2->getOutput(0)), *(right_resblock5_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock6_conv2_add != nullptr);
    right_resblock6_conv2_add->setName("right_resblock6_conv2_add");

    // right_resblock6_conv2_add_act ELU activation op.
    auto right_resblock6_conv2_add_act = addElu(plugin_factory, *network, *right_resblock6_conv2_add->getOutput(0), data_type, "right_resblock6_conv2_add_act");
    assert(right_resblock6_conv2_add_act != nullptr);
    right_resblock6_conv2_add_act->setName("right_resblock6_conv2_add_act");

    // left_resblock7_conv1 convolution op.
    auto left_resblock7_conv1 = network->addConvolution(*left_resblock6_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock7_conv1_k"), weights.at("left_resblock7_conv1_b"));
    assert(left_resblock7_conv1 != nullptr);
    left_resblock7_conv1->setName("left_resblock7_conv1");
    left_resblock7_conv1->setStride( DimsHW {1, 1});
    left_resblock7_conv1->setPadding(DimsHW {1, 1});

    // left_resblock7_conv1_act ELU activation op.
    auto left_resblock7_conv1_act = addElu(plugin_factory, *network, *left_resblock7_conv1->getOutput(0), data_type, "left_resblock7_conv1_act");
    assert(left_resblock7_conv1_act != nullptr);
    left_resblock7_conv1_act->setName("left_resblock7_conv1_act");

    // left_resblock7_conv2 convolution op.
    auto left_resblock7_conv2 = network->addConvolution(*left_resblock7_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock7_conv2_k"), weights.at("left_resblock7_conv2_b"));
    assert(left_resblock7_conv2 != nullptr);
    left_resblock7_conv2->setName("left_resblock7_conv2");
    left_resblock7_conv2->setStride( DimsHW {1, 1});
    left_resblock7_conv2->setPadding(DimsHW {1, 1});

    // left_resblock7_conv2_add tensor add op.
    auto left_resblock7_conv2_add = network->addElementWise(*(left_resblock7_conv2->getOutput(0)), *(left_resblock6_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock7_conv2_add != nullptr);
    left_resblock7_conv2_add->setName("left_resblock7_conv2_add");

    // left_resblock7_conv2_add_act ELU activation op.
    auto left_resblock7_conv2_add_act = addElu(plugin_factory, *network, *left_resblock7_conv2_add->getOutput(0), data_type, "left_resblock7_conv2_add_act");
    assert(left_resblock7_conv2_add_act != nullptr);
    left_resblock7_conv2_add_act->setName("left_resblock7_conv2_add_act");

    // right_resblock7_conv1 convolution op.
    auto right_resblock7_conv1 = network->addConvolution(*right_resblock6_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock7_conv1_k"), weights.at("right_resblock7_conv1_b"));
    assert(right_resblock7_conv1 != nullptr);
    right_resblock7_conv1->setName("right_resblock7_conv1");
    right_resblock7_conv1->setStride( DimsHW {1, 1});
    right_resblock7_conv1->setPadding(DimsHW {1, 1});

    // right_resblock7_conv1_act ELU activation op.
    auto right_resblock7_conv1_act = addElu(plugin_factory, *network, *right_resblock7_conv1->getOutput(0), data_type, "right_resblock7_conv1_act");
    assert(right_resblock7_conv1_act != nullptr);
    right_resblock7_conv1_act->setName("right_resblock7_conv1_act");

    // right_resblock7_conv2 convolution op.
    auto right_resblock7_conv2 = network->addConvolution(*right_resblock7_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock7_conv2_k"), weights.at("right_resblock7_conv2_b"));
    assert(right_resblock7_conv2 != nullptr);
    right_resblock7_conv2->setName("right_resblock7_conv2");
    right_resblock7_conv2->setStride( DimsHW {1, 1});
    right_resblock7_conv2->setPadding(DimsHW {1, 1});

    // right_resblock7_conv2_add tensor add op.
    auto right_resblock7_conv2_add = network->addElementWise(*(right_resblock7_conv2->getOutput(0)), *(right_resblock6_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock7_conv2_add != nullptr);
    right_resblock7_conv2_add->setName("right_resblock7_conv2_add");

    // right_resblock7_conv2_add_act ELU activation op.
    auto right_resblock7_conv2_add_act = addElu(plugin_factory, *network, *right_resblock7_conv2_add->getOutput(0), data_type, "right_resblock7_conv2_add_act");
    assert(right_resblock7_conv2_add_act != nullptr);
    right_resblock7_conv2_add_act->setName("right_resblock7_conv2_add_act");

    // left_resblock8_conv1 convolution op.
    auto left_resblock8_conv1 = network->addConvolution(*left_resblock7_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock8_conv1_k"), weights.at("left_resblock8_conv1_b"));
    assert(left_resblock8_conv1 != nullptr);
    left_resblock8_conv1->setName("left_resblock8_conv1");
    left_resblock8_conv1->setStride( DimsHW {1, 1});
    left_resblock8_conv1->setPadding(DimsHW {1, 1});

    // left_resblock8_conv1_act ELU activation op.
    auto left_resblock8_conv1_act = addElu(plugin_factory, *network, *left_resblock8_conv1->getOutput(0), data_type, "left_resblock8_conv1_act");
    assert(left_resblock8_conv1_act != nullptr);
    left_resblock8_conv1_act->setName("left_resblock8_conv1_act");

    // left_resblock8_conv2 convolution op.
    auto left_resblock8_conv2 = network->addConvolution(*left_resblock8_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_resblock8_conv2_k"), weights.at("left_resblock8_conv2_b"));
    assert(left_resblock8_conv2 != nullptr);
    left_resblock8_conv2->setName("left_resblock8_conv2");
    left_resblock8_conv2->setStride( DimsHW {1, 1});
    left_resblock8_conv2->setPadding(DimsHW {1, 1});

    // left_resblock8_conv2_add tensor add op.
    auto left_resblock8_conv2_add = network->addElementWise(*(left_resblock8_conv2->getOutput(0)), *(left_resblock7_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(left_resblock8_conv2_add != nullptr);
    left_resblock8_conv2_add->setName("left_resblock8_conv2_add");

    // left_resblock8_conv2_add_act ELU activation op.
    auto left_resblock8_conv2_add_act = addElu(plugin_factory, *network, *left_resblock8_conv2_add->getOutput(0), data_type, "left_resblock8_conv2_add_act");
    assert(left_resblock8_conv2_add_act != nullptr);
    left_resblock8_conv2_add_act->setName("left_resblock8_conv2_add_act");

    // right_resblock8_conv1 convolution op.
    auto right_resblock8_conv1 = network->addConvolution(*right_resblock7_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock8_conv1_k"), weights.at("right_resblock8_conv1_b"));
    assert(right_resblock8_conv1 != nullptr);
    right_resblock8_conv1->setName("right_resblock8_conv1");
    right_resblock8_conv1->setStride( DimsHW {1, 1});
    right_resblock8_conv1->setPadding(DimsHW {1, 1});

    // right_resblock8_conv1_act ELU activation op.
    auto right_resblock8_conv1_act = addElu(plugin_factory, *network, *right_resblock8_conv1->getOutput(0), data_type, "right_resblock8_conv1_act");
    assert(right_resblock8_conv1_act != nullptr);
    right_resblock8_conv1_act->setName("right_resblock8_conv1_act");

    // right_resblock8_conv2 convolution op.
    auto right_resblock8_conv2 = network->addConvolution(*right_resblock8_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_resblock8_conv2_k"), weights.at("right_resblock8_conv2_b"));
    assert(right_resblock8_conv2 != nullptr);
    right_resblock8_conv2->setName("right_resblock8_conv2");
    right_resblock8_conv2->setStride( DimsHW {1, 1});
    right_resblock8_conv2->setPadding(DimsHW {1, 1});

    // right_resblock8_conv2_add tensor add op.
    auto right_resblock8_conv2_add = network->addElementWise(*(right_resblock8_conv2->getOutput(0)), *(right_resblock7_conv2_add_act->getOutput(0)),
    ElementWiseOperation::kSUM);
    assert(right_resblock8_conv2_add != nullptr);
    right_resblock8_conv2_add->setName("right_resblock8_conv2_add");

    // right_resblock8_conv2_add_act ELU activation op.
    auto right_resblock8_conv2_add_act = addElu(plugin_factory, *network, *right_resblock8_conv2_add->getOutput(0), data_type, "right_resblock8_conv2_add_act");
    assert(right_resblock8_conv2_add_act != nullptr);
    right_resblock8_conv2_add_act->setName("right_resblock8_conv2_add_act");

    // left_encoder2D_out convolution op.
    auto left_encoder2D_out = network->addConvolution(*left_resblock8_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_encoder2D_out_k"), weights.at("left_encoder2D_out_b"));
    assert(left_encoder2D_out != nullptr);
    left_encoder2D_out->setName("left_encoder2D_out");
    left_encoder2D_out->setStride( DimsHW {1, 1});
    left_encoder2D_out->setPadding(DimsHW {1, 1});

    // right_encoder2D_out convolution op.
    auto right_encoder2D_out = network->addConvolution(*right_resblock8_conv2_add_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_encoder2D_out_k"), weights.at("right_encoder2D_out_b"));
    assert(right_encoder2D_out != nullptr);
    right_encoder2D_out->setName("right_encoder2D_out");
    right_encoder2D_out->setStride( DimsHW {1, 1});
    right_encoder2D_out->setPadding(DimsHW {1, 1});

    // cost_vol cost volume op.
    auto cost_vol = addCostVolume(plugin_factory, *network, *left_encoder2D_out->getOutput(0), *right_encoder2D_out->getOutput(0),
                             CostVolumeType::kDefault, 68, data_type, "cost_vol");
    assert(cost_vol != nullptr);
    cost_vol->setName("cost_vol");

    // conv3D_1a 3D convolution op.
    auto conv3D_1a = addConv3D(plugin_factory, *network, *cost_vol->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {32, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_1a_k"), weights.at("conv3D_1a_b"),
                         "conv3D_1a");
    assert(conv3D_1a != nullptr);
    conv3D_1a->setName("conv3D_1a");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_1a_tran = addTransform(plugin_factory, *network, *conv3D_1a->getOutput(0), {1, 0, 2, 3}, "conv3D_1a_tran_transform");
    assert(conv3D_1a_tran != nullptr);
    conv3D_1a_tran->setName("conv3D_1a_tran");

    // conv3D_1a_act ELU activation op.
    auto conv3D_1a_act = addElu(plugin_factory, *network, *conv3D_1a_tran->getOutput(0), data_type, "conv3D_1a_act");
    assert(conv3D_1a_act != nullptr);
    conv3D_1a_act->setName("conv3D_1a_act");

    // conv3D_1b 3D convolution op.
    auto conv3D_1b = addConv3D(plugin_factory, *network, *conv3D_1a_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {32, 3, 32, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_1b_k"), weights.at("conv3D_1b_b"),
                         "conv3D_1b");
    assert(conv3D_1b != nullptr);
    conv3D_1b->setName("conv3D_1b");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_1b_tran = addTransform(plugin_factory, *network, *conv3D_1b->getOutput(0), {1, 0, 2, 3}, "conv3D_1b_tran_transform");
    assert(conv3D_1b_tran != nullptr);
    conv3D_1b_tran->setName("conv3D_1b_tran");

    // conv3D_1b_act ELU activation op.
    auto conv3D_1b_act = addElu(plugin_factory, *network, *conv3D_1b_tran->getOutput(0), data_type, "conv3D_1b_act");
    assert(conv3D_1b_act != nullptr);
    conv3D_1b_act->setName("conv3D_1b_act");

    // conv3D_1ds_pad padding op.
    auto conv3D_1ds_pad = addPad(plugin_factory, *network, *conv3D_1b_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_1ds_pad");
    assert(conv3D_1ds_pad != nullptr);
    conv3D_1ds_pad->setName("conv3D_1ds_pad");

    // conv3D_1ds 3D convolution op.
    auto conv3D_1ds = addConv3D(plugin_factory, *network, *conv3D_1ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 32, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_1ds_k"), weights.at("conv3D_1ds_b"),
                         "conv3D_1ds");
    assert(conv3D_1ds != nullptr);
    conv3D_1ds->setName("conv3D_1ds");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_1ds_tran = addTransform(plugin_factory, *network, *conv3D_1ds->getOutput(0), {1, 0, 2, 3}, "conv3D_1ds_tran_transform");
    assert(conv3D_1ds_tran != nullptr);
    conv3D_1ds_tran->setName("conv3D_1ds_tran");

    // conv3D_1ds_act ELU activation op.
    auto conv3D_1ds_act = addElu(plugin_factory, *network, *conv3D_1ds_tran->getOutput(0), data_type, "conv3D_1ds_act");
    assert(conv3D_1ds_act != nullptr);
    conv3D_1ds_act->setName("conv3D_1ds_act");

    // conv3D_2a 3D convolution op.
    auto conv3D_2a = addConv3D(plugin_factory, *network, *conv3D_1ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_2a_k"), weights.at("conv3D_2a_b"),
                         "conv3D_2a");
    assert(conv3D_2a != nullptr);
    conv3D_2a->setName("conv3D_2a");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_2a_tran = addTransform(plugin_factory, *network, *conv3D_2a->getOutput(0), {1, 0, 2, 3}, "conv3D_2a_tran_transform");
    assert(conv3D_2a_tran != nullptr);
    conv3D_2a_tran->setName("conv3D_2a_tran");

    // conv3D_2a_act ELU activation op.
    auto conv3D_2a_act = addElu(plugin_factory, *network, *conv3D_2a_tran->getOutput(0), data_type, "conv3D_2a_act");
    assert(conv3D_2a_act != nullptr);
    conv3D_2a_act->setName("conv3D_2a_act");

    // conv3D_2b 3D convolution op.
    auto conv3D_2b = addConv3D(plugin_factory, *network, *conv3D_2a_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_2b_k"), weights.at("conv3D_2b_b"),
                         "conv3D_2b");
    assert(conv3D_2b != nullptr);
    conv3D_2b->setName("conv3D_2b");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_2b_tran = addTransform(plugin_factory, *network, *conv3D_2b->getOutput(0), {1, 0, 2, 3}, "conv3D_2b_tran_transform");
    assert(conv3D_2b_tran != nullptr);
    conv3D_2b_tran->setName("conv3D_2b_tran");

    // conv3D_2b_act ELU activation op.
    auto conv3D_2b_act = addElu(plugin_factory, *network, *conv3D_2b_tran->getOutput(0), data_type, "conv3D_2b_act");
    assert(conv3D_2b_act != nullptr);
    conv3D_2b_act->setName("conv3D_2b_act");

    // conv3D_2ds_pad padding op.
    auto conv3D_2ds_pad = addPad(plugin_factory, *network, *conv3D_2b_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_2ds_pad");
    assert(conv3D_2ds_pad != nullptr);
    conv3D_2ds_pad->setName("conv3D_2ds_pad");

    // conv3D_2ds 3D convolution op.
    auto conv3D_2ds = addConv3D(plugin_factory, *network, *conv3D_2ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_2ds_k"), weights.at("conv3D_2ds_b"),
                         "conv3D_2ds");
    assert(conv3D_2ds != nullptr);
    conv3D_2ds->setName("conv3D_2ds");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_2ds_tran = addTransform(plugin_factory, *network, *conv3D_2ds->getOutput(0), {1, 0, 2, 3}, "conv3D_2ds_tran_transform");
    assert(conv3D_2ds_tran != nullptr);
    conv3D_2ds_tran->setName("conv3D_2ds_tran");

    // conv3D_2ds_act ELU activation op.
    auto conv3D_2ds_act = addElu(plugin_factory, *network, *conv3D_2ds_tran->getOutput(0), data_type, "conv3D_2ds_act");
    assert(conv3D_2ds_act != nullptr);
    conv3D_2ds_act->setName("conv3D_2ds_act");

    // conv3D_3a 3D convolution op.
    auto conv3D_3a = addConv3D(plugin_factory, *network, *conv3D_2ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_3a_k"), weights.at("conv3D_3a_b"),
                         "conv3D_3a");
    assert(conv3D_3a != nullptr);
    conv3D_3a->setName("conv3D_3a");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_3a_tran = addTransform(plugin_factory, *network, *conv3D_3a->getOutput(0), {1, 0, 2, 3}, "conv3D_3a_tran_transform");
    assert(conv3D_3a_tran != nullptr);
    conv3D_3a_tran->setName("conv3D_3a_tran");

    // conv3D_3a_act ELU activation op.
    auto conv3D_3a_act = addElu(plugin_factory, *network, *conv3D_3a_tran->getOutput(0), data_type, "conv3D_3a_act");
    assert(conv3D_3a_act != nullptr);
    conv3D_3a_act->setName("conv3D_3a_act");

    // conv3D_3b 3D convolution op.
    auto conv3D_3b = addConv3D(plugin_factory, *network, *conv3D_3a_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_3b_k"), weights.at("conv3D_3b_b"),
                         "conv3D_3b");
    assert(conv3D_3b != nullptr);
    conv3D_3b->setName("conv3D_3b");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_3b_tran = addTransform(plugin_factory, *network, *conv3D_3b->getOutput(0), {1, 0, 2, 3}, "conv3D_3b_tran_transform");
    assert(conv3D_3b_tran != nullptr);
    conv3D_3b_tran->setName("conv3D_3b_tran");

    // conv3D_3b_act ELU activation op.
    auto conv3D_3b_act = addElu(plugin_factory, *network, *conv3D_3b_tran->getOutput(0), data_type, "conv3D_3b_act");
    assert(conv3D_3b_act != nullptr);
    conv3D_3b_act->setName("conv3D_3b_act");

    // conv3D_3ds_pad padding op.
    auto conv3D_3ds_pad = addPad(plugin_factory, *network, *conv3D_3b_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_3ds_pad");
    assert(conv3D_3ds_pad != nullptr);
    conv3D_3ds_pad->setName("conv3D_3ds_pad");

    // conv3D_3ds 3D convolution op.
    auto conv3D_3ds = addConv3D(plugin_factory, *network, *conv3D_3ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_3ds_k"), weights.at("conv3D_3ds_b"),
                         "conv3D_3ds");
    assert(conv3D_3ds != nullptr);
    conv3D_3ds->setName("conv3D_3ds");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_3ds_tran = addTransform(plugin_factory, *network, *conv3D_3ds->getOutput(0), {1, 0, 2, 3}, "conv3D_3ds_tran_transform");
    assert(conv3D_3ds_tran != nullptr);
    conv3D_3ds_tran->setName("conv3D_3ds_tran");

    // conv3D_3ds_act ELU activation op.
    auto conv3D_3ds_act = addElu(plugin_factory, *network, *conv3D_3ds_tran->getOutput(0), data_type, "conv3D_3ds_act");
    assert(conv3D_3ds_act != nullptr);
    conv3D_3ds_act->setName("conv3D_3ds_act");

    // conv3D_4a 3D convolution op.
    auto conv3D_4a = addConv3D(plugin_factory, *network, *conv3D_3ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_4a_k"), weights.at("conv3D_4a_b"),
                         "conv3D_4a");
    assert(conv3D_4a != nullptr);
    conv3D_4a->setName("conv3D_4a");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_4a_tran = addTransform(plugin_factory, *network, *conv3D_4a->getOutput(0), {1, 0, 2, 3}, "conv3D_4a_tran_transform");
    assert(conv3D_4a_tran != nullptr);
    conv3D_4a_tran->setName("conv3D_4a_tran");

    // conv3D_4a_act ELU activation op.
    auto conv3D_4a_act = addElu(plugin_factory, *network, *conv3D_4a_tran->getOutput(0), data_type, "conv3D_4a_act");
    assert(conv3D_4a_act != nullptr);
    conv3D_4a_act->setName("conv3D_4a_act");

    // conv3D_4b 3D convolution op.
    auto conv3D_4b = addConv3D(plugin_factory, *network, *conv3D_4a_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_4b_k"), weights.at("conv3D_4b_b"),
                         "conv3D_4b");
    assert(conv3D_4b != nullptr);
    conv3D_4b->setName("conv3D_4b");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_4b_tran = addTransform(plugin_factory, *network, *conv3D_4b->getOutput(0), {1, 0, 2, 3}, "conv3D_4b_tran_transform");
    assert(conv3D_4b_tran != nullptr);
    conv3D_4b_tran->setName("conv3D_4b_tran");

    // conv3D_4b_act ELU activation op.
    auto conv3D_4b_act = addElu(plugin_factory, *network, *conv3D_4b_tran->getOutput(0), data_type, "conv3D_4b_act");
    assert(conv3D_4b_act != nullptr);
    conv3D_4b_act->setName("conv3D_4b_act");

    // conv3D_4ds_pad padding op.
    auto conv3D_4ds_pad = addPad(plugin_factory, *network, *conv3D_4b_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_4ds_pad");
    assert(conv3D_4ds_pad != nullptr);
    conv3D_4ds_pad->setName("conv3D_4ds_pad");

    // conv3D_4ds 3D convolution op.
    auto conv3D_4ds = addConv3D(plugin_factory, *network, *conv3D_4ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 64, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_4ds_k"), weights.at("conv3D_4ds_b"),
                         "conv3D_4ds");
    assert(conv3D_4ds != nullptr);
    conv3D_4ds->setName("conv3D_4ds");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_4ds_tran = addTransform(plugin_factory, *network, *conv3D_4ds->getOutput(0), {1, 0, 2, 3}, "conv3D_4ds_tran_transform");
    assert(conv3D_4ds_tran != nullptr);
    conv3D_4ds_tran->setName("conv3D_4ds_tran");

    // conv3D_4ds_act ELU activation op.
    auto conv3D_4ds_act = addElu(plugin_factory, *network, *conv3D_4ds_tran->getOutput(0), data_type, "conv3D_4ds_act");
    assert(conv3D_4ds_act != nullptr);
    conv3D_4ds_act->setName("conv3D_4ds_act");

    // conv3D_5a 3D convolution op.
    auto conv3D_5a = addConv3D(plugin_factory, *network, *conv3D_4ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 128, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_5a_k"), weights.at("conv3D_5a_b"),
                         "conv3D_5a");
    assert(conv3D_5a != nullptr);
    conv3D_5a->setName("conv3D_5a");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_5a_tran = addTransform(plugin_factory, *network, *conv3D_5a->getOutput(0), {1, 0, 2, 3}, "conv3D_5a_tran_transform");
    assert(conv3D_5a_tran != nullptr);
    conv3D_5a_tran->setName("conv3D_5a_tran");

    // conv3D_5a_act ELU activation op.
    auto conv3D_5a_act = addElu(plugin_factory, *network, *conv3D_5a_tran->getOutput(0), data_type, "conv3D_5a_act");
    assert(conv3D_5a_act != nullptr);
    conv3D_5a_act->setName("conv3D_5a_act");

    // conv3D_5b 3D convolution op.
    auto conv3D_5b = addConv3D(plugin_factory, *network, *conv3D_5a_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 128, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_5b_k"), weights.at("conv3D_5b_b"),
                         "conv3D_5b");
    assert(conv3D_5b != nullptr);
    conv3D_5b->setName("conv3D_5b");

    // conv3D_5b_act ELU activation op.
    auto conv3D_5b_act = addElu(plugin_factory, *network, *conv3D_5b->getOutput(0), data_type, "conv3D_5b_act");
    assert(conv3D_5b_act != nullptr);
    conv3D_5b_act->setName("conv3D_5b_act");

    // deconv3D_1 3D transposed convolution op.
    Dims deconv3D_1_out_dims{4, {9, 64, 21, 65}};
    auto deconv3D_1 = addConv3DTranspose(plugin_factory, *network, *conv3D_5b_act->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {128, 3, 64, 3, 3}}, deconv3D_1_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                                  weights.at("deconv3D_1_k"), weights.at("deconv3D_1_b"),
                                  "deconv3D_1");
    assert(deconv3D_1 != nullptr);
    deconv3D_1->setName("deconv3D_1");

    // deconv3D_1_add_skip tensor add op.
    auto deconv3D_1_add_skip = network->addElementWise(*(deconv3D_1->getOutput(0)), *(conv3D_4b_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv3D_1_add_skip != nullptr);
    deconv3D_1_add_skip->setName("deconv3D_1_add_skip");

    // deconv3D_1_act ELU activation op.
    auto deconv3D_1_act = addElu(plugin_factory, *network, *deconv3D_1_add_skip->getOutput(0), data_type, "deconv3D_1_act");
    assert(deconv3D_1_act != nullptr);
    deconv3D_1_act->setName("deconv3D_1_act");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto deconv3D_1_transform = addTransform(plugin_factory, *network, *deconv3D_1_act->getOutput(0), {1, 0, 2, 3}, "deconv3D_1_transform_transform");
    assert(deconv3D_1_transform != nullptr);
    deconv3D_1_transform->setName("deconv3D_1_transform");

    // deconv3D_2 3D transposed convolution op.
    Dims deconv3D_2_out_dims{4, {17, 64, 41, 129}};
    auto deconv3D_2 = addConv3DTranspose(plugin_factory, *network, *deconv3D_1_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}}, deconv3D_2_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                                  weights.at("deconv3D_2_k"), weights.at("deconv3D_2_b"),
                                  "deconv3D_2");
    assert(deconv3D_2 != nullptr);
    deconv3D_2->setName("deconv3D_2");

    // deconv3D_2_add_skip tensor add op.
    auto deconv3D_2_add_skip = network->addElementWise(*(deconv3D_2->getOutput(0)), *(conv3D_3b_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv3D_2_add_skip != nullptr);
    deconv3D_2_add_skip->setName("deconv3D_2_add_skip");

    // deconv3D_2_act ELU activation op.
    auto deconv3D_2_act = addElu(plugin_factory, *network, *deconv3D_2_add_skip->getOutput(0), data_type, "deconv3D_2_act");
    assert(deconv3D_2_act != nullptr);
    deconv3D_2_act->setName("deconv3D_2_act");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto deconv3D_2_transform = addTransform(plugin_factory, *network, *deconv3D_2_act->getOutput(0), {1, 0, 2, 3}, "deconv3D_2_transform_transform");
    assert(deconv3D_2_transform != nullptr);
    deconv3D_2_transform->setName("deconv3D_2_transform");

    // deconv3D_3 3D transposed convolution op.
    Dims deconv3D_3_out_dims{4, {35, 64, 81, 257}};
    auto deconv3D_3 = addConv3DTranspose(plugin_factory, *network, *deconv3D_2_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}}, deconv3D_3_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {0, 1, 1}},
                                  weights.at("deconv3D_3_k"), weights.at("deconv3D_3_b"),
                                  "deconv3D_3");
    assert(deconv3D_3 != nullptr);
    deconv3D_3->setName("deconv3D_3");

    // deconv3D_3 output slice op.
    auto deconv3D_3_slice_layer = addSlice(plugin_factory, *network, *deconv3D_3->getOutput(0),
                                    deconv3D_3_out_dims,
                                    {4, {0, 0, 0, 0}},
                                    {4, {deconv3D_3_out_dims.d[0] - 1, deconv3D_3_out_dims.d[1], deconv3D_3_out_dims.d[2], deconv3D_3_out_dims.d[3]}},
                                    "deconv3D_3_slice");
    assert(deconv3D_3_slice_layer != nullptr);
    deconv3D_3_slice_layer->setName("deconv3D_3_slice_layer");

    // deconv3D_3_add_skip tensor add op.
    auto deconv3D_3_add_skip = network->addElementWise(*(deconv3D_3_slice_layer->getOutput(0)), *(conv3D_2b_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv3D_3_add_skip != nullptr);
    deconv3D_3_add_skip->setName("deconv3D_3_add_skip");

    // deconv3D_3_act ELU activation op.
    auto deconv3D_3_act = addElu(plugin_factory, *network, *deconv3D_3_add_skip->getOutput(0), data_type, "deconv3D_3_act");
    assert(deconv3D_3_act != nullptr);
    deconv3D_3_act->setName("deconv3D_3_act");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto deconv3D_3_transform = addTransform(plugin_factory, *network, *deconv3D_3_act->getOutput(0), {1, 0, 2, 3}, "deconv3D_3_transform_transform");
    assert(deconv3D_3_transform != nullptr);
    deconv3D_3_transform->setName("deconv3D_3_transform");

    // deconv3D_4 3D transposed convolution op.
    Dims deconv3D_4_out_dims{4, {69, 32, 161, 513}};
    auto deconv3D_4 = addConv3DTranspose(plugin_factory, *network, *deconv3D_3_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {64, 3, 32, 3, 3}}, deconv3D_4_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {0, 1, 1}},
                                  weights.at("deconv3D_4_k"), weights.at("deconv3D_4_b"),
                                  "deconv3D_4");
    assert(deconv3D_4 != nullptr);
    deconv3D_4->setName("deconv3D_4");

    // deconv3D_4 output slice op.
    auto deconv3D_4_slice_layer = addSlice(plugin_factory, *network, *deconv3D_4->getOutput(0),
                                    deconv3D_4_out_dims,
                                    {4, {0, 0, 0, 0}},
                                    {4, {deconv3D_4_out_dims.d[0] - 1, deconv3D_4_out_dims.d[1], deconv3D_4_out_dims.d[2], deconv3D_4_out_dims.d[3]}},
                                    "deconv3D_4_slice");
    assert(deconv3D_4_slice_layer != nullptr);
    deconv3D_4_slice_layer->setName("deconv3D_4_slice_layer");

    // deconv3D_4_add_skip tensor add op.
    auto deconv3D_4_add_skip = network->addElementWise(*(deconv3D_4_slice_layer->getOutput(0)), *(conv3D_1b_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv3D_4_add_skip != nullptr);
    deconv3D_4_add_skip->setName("deconv3D_4_add_skip");

    // deconv3D_4_act ELU activation op.
    auto deconv3D_4_act = addElu(plugin_factory, *network, *deconv3D_4_add_skip->getOutput(0), data_type, "deconv3D_4_act");
    assert(deconv3D_4_act != nullptr);
    deconv3D_4_act->setName("deconv3D_4_act");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto deconv3D_4_transform = addTransform(plugin_factory, *network, *deconv3D_4_act->getOutput(0), {1, 0, 2, 3}, "deconv3D_4_transform_transform");
    assert(deconv3D_4_transform != nullptr);
    deconv3D_4_transform->setName("deconv3D_4_transform");

    // deconv3D_5 3D transposed convolution op.
    Dims deconv3D_5_out_dims{4, {137, 1, 321, 1025}};
    auto deconv3D_5 = addConv3DTranspose(plugin_factory, *network, *deconv3D_4_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {32, 3, 1, 3, 3}}, deconv3D_5_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {0, 1, 1}},
                                  weights.at("deconv3D_5_k"), weights.at("deconv3D_5_b"),
                                  "deconv3D_5");
    assert(deconv3D_5 != nullptr);
    deconv3D_5->setName("deconv3D_5");

    // deconv3D_5 output slice op.
    auto deconv3D_5_slice_layer = addSlice(plugin_factory, *network, *deconv3D_5->getOutput(0),
                                    deconv3D_5_out_dims,
                                    {4, {0, 0, 0, 0}},
                                    {4, {deconv3D_5_out_dims.d[0] - 1, deconv3D_5_out_dims.d[1], deconv3D_5_out_dims.d[2], deconv3D_5_out_dims.d[3]}},
                                    "deconv3D_5_slice");
    assert(deconv3D_5_slice_layer != nullptr);
    deconv3D_5_slice_layer->setName("deconv3D_5_slice_layer");

    // Softargmax.
    auto disp = addSoftargmax(plugin_factory, *network, *deconv3D_5_slice_layer->getOutput(0), SoftargmaxType::kMin, data_type, "disp_softargmax");
    assert(disp != nullptr);
    disp->setName("disp");

    auto disp_out = disp->getOutput(0);
    disp_out->setName("disp");
    network->markOutput(*disp_out);

    return network;
}

} } // namespace
