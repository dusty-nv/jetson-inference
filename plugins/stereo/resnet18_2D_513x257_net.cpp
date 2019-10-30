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

INetworkDefinition* createResNet18_2D_513x257Network(IBuilder& builder, IPluginContainer& plugin_factory,
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
                             CostVolumeType::kCorrelation, 48, data_type, "cost_vol");
    assert(cost_vol != nullptr);
    cost_vol->setName("cost_vol");

    // Softargmax.
    auto softargmax = addSoftargmax(plugin_factory, *network, *cost_vol->getOutput(0), SoftargmaxType::kMax, data_type, "softargmax_softargmax");
    assert(softargmax != nullptr);
    softargmax->setName("softargmax");

    // concat tensor concat op.
    ITensor* concat_inputs[] = {left_conv1_act->getOutput(0), softargmax->getOutput(0)};
    auto concat = network->addConcatenation(concat_inputs, 2);
    assert(concat != nullptr);
    concat->setName("concat");

    // conv2D_1 convolution op.
    auto conv2D_1 = network->addConvolution(*concat->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("conv2D_1_k"), weights.at("conv2D_1_b"));
    assert(conv2D_1 != nullptr);
    conv2D_1->setName("conv2D_1");
    conv2D_1->setStride( DimsHW {1, 1});
    conv2D_1->setPadding(DimsHW {1, 1});

    // conv2D_1_act ELU activation op.
    auto conv2D_1_act = addElu(plugin_factory, *network, *conv2D_1->getOutput(0), data_type, "conv2D_1_act");
    assert(conv2D_1_act != nullptr);
    conv2D_1_act->setName("conv2D_1_act");

    // conv2D_2 convolution op.
    auto conv2D_2 = network->addConvolution(*conv2D_1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("conv2D_2_k"), weights.at("conv2D_2_b"));
    assert(conv2D_2 != nullptr);
    conv2D_2->setName("conv2D_2");
    conv2D_2->setStride( DimsHW {1, 1});
    conv2D_2->setPadding(DimsHW {1, 1});

    // conv2D_2_act ELU activation op.
    auto conv2D_2_act = addElu(plugin_factory, *network, *conv2D_2->getOutput(0), data_type, "conv2D_2_act");
    assert(conv2D_2_act != nullptr);
    conv2D_2_act->setName("conv2D_2_act");

    // conv2D_3ds convolution op.
    auto conv2D_3ds = network->addConvolution(*conv2D_2_act->getOutput(0), 64, DimsHW {3, 3},
                                       weights.at("conv2D_3ds_k"), weights.at("conv2D_3ds_b"));
    assert(conv2D_3ds != nullptr);
    conv2D_3ds->setName("conv2D_3ds");
    conv2D_3ds->setStride( DimsHW {2, 2});
    conv2D_3ds->setPadding(DimsHW {1, 1});

    // conv2D_3ds_act ELU activation op.
    auto conv2D_3ds_act = addElu(plugin_factory, *network, *conv2D_3ds->getOutput(0), data_type, "conv2D_3ds_act");
    assert(conv2D_3ds_act != nullptr);
    conv2D_3ds_act->setName("conv2D_3ds_act");

    // conv2D_4 convolution op.
    auto conv2D_4 = network->addConvolution(*conv2D_3ds_act->getOutput(0), 64, DimsHW {3, 3},
                                       weights.at("conv2D_4_k"), weights.at("conv2D_4_b"));
    assert(conv2D_4 != nullptr);
    conv2D_4->setName("conv2D_4");
    conv2D_4->setStride( DimsHW {1, 1});
    conv2D_4->setPadding(DimsHW {1, 1});

    // conv2D_4_act ELU activation op.
    auto conv2D_4_act = addElu(plugin_factory, *network, *conv2D_4->getOutput(0), data_type, "conv2D_4_act");
    assert(conv2D_4_act != nullptr);
    conv2D_4_act->setName("conv2D_4_act");

    // conv2D_5 convolution op.
    auto conv2D_5 = network->addConvolution(*conv2D_4_act->getOutput(0), 64, DimsHW {3, 3},
                                       weights.at("conv2D_5_k"), weights.at("conv2D_5_b"));
    assert(conv2D_5 != nullptr);
    conv2D_5->setName("conv2D_5");
    conv2D_5->setStride( DimsHW {1, 1});
    conv2D_5->setPadding(DimsHW {1, 1});

    // conv2D_5_act ELU activation op.
    auto conv2D_5_act = addElu(plugin_factory, *network, *conv2D_5->getOutput(0), data_type, "conv2D_5_act");
    assert(conv2D_5_act != nullptr);
    conv2D_5_act->setName("conv2D_5_act");

    // conv2D_6ds convolution op.
    auto conv2D_6ds = network->addConvolution(*conv2D_5_act->getOutput(0), 128, DimsHW {3, 3},
                                       weights.at("conv2D_6ds_k"), weights.at("conv2D_6ds_b"));
    assert(conv2D_6ds != nullptr);
    conv2D_6ds->setName("conv2D_6ds");
    conv2D_6ds->setStride( DimsHW {2, 2});
    conv2D_6ds->setPadding(DimsHW {1, 1});

    // conv2D_6ds_act ELU activation op.
    auto conv2D_6ds_act = addElu(plugin_factory, *network, *conv2D_6ds->getOutput(0), data_type, "conv2D_6ds_act");
    assert(conv2D_6ds_act != nullptr);
    conv2D_6ds_act->setName("conv2D_6ds_act");

    // conv2D_7 convolution op.
    auto conv2D_7 = network->addConvolution(*conv2D_6ds_act->getOutput(0), 128, DimsHW {3, 3},
                                       weights.at("conv2D_7_k"), weights.at("conv2D_7_b"));
    assert(conv2D_7 != nullptr);
    conv2D_7->setName("conv2D_7");
    conv2D_7->setStride( DimsHW {1, 1});
    conv2D_7->setPadding(DimsHW {1, 1});

    // conv2D_7_act ELU activation op.
    auto conv2D_7_act = addElu(plugin_factory, *network, *conv2D_7->getOutput(0), data_type, "conv2D_7_act");
    assert(conv2D_7_act != nullptr);
    conv2D_7_act->setName("conv2D_7_act");

    // conv2D_8 convolution op.
    auto conv2D_8 = network->addConvolution(*conv2D_7_act->getOutput(0), 128, DimsHW {3, 3},
                                       weights.at("conv2D_8_k"), weights.at("conv2D_8_b"));
    assert(conv2D_8 != nullptr);
    conv2D_8->setName("conv2D_8");
    conv2D_8->setStride( DimsHW {1, 1});
    conv2D_8->setPadding(DimsHW {1, 1});

    // conv2D_8_act ELU activation op.
    auto conv2D_8_act = addElu(plugin_factory, *network, *conv2D_8->getOutput(0), data_type, "conv2D_8_act");
    assert(conv2D_8_act != nullptr);
    conv2D_8_act->setName("conv2D_8_act");

    // deconv2D_1 transposed convolution op.
    auto deconv2D_1 = network->addDeconvolution(*conv2D_8_act->getOutput(0), 64, DimsHW {3, 3},
                                         weights.at("deconv2D_1_k"), weights.at("deconv2D_1_b"));
    assert(deconv2D_1 != nullptr);
    deconv2D_1->setName("deconv2D_1");
    deconv2D_1->setStride( DimsHW {2, 2});
    deconv2D_1->setPadding(DimsHW {1, 1});

    // deconv2D_1_add_skip tensor add op.
    auto deconv2D_1_add_skip = network->addElementWise(*(deconv2D_1->getOutput(0)), *(conv2D_5_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv2D_1_add_skip != nullptr);
    deconv2D_1_add_skip->setName("deconv2D_1_add_skip");

    // deconv2D_1_act ELU activation op.
    auto deconv2D_1_act = addElu(plugin_factory, *network, *deconv2D_1_add_skip->getOutput(0), data_type, "deconv2D_1_act");
    assert(deconv2D_1_act != nullptr);
    deconv2D_1_act->setName("deconv2D_1_act");

    // deconv2D_2 transposed convolution op.
    auto deconv2D_2 = network->addDeconvolution(*deconv2D_1_act->getOutput(0), 32, DimsHW {3, 3},
                                         weights.at("deconv2D_2_k"), weights.at("deconv2D_2_b"));
    assert(deconv2D_2 != nullptr);
    deconv2D_2->setName("deconv2D_2");
    deconv2D_2->setStride( DimsHW {2, 2});
    deconv2D_2->setPadding(DimsHW {1, 1});

    // deconv2D_2_add_skip tensor add op.
    auto deconv2D_2_add_skip = network->addElementWise(*(deconv2D_2->getOutput(0)), *(conv2D_2_act->getOutput(0)), ElementWiseOperation::kSUM);
    assert(deconv2D_2_add_skip != nullptr);
    deconv2D_2_add_skip->setName("deconv2D_2_add_skip");

    // deconv2D_2_act ELU activation op.
    auto deconv2D_2_act = addElu(plugin_factory, *network, *deconv2D_2_add_skip->getOutput(0), data_type, "deconv2D_2_act");
    assert(deconv2D_2_act != nullptr);
    deconv2D_2_act->setName("deconv2D_2_act");

    // deconv2D_3 transposed convolution op.
    auto deconv2D_3 = network->addDeconvolution(*deconv2D_2_act->getOutput(0), 1, DimsHW {3, 3},
                                         weights.at("deconv2D_3_k"), weights.at("deconv2D_3_b"));
    assert(deconv2D_3 != nullptr);
    deconv2D_3->setName("deconv2D_3");
    deconv2D_3->setStride( DimsHW {2, 2});
    deconv2D_3->setPadding(DimsHW {1, 1});

    // disp sigmoid activation op.
    auto disp = network->addActivation(*deconv2D_3->getOutput(0), ActivationType::kSIGMOID);
    assert(disp != nullptr);
    disp->setName("disp");

    auto disp_out = disp->getOutput(0);
    disp_out->setName("disp");
    network->markOutput(*disp_out);

    return network;
}

} } // namespace
