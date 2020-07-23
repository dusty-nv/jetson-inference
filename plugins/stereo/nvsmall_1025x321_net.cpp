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

INetworkDefinition* createNVSmall1025x321Network(IBuilder& builder, IPluginContainer& plugin_factory,
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

    // left_conv2 convolution op.
    auto left_conv2 = network->addConvolution(*left_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_conv2_k"), weights.at("left_conv2_b"));
    assert(left_conv2 != nullptr);
    left_conv2->setName("left_conv2");
    left_conv2->setStride( DimsHW {1, 1});
    left_conv2->setPadding(DimsHW {1, 1});

    // left_conv2_act ELU activation op.
    auto left_conv2_act = addElu(plugin_factory, *network, *left_conv2->getOutput(0), data_type, "left_conv2_act");
    assert(left_conv2_act != nullptr);
    left_conv2_act->setName("left_conv2_act");

    // right_conv2 convolution op.
    auto right_conv2 = network->addConvolution(*right_conv1_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_conv2_k"), weights.at("right_conv2_b"));
    assert(right_conv2 != nullptr);
    right_conv2->setName("right_conv2");
    right_conv2->setStride( DimsHW {1, 1});
    right_conv2->setPadding(DimsHW {1, 1});

    // right_conv2_act ELU activation op.
    auto right_conv2_act = addElu(plugin_factory, *network, *right_conv2->getOutput(0), data_type, "right_conv2_act");
    assert(right_conv2_act != nullptr);
    right_conv2_act->setName("right_conv2_act");

    // left_conv3 convolution op.
    auto left_conv3 = network->addConvolution(*left_conv2_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_conv3_k"), weights.at("left_conv3_b"));
    assert(left_conv3 != nullptr);
    left_conv3->setName("left_conv3");
    left_conv3->setStride( DimsHW {1, 1});
    left_conv3->setPadding(DimsHW {1, 1});

    // left_conv3_act ELU activation op.
    auto left_conv3_act = addElu(plugin_factory, *network, *left_conv3->getOutput(0), data_type, "left_conv3_act");
    assert(left_conv3_act != nullptr);
    left_conv3_act->setName("left_conv3_act");

    // right_conv3 convolution op.
    auto right_conv3 = network->addConvolution(*right_conv2_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_conv3_k"), weights.at("right_conv3_b"));
    assert(right_conv3 != nullptr);
    right_conv3->setName("right_conv3");
    right_conv3->setStride( DimsHW {1, 1});
    right_conv3->setPadding(DimsHW {1, 1});

    // right_conv3_act ELU activation op.
    auto right_conv3_act = addElu(plugin_factory, *network, *right_conv3->getOutput(0), data_type, "right_conv3_act");
    assert(right_conv3_act != nullptr);
    right_conv3_act->setName("right_conv3_act");

    // left_conv4 convolution op.
    auto left_conv4 = network->addConvolution(*left_conv3_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_conv4_k"), weights.at("left_conv4_b"));
    assert(left_conv4 != nullptr);
    left_conv4->setName("left_conv4");
    left_conv4->setStride( DimsHW {1, 1});
    left_conv4->setPadding(DimsHW {1, 1});

    // left_conv4_act ELU activation op.
    auto left_conv4_act = addElu(plugin_factory, *network, *left_conv4->getOutput(0), data_type, "left_conv4_act");
    assert(left_conv4_act != nullptr);
    left_conv4_act->setName("left_conv4_act");

    // right_conv4 convolution op.
    auto right_conv4 = network->addConvolution(*right_conv3_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_conv4_k"), weights.at("right_conv4_b"));
    assert(right_conv4 != nullptr);
    right_conv4->setName("right_conv4");
    right_conv4->setStride( DimsHW {1, 1});
    right_conv4->setPadding(DimsHW {1, 1});

    // right_conv4_act ELU activation op.
    auto right_conv4_act = addElu(plugin_factory, *network, *right_conv4->getOutput(0), data_type, "right_conv4_act");
    assert(right_conv4_act != nullptr);
    right_conv4_act->setName("right_conv4_act");

    // left_conv5 convolution op.
    auto left_conv5 = network->addConvolution(*left_conv4_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("left_conv5_k"), weights.at("left_conv5_b"));
    assert(left_conv5 != nullptr);
    left_conv5->setName("left_conv5");
    left_conv5->setStride( DimsHW {1, 1});
    left_conv5->setPadding(DimsHW {1, 1});

    // right_conv5 convolution op.
    auto right_conv5 = network->addConvolution(*right_conv4_act->getOutput(0), 32, DimsHW {3, 3},
                                       weights.at("right_conv5_k"), weights.at("right_conv5_b"));
    assert(right_conv5 != nullptr);
    right_conv5->setName("right_conv5");
    right_conv5->setStride( DimsHW {1, 1});
    right_conv5->setPadding(DimsHW {1, 1});

    // cost_vol cost volume op.
    auto cost_vol = addCostVolume(plugin_factory, *network, *left_conv5->getOutput(0), *right_conv5->getOutput(0),
                             CostVolumeType::kDefault, 48, data_type, "cost_vol");
    assert(cost_vol != nullptr);
    cost_vol->setName("cost_vol");

    // conv3D_1 3D convolution op.
    auto conv3D_1 = addConv3D(plugin_factory, *network, *cost_vol->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {32, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_1_k"), weights.at("conv3D_1_b"),
                         "conv3D_1");
    assert(conv3D_1 != nullptr);
    conv3D_1->setName("conv3D_1");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_1_tran = addTransform(plugin_factory, *network, *conv3D_1->getOutput(0), {1, 0, 2, 3}, "conv3D_1_tran_transform");
    assert(conv3D_1_tran != nullptr);
    conv3D_1_tran->setName("conv3D_1_tran");

    // conv3D_1_act ELU activation op.
    auto conv3D_1_act = addElu(plugin_factory, *network, *conv3D_1_tran->getOutput(0), data_type, "conv3D_1_act");
    assert(conv3D_1_act != nullptr);
    conv3D_1_act->setName("conv3D_1_act");

    // conv3D_2 3D convolution op.
    auto conv3D_2 = addConv3D(plugin_factory, *network, *conv3D_1_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {32, 3, 32, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_2_k"), weights.at("conv3D_2_b"),
                         "conv3D_2");
    assert(conv3D_2 != nullptr);
    conv3D_2->setName("conv3D_2");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_2_tran = addTransform(plugin_factory, *network, *conv3D_2->getOutput(0), {1, 0, 2, 3}, "conv3D_2_tran_transform");
    assert(conv3D_2_tran != nullptr);
    conv3D_2_tran->setName("conv3D_2_tran");

    // conv3D_2_act ELU activation op.
    auto conv3D_2_act = addElu(plugin_factory, *network, *conv3D_2_tran->getOutput(0), data_type, "conv3D_2_act");
    assert(conv3D_2_act != nullptr);
    conv3D_2_act->setName("conv3D_2_act");

    // conv3D_3ds_pad padding op.
    auto conv3D_3ds_pad = addPad(plugin_factory, *network, *conv3D_2_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_3ds_pad");
    assert(conv3D_3ds_pad != nullptr);
    conv3D_3ds_pad->setName("conv3D_3ds_pad");

    // conv3D_3ds 3D convolution op.
    auto conv3D_3ds = addConv3D(plugin_factory, *network, *conv3D_3ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 32, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {1, 1, 1}},
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

    // conv3D_4 3D convolution op.
    auto conv3D_4 = addConv3D(plugin_factory, *network, *conv3D_3ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_4_k"), weights.at("conv3D_4_b"),
                         "conv3D_4");
    assert(conv3D_4 != nullptr);
    conv3D_4->setName("conv3D_4");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_4_tran = addTransform(plugin_factory, *network, *conv3D_4->getOutput(0), {1, 0, 2, 3}, "conv3D_4_tran_transform");
    assert(conv3D_4_tran != nullptr);
    conv3D_4_tran->setName("conv3D_4_tran");

    // conv3D_4_act ELU activation op.
    auto conv3D_4_act = addElu(plugin_factory, *network, *conv3D_4_tran->getOutput(0), data_type, "conv3D_4_act");
    assert(conv3D_4_act != nullptr);
    conv3D_4_act->setName("conv3D_4_act");

    // conv3D_5 3D convolution op.
    auto conv3D_5 = addConv3D(plugin_factory, *network, *conv3D_4_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {64, 3, 64, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_5_k"), weights.at("conv3D_5_b"),
                         "conv3D_5");
    assert(conv3D_5 != nullptr);
    conv3D_5->setName("conv3D_5");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_5_tran = addTransform(plugin_factory, *network, *conv3D_5->getOutput(0), {1, 0, 2, 3}, "conv3D_5_tran_transform");
    assert(conv3D_5_tran != nullptr);
    conv3D_5_tran->setName("conv3D_5_tran");

    // conv3D_5_act ELU activation op.
    auto conv3D_5_act = addElu(plugin_factory, *network, *conv3D_5_tran->getOutput(0), data_type, "conv3D_5_act");
    assert(conv3D_5_act != nullptr);
    conv3D_5_act->setName("conv3D_5_act");

    // conv3D_6ds_pad padding op.
    auto conv3D_6ds_pad = addPad(plugin_factory, *network, *conv3D_5_act->getOutput(0), {0, 0, 0, 0}, {1, 0, 0, 0}, "conv3D_6ds_pad");
    assert(conv3D_6ds_pad != nullptr);
    conv3D_6ds_pad->setName("conv3D_6ds_pad");

    // conv3D_6ds 3D convolution op.
    auto conv3D_6ds = addConv3D(plugin_factory, *network, *conv3D_6ds_pad->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 64, 3, 3}},
                         Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_6ds_k"), weights.at("conv3D_6ds_b"),
                         "conv3D_6ds");
    assert(conv3D_6ds != nullptr);
    conv3D_6ds->setName("conv3D_6ds");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_6ds_tran = addTransform(plugin_factory, *network, *conv3D_6ds->getOutput(0), {1, 0, 2, 3}, "conv3D_6ds_tran_transform");
    assert(conv3D_6ds_tran != nullptr);
    conv3D_6ds_tran->setName("conv3D_6ds_tran");

    // conv3D_6ds_act ELU activation op.
    auto conv3D_6ds_act = addElu(plugin_factory, *network, *conv3D_6ds_tran->getOutput(0), data_type, "conv3D_6ds_act");
    assert(conv3D_6ds_act != nullptr);
    conv3D_6ds_act->setName("conv3D_6ds_act");

    // conv3D_7 3D convolution op.
    auto conv3D_7 = addConv3D(plugin_factory, *network, *conv3D_6ds_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 128, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_7_k"), weights.at("conv3D_7_b"),
                         "conv3D_7");
    assert(conv3D_7 != nullptr);
    conv3D_7->setName("conv3D_7");

    // Transpose output: KDHW -> DKHW for conv3d and DKHW -> KDHW for conv3d_transpose
    auto conv3D_7_tran = addTransform(plugin_factory, *network, *conv3D_7->getOutput(0), {1, 0, 2, 3}, "conv3D_7_tran_transform");
    assert(conv3D_7_tran != nullptr);
    conv3D_7_tran->setName("conv3D_7_tran");

    // conv3D_7_act ELU activation op.
    auto conv3D_7_act = addElu(plugin_factory, *network, *conv3D_7_tran->getOutput(0), data_type, "conv3D_7_act");
    assert(conv3D_7_act != nullptr);
    conv3D_7_act->setName("conv3D_7_act");

    // conv3D_8 3D convolution op.
    auto conv3D_8 = addConv3D(plugin_factory, *network, *conv3D_7_act->getOutput(0),
                         Conv3DType::kTensorFlow, {5, {128, 3, 128, 3, 3}},
                         Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}}, Dims{3, {1, 1, 1}},
                         weights.at("conv3D_8_k"), weights.at("conv3D_8_b"),
                         "conv3D_8");
    assert(conv3D_8 != nullptr);
    conv3D_8->setName("conv3D_8");

    // conv3D_8_act ELU activation op.
    auto conv3D_8_act = addElu(plugin_factory, *network, *conv3D_8->getOutput(0), data_type, "conv3D_8_act");
    assert(conv3D_8_act != nullptr);
    conv3D_8_act->setName("conv3D_8_act");

    // deconv3D_1 3D transposed convolution op.
    Dims deconv3D_1_out_dims{4, {25, 64, 81, 257}};
    auto deconv3D_1 = addConv3DTranspose(plugin_factory, *network, *conv3D_8_act->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {128, 3, 64, 3, 3}}, deconv3D_1_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {0, 1, 1}},
                                  weights.at("deconv3D_1_k"), weights.at("deconv3D_1_b"),
                                  "deconv3D_1");
    assert(deconv3D_1 != nullptr);
    deconv3D_1->setName("deconv3D_1");

    // deconv3D_1 output slice op.
    auto deconv3D_1_slice_layer = addSlice(plugin_factory, *network, *deconv3D_1->getOutput(0),
                                    deconv3D_1_out_dims,
                                    {4, {0, 0, 0, 0}},
                                    {4, {deconv3D_1_out_dims.d[0] - 1, deconv3D_1_out_dims.d[1], deconv3D_1_out_dims.d[2], deconv3D_1_out_dims.d[3]}},
                                    "deconv3D_1_slice");
    assert(deconv3D_1_slice_layer != nullptr);
    deconv3D_1_slice_layer->setName("deconv3D_1_slice_layer");

    // deconv3D_1_add_skip tensor add op.
    auto deconv3D_1_add_skip = network->addElementWise(*(deconv3D_1_slice_layer->getOutput(0)), *(conv3D_5_act->getOutput(0)), ElementWiseOperation::kSUM);
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
    Dims deconv3D_2_out_dims{4, {49, 32, 161, 513}};
    auto deconv3D_2 = addConv3DTranspose(plugin_factory, *network, *deconv3D_1_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {64, 3, 32, 3, 3}}, deconv3D_2_out_dims,
                                  Dims{3, {2, 2, 2}}, Dims{3, {0, 1, 1}}, Dims{3, {0, 1, 1}},
                                  weights.at("deconv3D_2_k"), weights.at("deconv3D_2_b"),
                                  "deconv3D_2");
    assert(deconv3D_2 != nullptr);
    deconv3D_2->setName("deconv3D_2");

    // deconv3D_2 output slice op.
    auto deconv3D_2_slice_layer = addSlice(plugin_factory, *network, *deconv3D_2->getOutput(0),
                                    deconv3D_2_out_dims,
                                    {4, {0, 0, 0, 0}},
                                    {4, {deconv3D_2_out_dims.d[0] - 1, deconv3D_2_out_dims.d[1], deconv3D_2_out_dims.d[2], deconv3D_2_out_dims.d[3]}},
                                    "deconv3D_2_slice");
    assert(deconv3D_2_slice_layer != nullptr);
    deconv3D_2_slice_layer->setName("deconv3D_2_slice_layer");

    // deconv3D_2_add_skip tensor add op.
    auto deconv3D_2_add_skip = network->addElementWise(*(deconv3D_2_slice_layer->getOutput(0)), *(conv3D_2_act->getOutput(0)), ElementWiseOperation::kSUM);
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
    Dims deconv3D_3_out_dims{4, {97, 1, 321, 1025}};
    auto deconv3D_3 = addConv3DTranspose(plugin_factory, *network, *deconv3D_2_transform->getOutput(0),
                                  Conv3DType::kTensorFlow, {5, {32, 3, 1, 3, 3}}, deconv3D_3_out_dims,
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

    // Softargmax.
    auto disp = addSoftargmax(plugin_factory, *network, *deconv3D_3_slice_layer->getOutput(0), SoftargmaxType::kMin, data_type, "disp_softargmax");
    assert(disp != nullptr);
    disp->setName("disp");

    auto disp_out = disp->getOutput(0);
    disp_out->setName("disp");
    network->markOutput(*disp_out);

    return network;
}

} } // namespace
