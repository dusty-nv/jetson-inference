// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>
#include <numeric>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Tensor transform plugin.
// -----------------------------------------------------------------
class TransformPlugin: public IPlugin
{
public:
    TransformPlugin(Permutation permutation, ILogger& log, std::string name):
        permutation_(permutation), log_(log), name_(name)
    {
    }

    TransformPlugin(TransformPlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        // Only NCHW format is supported for now, TRT does not work (assert) with generic Dims.
        assert(inputs[0].nbDims == 4);

        in_dims_ = DimsNCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2], inputs[0].d[3]);
        // Sanity check.
        int actual_sum   = std::accumulate(permutation_.order, permutation_.order + in_dims_.nbDims, 0, std::plus<int>());
        int expected_sum = in_dims_.nbDims * (in_dims_.nbDims - 1) / 2;
        assert(actual_sum == expected_sum);
        UNUSEDR(actual_sum);
        UNUSEDR(expected_sum);
        // Transposed output dimensions.
        for (int i = 0; i < in_dims_.nbDims; i++)
            out_dims_.d[i] = in_dims_.d[permutation_.order[i]];
        return out_dims_;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
        assert(nbInputs  == 1);
        assert(nbOutputs == 1);
        assert(DimsUtils::areEqual(inputDims[0],  in_dims_));
        assert(DimsUtils::areEqual(outputDims[0], out_dims_));

        createDescriptors();
        setTensorDescriptors(maxBatchSize);

        log_.log(ILogger::Severity::kINFO, (name_ + ": InDims : " + DimsUtils::toString(in_dims_)).c_str());
        log_.log(ILogger::Severity::kINFO, (name_ + ": OutDims: " + DimsUtils::toString(out_dims_)).c_str());

        assert(isValid());
    }

    int initialize() override
    {
        assert(isValid());
        return 0;
    }

    void terminate() override
    {
        assert(isValid());

        if (in_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(in_desc_));
        if (out_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(out_desc_));
        if (cudnn_ != nullptr)
            CHECK(cudnnDestroy(cudnn_));
        in_desc_  = nullptr;
        out_desc_ = nullptr;
        cudnn_    = nullptr;

        assert(!isValid());
    }

    size_t getWorkspaceSize(int maxBatchSize) const
    {
        return 0;
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        assert(isValid());

        cudnnStatus_t status;

        CHECK(status = cudnnSetStream(cudnn_, stream));

        if (batchSize != tensor_dims_.d[0])
            setTensorDescriptors(batchSize);

        CHECK(status = cudnnTransformTensor(cudnn_, &Consts::kOne, in_desc_, inputs[0], &Consts::kZero, out_desc_, outputs[0]));

        return status == CUDNN_STATUS_SUCCESS ? 0 : -1;
    }

    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
    }

private:
    bool isValid() const
    {
        return cudnn_ != nullptr;
    }

    void createDescriptors()
    {
        if (cudnn_ == nullptr)
            CHECK(cudnnCreate(&cudnn_));
        if (in_desc_ == nullptr)
            CHECK(cudnnCreateTensorDescriptor(&in_desc_));
        if (out_desc_ == nullptr)
            CHECK(cudnnCreateTensorDescriptor(&out_desc_));
    }

    void setTensorDescriptors(int batch_size)
    {
        // cudnnTransformTensor requires dimensions of both tensors to be the same.
        // The transform is represented by different strides.

        // Add batch dimension to final tensor and copy dims from input.
        tensor_dims_.nbDims = in_dims_.nbDims + 1;
        tensor_dims_.d[0]   = batch_size;
        std::copy(in_dims_.d, in_dims_.d + in_dims_.nbDims, tensor_dims_.d + 1);

        in_tensor_strides_ = DimsUtils::getStrides(tensor_dims_);
        CHECK(cudnnSetTensorNdDescriptor(in_desc_,  CUDNN_DATA_FLOAT,
                                         tensor_dims_.nbDims, tensor_dims_.d, in_tensor_strides_.d));

        // Compute output tensor strides.
        // 1. Create output tensor dims with batch dimension.
        Dims out_tensor_dims;
        out_tensor_dims.nbDims = out_dims_.nbDims + 1;
        out_tensor_dims.d[0]   = batch_size;
        std::copy(out_dims_.d, out_dims_.d + out_dims_.nbDims, out_tensor_dims.d + 1);
        // 2. Compute the strides for the output tensor. We need a copy to do a transform later.
        Dims out_strides = DimsUtils::getStrides(out_tensor_dims);
        out_tensor_strides_ = out_strides;
        // 3. As cudnnTransformTensor requires both dims to be the same, we need
        // to change strides on the _output_ tensor to achieve desired transform effect.
        for (int i = 1; i < out_strides.nbDims; i++)
            out_tensor_strides_.d[i] = out_strides.d[permutation_.order[i - 1] + 1];
        CHECK(cudnnSetTensorNdDescriptor(out_desc_, CUDNN_DATA_FLOAT,
                                         tensor_dims_.nbDims, tensor_dims_.d, out_tensor_strides_.d));

    }

private:
    cudnnHandle_t           cudnn_    = nullptr;
    cudnnTensorDescriptor_t in_desc_ = nullptr;
    cudnnTensorDescriptor_t out_desc_ = nullptr;

    Permutation permutation_;
    DimsNCHW    in_dims_;
    DimsNCHW    out_dims_;

    Dims        tensor_dims_;
    Dims        in_tensor_strides_;
    Dims        out_tensor_strides_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createTransformPlugin(Permutation permutation, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new TransformPlugin(permutation, log_, name));
    return plugins_.back();
}

} }