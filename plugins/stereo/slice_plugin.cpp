// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Tensor slice plugin.
// REVIEW alexey: currently the only supported slicing is for the
// second outermost dimension of 5D tensor as this is all we need so far.
// -----------------------------------------------------------------
class SlicePlugin: public IPlugin
{
public:
    SlicePlugin(Dims dims, Dims slice_start, Dims slice_end, ILogger& log, std::string name):
        in_dims_(dims), slice_start_(slice_start), slice_end_(slice_end),
        log_(log), name_(name)
    {
        assert(in_dims_.nbDims == 4);
        assert(in_dims_.nbDims == slice_start_.nbDims);
        assert(in_dims_.nbDims == slice_end_.nbDims);
        // For now no negative indices or empty tensors either.
        assert(0 <= slice_start_.d[0] && slice_start_.d[0] < slice_end_.d[0]);
        assert(0 <  slice_end_.d[0]   && slice_end_.d[0]   <= in_dims_.d[0]);

        for (int i = 1; i < in_dims_.nbDims; i++)
        {
            assert(slice_start_.d[i] == 0);
            assert(slice_end_.d[i]   == in_dims_.d[i]);
        }
    }

    SlicePlugin(SlicePlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        // Only NCHW format is supported for now.
        assert(inputs[0].nbDims == 4);

        out_dims_ = DimsNCHW(slice_end_.d[0] - slice_start_.d[0], inputs[0].d[1], inputs[0].d[2], inputs[0].d[3]);
        return out_dims_;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
        assert(nbInputs  == 1);
        assert(nbOutputs == 1);
        assert(DimsUtils::areEqual(inputDims[0],  in_dims_));
        assert(DimsUtils::areEqual(outputDims[0], out_dims_));

        log_.log(ILogger::Severity::kINFO, (name_ + ": InDims : " + DimsUtils::toString(in_dims_)).c_str());
        log_.log(ILogger::Severity::kINFO, (name_ + ": OutDims: " + DimsUtils::toString(out_dims_)).c_str());
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int maxBatchSize) const
    {
        return 0;
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        size_t d_dim_size = DimsUtils::getTensorSize(in_dims_) / in_dims_.d[0];
        assert(d_dim_size * in_dims_.d[0] == DimsUtils::getTensorSize(in_dims_));

        auto   psrc = static_cast<const float*>(inputs[0]) + d_dim_size * slice_start_.d[0];
        size_t size = out_dims_.d[0] * d_dim_size;

        cudaError_t status;
        CHECK(status = cudaMemcpyAsync(outputs[0], psrc, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        return status;
    }

    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
    }

private:
    Dims     in_dims_;
    DimsNCHW out_dims_;
    Dims     slice_start_;
    Dims     slice_end_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createSlicePlugin(Dims dims, Dims slice_start, Dims slice_end, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new SlicePlugin(dims, slice_start, slice_end, log_, name));
    return plugins_.back();
}

} }