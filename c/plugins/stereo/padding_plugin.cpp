// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Tensor padding plugin.
// -----------------------------------------------------------------
class PaddingPlugin: public IPlugin
{
public:
    PaddingPlugin(DimsNCHW pad_start, DimsNCHW pad_end, ILogger& log, std::string name):
        pad_start_(pad_start), pad_end_(pad_end), log_(log), name_(name)
    {
        // Only D end padding is currently supported.
        assert(pad_start_.n() == 0);
        assert(pad_start_.c() == 0);
        assert(pad_start_.h() == 0);
        assert(pad_start_.w() == 0);

        assert(pad_end_.n() >= 0);
        assert(pad_end_.c() == 0);
        assert(pad_end_.h() == 0);
        assert(pad_end_.w() == 0);
    }

    PaddingPlugin(PaddingPlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        // Only NCHW format is supported for now, TRT does not work (assert) with generic Dims.
        assert(inputs[0].nbDims == 4);

        in_dims_  = DimsNCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2], inputs[0].d[3]);
        out_dims_ = DimsNCHW(in_dims_.n() + pad_start_.n() + pad_end_.n(),
                             in_dims_.c() + pad_start_.c() + pad_end_.c(),
                             in_dims_.h() + pad_start_.h() + pad_end_.h(),
                             in_dims_.w() + pad_start_.w() + pad_end_.w());
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
        // Copy original tensor.
        size_t in_size_bytes = DimsUtils::getTensorSize(in_dims_) * sizeof(float);
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], in_size_bytes, cudaMemcpyDeviceToDevice, stream));

        // Zero out end D padding if needed.
        size_t out_size_bytes = DimsUtils::getTensorSize(out_dims_) * sizeof(float);
        if (out_size_bytes > in_size_bytes)
        {
            auto pdst = static_cast<unsigned char*>(outputs[0]) + in_size_bytes;
            CHECK(cudaMemsetAsync(pdst, 0, out_size_bytes - in_size_bytes, stream));
        }

        return 0;
    }

    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
    }

private:

private:
    DimsNCHW  pad_start_;
    DimsNCHW  pad_end_;
    DimsNCHW  in_dims_;
    DimsNCHW  out_dims_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createPaddingPlugin(DimsNCHW pad_start, DimsNCHW pad_end, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new PaddingPlugin(pad_start, pad_end, log_, name));
    return plugins_.back();
}

} }