// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>
#include <cstring>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Cost volume plugin.
// -----------------------------------------------------------------
class CostVolumePlugin: public IPluginExt
{
public:
    CostVolumePlugin(DataType data_type, CostVolumeType cv_type, int max_disparity,
                     ILogger& log, std::string name):
        data_type_(data_type), format_(PluginFormat::kNCHW), 
        cv_type_(cv_type), max_disparity_(max_disparity), log_(log), name_(name)
    {
        assert(data_type_ == DataType::kFLOAT || data_type_ == DataType::kHALF);
        assert(max_disparity_ > 0);
        assert(cv_type_ == CostVolumeType::kDefault || cv_type_ == CostVolumeType::kCorrelation);
    }

    // Deserialization ctor.
    CostVolumePlugin(const char* name, const void* data, size_t size, ILogger& log):
        log_(log), name_(name)
    {
        // REVIEW alexeyk: add asserts.
        std::istringstream ss(std::string((const char*)data, size));
        // Note: starting with data_type_ as plugin type was already processed by the factory.
        data_type_ = read_stream<DataType>(ss);
        format_    = read_stream<PluginFormat>(ss);
        cv_type_   = read_stream<CostVolumeType>(ss);
        max_disparity_  = read_stream<int>(ss);
        in_dims_.nbDims = read_stream<int>(ss);
        for (int i = 0; i < in_dims_.nbDims; i++)
            in_dims_.d[i] = read_stream<int>(ss);
        out_dims_.nbDims = read_stream<int>(ss);
        for (int i = 0; i < out_dims_.nbDims; i++)
            out_dims_.d[i] = read_stream<int>(ss);

        // Check that nothing is left in the stream.
        assert((ss >> std::ws).eof());
    }

    CostVolumePlugin(CostVolumePlugin&&) = delete;

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        // See EluPlugin::supportsFormat for the notes.
        // Other combinations are not currently implemented.
        // REVIEW alexeyk: kHALF && kNCHW is only for testing on the host as TRT fails with assert when using kHALF && kNC2HW2.
        bool supported_formats = (type == DataType::kFLOAT && format == PluginFormat::kNCHW)  ||
                                 //(type == DataType::kHALF  && format == PluginFormat::kNCHW)  ||
                                 (type == DataType::kHALF  && format == PluginFormat::kNC2HW2);
        return (type == data_type_) && supported_formats;
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        // Expecting 2 3D inputs, left and right, which are outputs of 2D convolutions
        // and are of the same shape.
        assert(nbInputDims == 2);
        for (int i = 0; i < nbInputDims; i++)
            assert(inputs[i].nbDims == 3);
        assert(DimsUtils::areEqual(inputs[0], inputs[1]));

        in_dims_  = inputs[0];
        // 4D output in case of default CV.
        // 3D - correlation.
        if (cv_type_ == CostVolumeType::kDefault)
            out_dims_ = DimsNCHW(max_disparity_, 2 * in_dims_.d[0], in_dims_.d[1], in_dims_.d[2]);
        else if (cv_type_ == CostVolumeType::kCorrelation)
            out_dims_ = DimsCHW(max_disparity_, in_dims_.d[1], in_dims_.d[2]);
        else
            assert(false);
        return out_dims_;
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                             DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert(nbInputs  == 2);
        assert(nbOutputs == 1);
        assert(DimsUtils::areEqual(inputDims[0],  in_dims_));
        assert(DimsUtils::areEqual(inputDims[1],  in_dims_));
        assert(DimsUtils::areEqual(outputDims[0], out_dims_));
        assert(type == data_type_);
        assert(maxBatchSize == 1);

        format_ = format;

        log_.log(ILogger::Severity::kINFO, (name_ + ": InDims(x2): " + DimsUtils::toString(in_dims_)).c_str());
        log_.log(ILogger::Severity::kINFO, (name_ + ": OutDims   : " + DimsUtils::toString(out_dims_)).c_str());
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
        assert(maxBatchSize == 1);
        return 0;
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        assert(batchSize == 1);

        auto pleft     = static_cast<const float*>(inputs[0]);
        auto pright    = static_cast<const float*>(inputs[1]);
        auto pcost_vol = static_cast<float*>(outputs[0]);

        cudaError_t status = (cudaError_t)-1;
        if (cv_type_ == CostVolumeType::kDefault)
            CHECK(status = CudaKernels::computeCostVolume(data_type_, pleft, pright, in_dims_, pcost_vol, out_dims_, stream));
        else if (cv_type_ == CostVolumeType::kCorrelation)
            CHECK(status = CudaKernels::computeCorrCostVolume(data_type_, pleft, pright, in_dims_, pcost_vol, out_dims_, stream));
        else
            assert(false);

        return status;
    }

    size_t getSerializationSize() override
    {
        return serialize().size();
    }

    void serialize(void* buffer) override
    {
        assert(buffer != nullptr);

        auto data = serialize();
        std::memcpy(buffer, data.c_str(), data.size());
    }

private:
    std::string serialize()
    {
        std::ostringstream ss(std::ios_base::binary);
        write_stream((int32_t)StereoDnnPluginFactory::PluginType::kCostVolume, ss);
        write_stream((int32_t)data_type_, ss);
        write_stream((uint8_t)format_, ss);
        write_stream((int32_t)cv_type_, ss);
        write_stream(max_disparity_, ss);
        write_stream(in_dims_.nbDims, ss);
        assert(in_dims_.nbDims <= Dims::MAX_DIMS);
        for (int i = 0; i < in_dims_.nbDims; i++)
            write_stream(in_dims_.d[i], ss);
        write_stream(out_dims_.nbDims, ss);
        assert(out_dims_.nbDims <= Dims::MAX_DIMS);
        for (int i = 0; i < out_dims_.nbDims; i++)
            write_stream(out_dims_.d[i], ss);

        return ss.str();
    }

private:
    DataType       data_type_;
    PluginFormat   format_;
    CostVolumeType cv_type_;

    int  max_disparity_;
    Dims in_dims_;
    Dims out_dims_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createCostVolumePlugin(DataType data_type, CostVolumeType cv_type, int max_disparity,
                                                 std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new CostVolumePlugin(data_type, cv_type, max_disparity, log_, name));
    return plugins_.back();
}

// Deserialization method.
IPlugin* PluginContainer::deserializeCostVolumePlugin(const char* name, const void* data, size_t size)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new CostVolumePlugin(name, data, size, log_));
    return plugins_.back();
}

} }