// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>
#include <cstring>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// ELU activation function plugin.
// -----------------------------------------------------------------
class EluPlugin: public IPluginExt
{
public:
    EluPlugin(DataType data_type, ILogger& log, std::string name):
        data_type_(data_type), format_(PluginFormat::kNCHW),
        max_batch_size_(0), log_(log), name_(name)
    {
        assert(data_type_ == DataType::kFLOAT || data_type_ == DataType::kHALF);
    }

    // Deserialization ctor.
    EluPlugin(const char* name, const void* data, size_t size, ILogger& log):
        max_batch_size_(1), log_(log), name_(name)
    {
        // REVIEW alexeyk: add asserts.
        std::istringstream ss(std::string((const char*)data, size));
        // Note: starting with data_type_ as plugin type was already processed by the factory.
        data_type_ = read_stream<DataType>(ss);
        format_    = read_stream<PluginFormat>(ss);
        in_dims_.nbDims = read_stream<int>(ss);
        for (int i = 0; i < in_dims_.nbDims; i++)
            in_dims_.d[i] = read_stream<int>(ss);

        // Check that nothing is left in the stream.
        assert((ss >> std::ws).eof());
    }

    EluPlugin(EluPlugin&&) = delete;

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        // On TX2, the most efficient format in FP16 is kNC2HW2. Using other formats will make
        // TRT to insert reformat layers which hurts performance.
        bool supported_formats = (format == PluginFormat::kNCHW || format == PluginFormat::kNC2HW2);
        // REVIEW alexeyk: by using data type provided in ctor we effectively disabling
        // TRT autotuner which could use different types during tuning. Fine for now.
        return (type == data_type_) && supported_formats;
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        in_dims_ = inputs[0];
        // No restrictions on input dims.

        return in_dims_;
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                             DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
        // Sanity check.
        assert(DimsUtils::areEqual(in_dims_, inputDims[0]));
        assert(DimsUtils::areEqual(in_dims_, outputDims[0]));
        assert(type == data_type_);

        format_         = format;
        max_batch_size_ = maxBatchSize;

        auto str = name_ + ": Dims: " + DimsUtils::toString(in_dims_) +
                           ", Format: [" + StrUtils::toString(type) + ", " + StrUtils::toString(format) + "]";
        log_.log(ILogger::Severity::kINFO, str.c_str());
    }

    int initialize() override
    {
        cudnnStatus_t status;

        CHECK(status = cudnnCreate(&cudnn_));
        CHECK(status = cudnnCreateActivationDescriptor(&act_));
        CHECK(status = cudnnSetActivationDescriptor(act_, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, 1.0));
        CHECK(status = cudnnCreateTensorDescriptor(&t_desc_));

        setTensorDescriptor();

        return 0;
    }

    void terminate() override
    {
        assert(isValid());

        if (act_ != nullptr)
            CHECK(cudnnDestroyActivationDescriptor(act_));
        if (t_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(t_desc_));
        if (cudnn_ != nullptr)
            CHECK(cudnnDestroy(cudnn_));
        act_    = nullptr;
        t_desc_ = nullptr;
        cudnn_  = nullptr;

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
            updateTensorDescriptor(batchSize);
        CHECK(status = cudnnActivationForward(cudnn_, act_, &Consts::kOne, t_desc_, inputs[0],  &Consts::kZero, t_desc_, outputs[0]));

        return status == CUDNN_STATUS_SUCCESS ? 0 : -1;
    }

    size_t getSerializationSize() override
    {
        assert(isValid());
        return serialize().size();
    }

    void serialize(void* buffer) override
    {
        assert(buffer != nullptr);
        assert(isValid());

        auto data = serialize();
        std::memcpy(buffer, data.c_str(), data.size());
    }

private:
    bool isValid() const
    {
        return cudnn_ != nullptr;
    }

    void setTensorDescriptor()
    {
        assert(isValid());

        tensor_dims_.nbDims = in_dims_.nbDims + 1;
        tensor_dims_.d[0]   = max_batch_size_;
        std::copy(in_dims_.d, in_dims_.d + in_dims_.nbDims, tensor_dims_.d + 1);
        // If the current format is kNC2HW2 and C is odd, we need to adjust actual
        // tensor dimensions to reflect packing.
        if (format_ == PluginFormat::kNC2HW2)
            tensor_dims_.d[1] += in_dims_.d[0] % 2;

        tensor_strides_ = DimsUtils::getStrides(tensor_dims_);

        CHECK(cudnnSetTensorNdDescriptor(t_desc_, trtToCudnnDataType(data_type_), tensor_dims_.nbDims, tensor_dims_.d, tensor_strides_.d));
    }

    // Updates tensor descriptor according to batch_size.
    void updateTensorDescriptor(int batch_size)
    {
        max_batch_size_   = batch_size;
        // No other parameters require update.
        tensor_dims_.d[0] = batch_size;
        CHECK(cudnnSetTensorNdDescriptor(t_desc_, trtToCudnnDataType(data_type_), tensor_dims_.nbDims, tensor_dims_.d, tensor_strides_.d));
    }

    std::string serialize()
    {
        std::ostringstream ss(std::ios_base::binary);
        write_stream((int32_t)StereoDnnPluginFactory::PluginType::kElu, ss);
        write_stream((int32_t)data_type_, ss);
        write_stream((uint8_t)format_, ss);
        write_stream(in_dims_.nbDims, ss);
        assert(in_dims_.nbDims <= Dims::MAX_DIMS);
        for (int i = 0; i < in_dims_.nbDims; i++)
            write_stream(in_dims_.d[i], ss);

        return ss.str();
    }

private:
    DataType     data_type_;
    PluginFormat format_;

    cudnnHandle_t               cudnn_  = nullptr;
    cudnnActivationDescriptor_t act_    = nullptr;
    cudnnTensorDescriptor_t     t_desc_ = nullptr;

    Dims in_dims_;
    Dims tensor_dims_;
    Dims tensor_strides_;
    int  max_batch_size_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createEluPlugin(DataType data_type, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new EluPlugin(data_type, log_, name));
    return plugins_.back();
}

// Deserialization method.
IPlugin* PluginContainer::deserializeEluPlugin(const char* name, const void* data, size_t size)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new EluPlugin(name, data, size, log_));
    return plugins_.back();
}

} }