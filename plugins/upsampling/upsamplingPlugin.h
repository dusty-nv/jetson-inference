#ifndef __UPSAMPLING_PLUGIN_H__
#define __UPSAMPLING_PLUGIN_H__

#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "upsampling.h"

#include <vector>
#include <string>
#include <cassert>
#include <iostream>

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

using namespace nvinfer1;

class NearestNeighborUpsamplingPlugin : public IPluginV2
{
public:
    NearestNeighborUpsamplingPlugin(int nbInputChannels, int inputHeight, int inputWidth)
    {
        mNbInputChannels = nbInputChannels;
        mInputWidth = inputWidth;
        mInputHeight = inputHeight;
    }

    NearestNeighborUpsamplingPlugin(const Weights *weights, size_t nbWeights) { }

    NearestNeighborUpsamplingPlugin(const void* data, size_t length)
    {
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mInputWidth);
        read(d, mInputHeight);
        read(d, mDataType);
        assert(d == a + length);
    }

    ~NearestNeighborUpsamplingPlugin() { }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        mNbInputChannels = inputs[0].d[0];
        mInputHeight = inputs[0].d[1];
        mInputWidth = inputs[0].d[2];
        return nvinfer1::Dims3(inputs[0].d[0], 2 * inputs[0].d[1], 2 * inputs[0].d[2]);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override 
    { 
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; 
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert(supportsFormat(type, format));
        mDataType = type;
    }

    int initialize() override
    {
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    virtual void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        // write below code for custom variables
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    virtual size_t getSerializationSize() const override
    {
        // 3 size_t values: input width, input height, and number of channels
        // and one more value for data type
        return sizeof(DataType) + 3 * sizeof(int);
    }

    virtual void serialize(void* buffer) const override
    {
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mNbInputChannels);
        write(d, mInputWidth);
        write(d, mInputHeight);
        write(d, mDataType);
        assert(d == a + getSerializationSize());
    }

    const char* getPluginType() const override 
    { 
        return "ResizeNearestNeighbor";
    }

    const char* getPluginVersion() const override 
    { 
        return "1";
    }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new NearestNeighborUpsamplingPlugin(mNbInputChannels, mInputHeight, mInputWidth);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> T read(const char*& buffer, T& val) const
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    DataType mDataType{DataType::kFLOAT};
    cudnnHandle_t mCudnn;
    cublasHandle_t mCublas;
    int mNbInputChannels=0, mInputWidth=0, mInputHeight=0;
    std::string mNamespace = "";
};


class NearestNeighborUpsamplingPluginCreator: public IPluginCreator
{
public:
    NearestNeighborUpsamplingPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("nbInputChannels", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputHeight", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputWidth", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~NearestNeighborUpsamplingPluginCreator() {}

    const char* getPluginName() const override 
    {
        return "ResizeNearestNeighbor"; 
    }

    const char* getPluginVersion() const override
    {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new NearestNeighborUpsamplingPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
    int mNbInputChannels, mInputHeight, mInputWidth;
};

PluginFieldCollection NearestNeighborUpsamplingPluginCreator::mFC{};
std::vector<PluginField> NearestNeighborUpsamplingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(NearestNeighborUpsamplingPluginCreator);

#endif
