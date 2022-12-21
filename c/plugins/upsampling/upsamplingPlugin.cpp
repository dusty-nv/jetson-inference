#include "upsamplingPlugin.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "upsampling.h"

int NearestNeighborUpsamplingPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // perform nearest neighbor upsampling using cuda
    CHECK(cublasSetStream(mCublas, stream));
    CHECK(cudnnSetStream(mCudnn, stream));
    if (mDataType == DataType::kFLOAT)
        CHECK(cudaResizeNearestNeighbor<float>((float*)inputs[0], mNbInputChannels, mInputHeight, mInputWidth, (float*)outputs[0], stream));
    else
        CHECK(cudaResizeNearestNeighbor<__half>((__half*)inputs[0], mNbInputChannels, mInputHeight, mInputWidth, (__half*)outputs[0], stream));
    return 0;
}

IPluginV2* NearestNeighborUpsamplingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "nbInputChannels"))
        {
            mNbInputChannels = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "inputHeight"))
        {
            mInputHeight = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "inputWidth"))
        {
            mInputWidth = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new NearestNeighborUpsamplingPlugin(mNbInputChannels, mInputHeight, mInputWidth);
}
