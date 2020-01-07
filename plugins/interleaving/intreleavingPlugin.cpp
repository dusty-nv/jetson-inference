#include "interleavingPlugin.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "interleaving.h"

int InterleavingPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // perform interleaving operation using cuda
    CHECK(cublasSetStream(mCublas, stream));
    CHECK(cudnnSetStream(mCudnn, stream));
    if (mDataType == DataType::kFLOAT)
        CHECK(cudaInterleave<float>((float*)inputs[0],
                                    (float*)inputs[1],
                                    (float*)inputs[2],
                                    (float*)inputs[3],
                                    mNbInputChannels,
                                    mInputHeight,
                                    mInputWidth,
                                    (float*)outputs[0],
                                    stream));
    else
        CHECK(cudaInterleave<__half>((__half*)inputs[0],
                                    (__half*)inputs[1],
                                    (__half*)inputs[2],
                                    (__half*)inputs[3],
                                    mNbInputChannels,
                                    mInputHeight,
                                    mInputWidth,
                                    (__half*)outputs[0],
                                    stream));
    return 0;
}

IPluginV2* InterleavingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
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
    return new InterleavingPlugin(mNbInputChannels, mInputHeight, mInputWidth);
}
