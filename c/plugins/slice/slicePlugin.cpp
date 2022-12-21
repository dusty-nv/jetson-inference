#include "slicePlugin.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "stridedSlice.h"

int StridedSlicePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // perform slicing using cuda
    CHECK(cublasSetStream(mCublas, stream));
    CHECK(cudnnSetStream(mCudnn, stream));
    if (mDataType == DataType::kFLOAT)
        CHECK(cudaSlice<float>((float*)inputs[0], SLICE_INPUT_C, SLICE_INPUT_H, SLICE_INPUT_W, (float*)outputs[0], stream));
    else
        CHECK(cudaSlice<__half>((__half*)inputs[0], SLICE_INPUT_C, SLICE_INPUT_H, SLICE_INPUT_W, (__half*)outputs[0], stream));
    return 0;
}

IPluginV2* StridedSlicePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "Index"))
        {
            mIndex = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "T"))
        {
            mT = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new StridedSlicePlugin(0, 0, 0);
}
