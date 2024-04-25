#ifndef _YOLOV3_
#define _YOLOV3_

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "../util/logger.h"
#include <fstream>
#include <vector>
#include "jetson-utils/logging.h"
#include "../util/utils.h"
#include <iostream>
#include "../yololayer.h"
#include "jetson-utils/cudaMappedMemory.h"
#include "kernel.h"

using sample::gLogError;
using sample::gLogInfo;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define NMS_THRESH 0.4f
#define BBOX_CONF_THRESH 0.5f

#define INPUT_H 320//Yolo::INPUT_H;
#define INPUT_W 320//Yolo::INPUT_W;
#define OUTPUT_SIZE 1000 * 7 + 1  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"

class YoloV3{
    public:
	YoloV3(const std::string &engineFilename);
	~YoloV3();
	bool initEngine();
	bool infer();
	void doInference(float* input, float* output, int batchSize);
	void process(uchar3* input_img, int width, int height);
	void getRgb(uchar3* input_img,uchar3* out_img,int srcWidth, int srcHeight, int dstWidth, int dstHeight);
	void **mBindings;
private:
	nvinfer1::ICudaEngine *mEngine;
	nvinfer1::IRuntime *mInfer;
	nvinfer1::IExecutionContext *mContext;
	cudaStream_t mStream;
};

#endif