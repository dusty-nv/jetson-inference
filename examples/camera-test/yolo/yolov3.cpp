#include "yolov3.h"


YoloV3::YoloV3(const std::string &engineFilename)
{
	mEngine = NULL;
	mInfer = NULL;
	mContext = NULL;
	mStream = NULL;
	// initLibNvInferPlugins(&sample::gLogger, "");
	//  De-serialize engine from file
	std::ifstream engineFile(engineFilename, std::ios::binary);
	if (engineFile.fail())
	{
		LogError("Failed to deserialize engine\n");
		return;
	}

	engineFile.seekg(0, std::ifstream::end);
	auto fsize = engineFile.tellg();
	engineFile.seekg(0, std::ifstream::beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	mInfer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	// mInfer = CREATE_INFER_RUNTIME(gLogger);
	mEngine = mInfer->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}
YoloV3::~YoloV3()
{
	if (mEngine)
	{
		mEngine->destroy();
	}
	if (mInfer)
	{
		mInfer->destroy();
	}
	if (mContext)
	{
		mContext->destroy();
	}
	if (mBindings != NULL)
	{

		CUDA_FREE_HOST(mBindings[0]);
		CUDA_FREE_HOST(mBindings[1]);
		free(mBindings);
	}
	cudaStreamDestroy(mStream);
}
bool YoloV3::initEngine()
{
	// Context
	if (!mEngine)
		return false;

	mContext = mEngine->createExecutionContext();
	if (!mContext)
	{
		LogError("Failed to create execution context\n");
		return 0;
	}
	auto input_idx = mEngine->getBindingIndex("data");

	if (input_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
	auto input_dims = mContext->getBindingDimensions(input_idx);
	mContext->setBindingDimensions(input_idx, input_dims);
	auto input_size = sizeDims(input_dims, 1) * sizeof(float);

	auto output_idx = mEngine->getBindingIndex("prob");
	if (output_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(output_idx) == nvinfer1::DataType::kFLOAT);
	auto output_dims = mContext->getBindingDimensions(output_idx);
	const size_t output_size = sizeDims(output_dims, 1) * sizeof(float);

	void *outputCUDA = NULL;
	void *outputCPU = NULL;
	if (!cudaAllocMapped((void **)&outputCPU, (void **)&outputCUDA, output_size))
	{
		LogError("Could not allocate output CUDA\n");
		return false;
	}

	void *inputCUDA = NULL;
	void *inputCPU = NULL;
	if (!cudaAllocMapped((void **)&inputCPU, (void **)&inputCUDA, input_size))
	{
		LogError("Could not allocate input CUDA\n");
		return false;
	}
	LogVerbose("Input size: %d\n", input_size);
	LogVerbose("Output size: %d\n", output_size);

	const int bindingSize = 2 * sizeof(void *);
	mBindings = (void **)malloc(bindingSize);
	mBindings[0] = (float *)inputCUDA;
	mBindings[1] = (float *)outputCUDA;

	if (cudaStreamCreate(&mStream) != cudaSuccess)
	{
		LogError("ERROR: cuda stream creation failed.\n");
		return false;
	}

	LogVerbose("Successfully initialized yolo network\n");
	return true;
}
bool YoloV3::infer()
{
	if (!mContext->enqueue(1,mBindings, mStream, NULL))
	{
		LogError("failed to enqueue TensorRT context on device\n");
		return false;
	}
	cudaStreamSynchronize(mStream);
	return true;
}
void YoloV3::doInference(float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(mBindings[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mContext->enqueue(batchSize, mBindings, mStream, nullptr);
    CHECK(cudaMemcpyAsync(output, mBindings[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
}
void YoloV3::process(uchar3* input_img, int width, int height){
	preprocessROIImgK(input_img, (float *)mBindings[0]);
	//preprocessImgK(input_img, (float *)mBindings[0], width, height, 320, 320, 0, 70);
	infer();
	//CUDA(cudaDeviceSynchronize());
}
void YoloV3::getRgb(uchar3* input_img,uchar3* out_img,int srcWidth, int srcHeight, int dstWidth, int dstHeight){
	// Assume dstWidth and dstHeight are 540
    // Assume srcWidth and srcHeight are 1920 and 1080 respectively

    int offsetX = srcWidth - dstWidth;  // 1920 - 540 = 1380
    int offsetY = 0;  // Top corner

    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            // Calculate position in source and destination arrays
            int srcPos = (y + offsetY) * srcWidth + (x + offsetX);
            int dstPos = y * dstWidth + x;

            // Copy pixel
            out_img[dstPos] = input_img[srcPos];
        }
    }
}