#include "YoloV3.hpp"

static size_t sizeDims(const nvinfer1::Dims &dims, const size_t elementSize=1)
{
    size_t sz = dims.d[0];

    for (int n = 1; n < dims.nbDims; n++)
        sz *= dims.d[n];

    return sz * elementSize;
}

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
bool YoloV3::InitEngine()
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

void YoloV3::PreProcess(uchar3 *input_img)
{
	preprocessROIImgK(input_img, DETECTION_ROI_W, (float *)mBindings[0]);
}

void YoloV3::Process(){
	if (!mContext->enqueue(1,mBindings, mStream, NULL))
	{
		LogError("failed to enqueue TensorRT context on device\n");
	}
	cudaStreamSynchronize(mStream);
}

void YoloV3::PostProcess()
{
	detected_objects.clear();
	nms(detected_objects, (float *)mBindings[1]); // 3us
}

void YoloV3::OverlayBBoxesOnVisImage(uchar3 *out_image, int img_width, int img_height)
{
	uchar3 color = {0,255,0};
	for(int i = 0; i < detected_objects.size(); i++){
		int bbox_x = (detected_objects.at(i).bbox[0] * INPUT_W) / img_width;
		int bbox_y = (detected_objects.at(i).bbox[1] * INPUT_H) / img_height;
		int bbox_w = (detected_objects.at(i).bbox[2] * img_width) / INPUT_W;
		int bbox_h = (detected_objects.at(i).bbox[3] * img_height) / INPUT_H;

		drawBoundingBox(out_image, img_width, img_height, bbox_x, bbox_y, bbox_w, bbox_h, color);

		LogInfo("Detected: class: %f, x: %d,y: %d, w: %d,h: %d\n",detected_objects.at(i).class_id, bbox_x, bbox_y, bbox_w, bbox_h);
	}
}
