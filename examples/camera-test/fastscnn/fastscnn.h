#ifndef _FASTSCNN_
#define _FASTSCNN_

#include "jetson-utils/cudaMappedMemory.h"

#include "../util/logger.h"
#include "../util/utils.h"
#include "../ipm.h"
#include "../argmax.h"
//#include "../../../c/tensorNet.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <vector>

#define OUT_IMG_W 1024
#define OUT_IMG_H 512
#define IN_IMG_W 1280
#define IN_IMH_H 720
#define UV_GRID_COLS 524288

#define VOID 0
#define RIGHT_LANE 1
#define LEFT_LANE 2
#define MARKINGS 3
#define CHARGING_PAD 4
#define OBSTACLE 5

#define OBSTACLE_THRESH 10000

#if NV_TENSORRT_MAJOR >= 6
typedef nvinfer1::Dims3 Dims3;
#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]
#endif

typedef uchar3 pixelType; // this can be uchar3, uchar4, float3, float4

using sample::gLogError;
using sample::gLogInfo;

struct Obstacle{
	int numPixels;
	int smallest_x_obst;
	int biggest_x_obst;
	int smallest_y_obst;
	int biggest_y_obst;
};

class FastScnn
{
public:
	FastScnn(const std::string &engineFilename);

	~FastScnn()
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
		// if (mBindings != NULL)
		// {

		// 	CUDA_FREE_HOST(mBindings[0]);
		// 	CUDA_FREE_HOST(mBindings[1]);
		// 	CUDA_FREE_HOST(outputBindGPU1);
		// 	free(mBindings);
		// }
		if (mClassMap)
		{
			CUDA_FREE_HOST(mClassMap);
		}

		for( size_t n=0; n < mInputs.size(); n++ )
		{
			CUDA_FREE(mInputs[n].CUDA);
		}
		
		for( size_t n=0; n < mOutputs.size(); n++ )
			CUDA_FREE_HOST(mOutputs[n].CPU);
		
		free(mBindings);

		cudaStreamDestroy(mStream);
	}
	int initEngine();
	bool process(uchar3 *image, uint32_t width, uint32_t height);
	bool infer();
	void toInputBinding(float3 *img);
	void toInput(float *arr);
	bool classify();
	bool getRGB(pixelType *img, int middle_x);
	bool loadGrid();
	int getLaneCenter(int laneIdx);
	int getParkingDirection(int offset);
	bool isObstacleOnLane(int dev);

		/**
	 * Retrieve the number of input layers to the network.
	 */
	inline uint32_t GetInputLayers() const					{ return mInputs.size(); }

	/**
	 * Retrieve the number of output layers to the network.
	 */
	inline uint32_t GetOutputLayers() const					{ return mOutputs.size(); }

	// float *outputBind;
	// float *outputBindCPU1;
	// float *outputBindGPU1;
	// float *inputBind;
	uint8_t *mClassMap;
	int *uGrid;
	int *vGrid;
	void **mBindings;

	struct layerInfo
	{
		std::string name;
		Dims3 dims;
		uint32_t size;
		uint32_t binding;
		float* CPU;
		float* CUDA;
	};

	std::vector<layerInfo> mInputs;
	std::vector<layerInfo> mOutputs;
	
private:
	void loopThroughClassmap(std::vector<int> &y_vals_lane, std::vector<int> &x_vals_lane, int classidx);
	nvinfer1::ICudaEngine *mEngine;
	nvinfer1::IRuntime *mInfer;
	nvinfer1::IExecutionContext *mContext;
	cudaStream_t mStream;

	Obstacle obstacle;
};

#endif