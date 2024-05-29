#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <map>
#include <algorithm>

#include "yololayer.h"
#include "kernel.h"
#include "logger.h"

#include "jetson-utils/cudaMappedMemory.h"
#include "jetson-utils/logging.h"


using sample::gLogError;
using sample::gLogInfo;

#define NMS_THRESH 0.4f
#define BBOX_CONF_THRESH 0.5f

#define INPUT_H 320//Yolo::INPUT_H;
#define INPUT_W 320//Yolo::INPUT_W;
#define OUTPUT_SIZE 1000 * 7 + 1  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"

#define DETECTION_ROI_W 540
#define DETECTION_ROI_H 540

static float iou(float lbox[4], float rbox[4])
{
	float interBox[] = {
		std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
		std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
		std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
		std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Yolo::Detection &a, const Yolo::Detection &b)
{
	return a.det_confidence > b.det_confidence;
}

static void nms(std::vector<Yolo::Detection> &res, float *output, float nms_thresh = NMS_THRESH)
{
	std::map<float, std::vector<Yolo::Detection>> m;
	for (int i = 0; i < output[0] && i < 1000; i++)
	{
		if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH)
			continue;
		Yolo::Detection det;
		memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
		if (m.count(det.class_id) == 0)
			m.emplace(det.class_id, std::vector<Yolo::Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++)
	{
		// std::cout << it->second[0].class_id << " --- " << std::endl;
		auto &dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto &item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) > nms_thresh)
				{
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

class YoloV3{
    public:
	YoloV3(const std::string &engineFilename);
	~YoloV3();
	bool InitEngine();
	bool infer();
	void PreProcess(uchar3 *input_img);
	void Process();
	void PostProcess();
	void OverlayBBoxesOnVisImage(uchar3 *out_image, int img_width, int img_height);
private:
	nvinfer1::ICudaEngine *mEngine;
	nvinfer1::IRuntime *mInfer;
	nvinfer1::IExecutionContext *mContext;
	cudaStream_t mStream;
	void **mBindings;

	std::vector<Yolo::Detection> detected_objects;
};