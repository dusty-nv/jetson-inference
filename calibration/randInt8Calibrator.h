/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __RAND_INT8_CALIBRATOR_H__
#define __RAND_INT8_CALIBRATOR_H__

#include "NvInfer.h"

#include <map>
#include <string>
#include <vector>


#if NV_TENSORRT_MAJOR >= 4

/**
 * Random INT8 Calibrator.
 * This calibrator is for testing performance without needing
 * sample datasets to generate the INT8 calibration table.
 */
class randInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
	/**
	 * Constructor
	 */
	randInt8Calibrator( int totalSamples, std::string cacheFile, 
					const std::map<std::string, nvinfer1::Dims3>& inputDimensions );

	/**
	 * Destructor
	 */
	~randInt8Calibrator();

	/**
	 * getBatchSize()
	 */
	inline int getBatchSize() const override	{ return 1; }

	/**
	 * getBatch()
	 */
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
	
	/**
	 * readCalibrationCache()
	 */
	const void* readCalibrationCache(size_t& length) override;
    
	/**
	 * writeCalibrationCache()
	 */
	virtual void writeCalibrationCache(const void*, size_t) override;

private:
	int mTotalSamples;
	int mCurrentSample;

	std::string mCacheFile;
	std::map<std::string, void*> mInputDeviceBuffers;
	std::map<std::string, nvinfer1::Dims3> mInputDimensions;
	std::vector<char> mCalibrationCache;
};

#endif
#endif

