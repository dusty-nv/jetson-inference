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

#include "randInt8Calibrator.h"
#include "cudaUtility.h"

#include <random>
#include <iterator>
#include <fstream>


#if NV_TENSORRT_MAJOR >= 4

//-------------------------------------------------------------------------------------------------
static inline int volume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}
//-------------------------------------------------------------------------------------------------


// constructor
randInt8Calibrator::randInt8Calibrator( int totalSamples, std::string cacheFile, const std::map<std::string, nvinfer1::Dims3>& inputDimensions )
								 : mTotalSamples(totalSamples)
								 , mCurrentSample(0)
								 , mCacheFile(cacheFile)
								 , mInputDimensions(inputDimensions)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

	for( auto& elem : mInputDimensions )
	{
		int elemCount = volume(elem.second);

		std::vector<float> rnd_data(elemCount);

		for( auto& val : rnd_data )
			val = distribution(generator);

		void* data;

		CUDA(cudaMalloc(&data, elemCount * sizeof(float)));
		CUDA(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

		mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
	}
}


// destructor
randInt8Calibrator::~randInt8Calibrator()
{
	for( auto& elem : mInputDeviceBuffers )
		CUDA(cudaFree(elem.second));
}


// getBatch()
bool randInt8Calibrator::getBatch( void* bindings[], const char* names[], int nbBindings ) NOEXCEPT
{
	if( mCurrentSample >= mTotalSamples )
		return false;

	for( int i = 0; i < nbBindings; ++i )
		bindings[i] = mInputDeviceBuffers[names[i]];

	++mCurrentSample;
	return true;
}


// readCalibrationCache()
const void* randInt8Calibrator::readCalibrationCache( size_t& length ) NOEXCEPT
{
	mCalibrationCache.clear();
	std::ifstream input(mCacheFile, std::ios::binary);
	input >> std::noskipws;

	if (input.good())
		std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

	length = mCalibrationCache.size();
	return length ? &mCalibrationCache[0] : nullptr;
}


// writeCalibrationCache()
void randInt8Calibrator::writeCalibrationCache( const void*, size_t ) NOEXCEPT
{

}

#endif

