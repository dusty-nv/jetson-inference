/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorNet.h"


/**
 * @note superResNet is only supported with TensorRT 5.0 and newer,
 * as it uses ONNX models and requires ONNX import support in TensorRT.
 */
#if NV_TENSORRT_MAJOR >= 5
#define HAS_SUPERRES_NET
#endif


/**
 * Super Resolution Network
 */
class superResNet : public tensorNet
{
public:
	/**
	 * Load super resolution network
	 */
	static superResNet* Create();

	/**
	 * Destroy
	 */
	~superResNet();

	/**
	 * Upscale a 4-channel RGBA image.
	 */
	bool UpscaleRGBA( float* input, uint32_t inputWidth, uint32_t inputHeight,
			        float* output, uint32_t outputWidth, uint32_t outputHeight,
			        float maxPixelValue=255.0f );

	/**
	 * Upscale a 4-channel RGBA image.
	 */
	bool UpscaleRGBA( float* input, float* output, float maxPixelValue=255.0f );

	/**
	 * Retrieve the width of the input image, in pixels.
	 */
	inline uint32_t GetInputWidth() const						{ return mWidth; }

	/**
	 * Retrieve the height of the input image, in pixels.
	 */
	inline uint32_t GetInputHeight() const						{ return mHeight; }

	/**
	 * Retrieve the width of the output image, in pixels.
	 */
	inline uint32_t GetOutputWidth() const						{ return DIMS_W(mOutputs[0].dims); }

	/**
	 * Retrieve the height of the output image, in pixels.
	 */
	inline uint32_t GetOutputHeight() const						{ return DIMS_H(mOutputs[0].dims); }

	/**
	 * Retrieve the scale factor between the input and output.
	 */
	inline uint32_t GetScaleFactor() const						{ return GetOutputWidth() / GetInputWidth(); }

protected:
	superResNet();
};


