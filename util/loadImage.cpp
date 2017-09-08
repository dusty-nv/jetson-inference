/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#include "loadImage.h"
#include "cudaMappedMemory.h"

#include <QImage>



bool saveImageRGBA( const char* filename, float4* cpu, int width, int height, float max_pixel )
{
	if( !filename || !cpu || !width || !height )
	{
		printf("saveImageRGBA - invalid parameter\n");
		return false;
	}
	
	const float scale = 255.0f / max_pixel;
	QImage img(width, height, QImage::Format_RGB32);

	for( int y=0; y < height; y++ )
	{
		for( int x=0; x < width; x++ )
		{
			const float4 px = cpu[y * width + x];
			//printf("%03u %03u   %f\n", x, y, normPx);
			img.setPixel(x, y, qRgb(px.x * scale, px.y * scale, px.z * scale));
		}
	}


	/*
	 * save file
	 */
	if( !img.save(filename/*, "PNG", 100*/) )
	{
		printf("failed to save %ix%i output image to %s\n", width, height, filename);
		return false;
	}
	
	return true;
}


// loadImageRGBA
bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height )
{
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf("loadImageRGBA - invalid parameter\n");
		return false;
	}
	
	// load original image
	QImage qImg;

	if( !qImg.load(filename) )
	{
		printf("failed to load image %s\n", filename);
		return false;
	}

	if( *width != 0 && *height != 0 )
		qImg = qImg.scaled(*width, *height, Qt::IgnoreAspectRatio);
	
	const uint32_t imgWidth  = qImg.width();
	const uint32_t imgHeight = qImg.height();
	const uint32_t imgPixels = imgWidth * imgHeight;
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 4;

	printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);

	// allocate buffer for the image
	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, filename);
		return false;
	}

	float4* cpuPtr = *cpu;
	
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			const QRgb rgb  = qImg.pixel(x,y);
			const float4 px = make_float4(float(qRed(rgb)), 
										  float(qGreen(rgb)), 
										  float(qBlue(rgb)),
										  float(qAlpha(rgb)));
			
			cpuPtr[y*imgWidth+x] = px;
		}
	}
	
	*width  = imgWidth;
	*height = imgHeight;	
	return true;
}


// loadImageRGB
bool loadImageRGB( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf("loadImageRGB - invalid parameter\n");
		return false;
	}
	
	// load original image
	QImage qImg;

	if( !qImg.load(filename) )
	{
		printf("failed to load image %s\n", filename);
		return false;
	}

	if( *width != 0 && *height != 0 )
		qImg = qImg.scaled(*width, *height, Qt::IgnoreAspectRatio);
	
	const uint32_t imgWidth  = qImg.width();
	const uint32_t imgHeight = qImg.height();
	const uint32_t imgPixels = imgWidth * imgHeight;
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

	printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);

	// allocate buffer for the image
	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, filename);
		return false;
	}

	float* cpuPtr = (float*)*cpu;
	
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			const QRgb rgb  = qImg.pixel(x,y);
			const float mul = 1.0f; 	//1.0f / 255.0f;
			const float3 px = make_float3((float(qRed(rgb))   - mean.x) * mul, 
										  (float(qGreen(rgb)) - mean.y) * mul, 
										  (float(qBlue(rgb))  - mean.z) * mul );
			
			// note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
			cpuPtr[imgPixels * 0 + y * imgWidth + x] = px.x; 
			cpuPtr[imgPixels * 1 + y * imgWidth + x] = px.y; 
			cpuPtr[imgPixels * 2 + y * imgWidth + x] = px.z; 
		}
	}
		
	*width  = imgWidth;
	*height = imgHeight;
	return true;
}


// loadImageBGR
bool loadImageBGR( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf("loadImageRGB - invalid parameter\n");
		return false;
	}
	
	// load original image
	QImage qImg;

	if( !qImg.load(filename) )
	{
		printf("failed to load image %s\n", filename);
		return false;
	}

	if( *width != 0 && *height != 0 )
		qImg = qImg.scaled(*width, *height, Qt::IgnoreAspectRatio);
	
	const uint32_t imgWidth  = qImg.width();
	const uint32_t imgHeight = qImg.height();
	const uint32_t imgPixels = imgWidth * imgHeight;
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

	printf("loaded image  %s  (%u x %u)  %zu bytes\n", filename, imgWidth, imgHeight, imgSize);

	// allocate buffer for the image
	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, filename);
		return false;
	}

	float* cpuPtr = (float*)*cpu;
	
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			const QRgb rgb  = qImg.pixel(x,y);
			const float mul = 1.0f; 	//1.0f / 255.0f;
			const float3 px = make_float3((float(qBlue(rgb))  - mean.x) * mul, 
										  (float(qGreen(rgb)) - mean.y) * mul, 
										  (float(qRed(rgb))   - mean.z) * mul );
			
			// note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
			cpuPtr[imgPixels * 0 + y * imgWidth + x] = px.x; 
			cpuPtr[imgPixels * 1 + y * imgWidth + x] = px.y; 
			cpuPtr[imgPixels * 2 + y * imgWidth + x] = px.z; 
		}
	}
			
	return true;
}
