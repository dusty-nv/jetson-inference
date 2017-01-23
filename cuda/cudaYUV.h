/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_YUV_CONVERT_H
#define __CUDA_YUV_CONVERT_H

#include <stdint.h>
#include <NVX/nvx.h>
#include <VX/vx_types.h>
#include "cudaUtility.h"

//////////////////////////////////////////////////////////////////////////////////
/// @name YUV to RGBf
//////////////////////////////////////////////////////////////////////////////////
bool ConvertYUVtoRGBA( void* input, void** outputCPU, void** outputGPU, size_t width, size_t height );

cudaError_t cudaYUVToRGBA( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaYUVToRGBA( uint8_t* input, uint8_t* output, size_t width, size_t height );

bool ConvertRGBtoYUV( void* input, bool gpuAddr, void** output, size_t width, size_t height );

cudaError_t cudaRGBAToYUV( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaRGBAToYUV( uint8_t* input, uint8_t* output, size_t width, size_t height );

cudaError_t cudaRGBToYUV( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaRGBToYUV( uint8_t* input, uint8_t* output, size_t width, size_t height );

cudaError_t cudaMotionFields( uint8_t* image, vx_float32* motionfeilds, size_t width, size_t height);

cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, size_t srcPitch, float4* destDev, size_t destPitch, size_t width, size_t height );
cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, float4* destDev, size_t width, size_t height );

cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, size_t srcPitch, float4* destDev, size_t destPitch, size_t width, size_t height );
cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, float4* destDev, size_t width, size_t height );

cudaError_t cudaYUVToRGBAf( uint8_t* srcDev, size_t srcPitch, float4* destDev, size_t destPitch, size_t width, size_t height );
cudaError_t cudaYUVToRGBAf( uint8_t* srcDev, float4* destDev, size_t width, size_t height );

///@}

#endif

