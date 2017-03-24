/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __IMAGE_LOADER_H_
#define __IMAGE_LOADER_H_


#include "cudaUtility.h"


/**
 * Load a color image from disk into CUDA memory with alpha.
 * This function loads the image into shared CPU/GPU memory, using the functions from cudaMappedMemory.h
 *
 * @param filename Path to the image file on disk.
 * @param cpu Pointer to CPU buffer allocated containing the image.
 * @param gpu Pointer to CUDA device buffer residing on GPU containing image.
 * @param width Variable containing width in pixels of the image.
 * @param height Variable containing height in pixels of the image.
 *
 * @ingroup util
 */
bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height );


/**
 * Save an image to disk
 * @ingroup util
 */
bool saveImageRGBA( const char* filename, float4* cpu, int width, int height, float max_pixel=255.0f );


/**
 * Load a color image from disk into CUDA memory.
 * This function loads the image into shared CPU/GPU memory, using the functions from cudaMappedMemory.h
 *
 * @param filename Path to the image file on disk.
 * @param cpu Pointer to CPU buffer allocated containing the image.
 * @param gpu Pointer to CUDA device buffer residing on GPU containing image.
 * @param width Variable containing width in pixels of the image.
 * @param height Variable containing height in pixels of the image.
 *
 * @ingroup util
 */
bool loadImageRGB( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean=make_float3(0,0,0) );


/**
 * Load a color image from disk into CUDA memory.
 * This function loads the image into shared CPU/GPU memory, using the functions from cudaMappedMemory.h
 *
 * @param filename Path to the image file on disk.
 * @param cpu Pointer to CPU buffer allocated containing the image.
 * @param gpu Pointer to CUDA device buffer residing on GPU containing image.
 * @param width Variable containing width in pixels of the image.
 * @param height Variable containing height in pixels of the image.
 *
 * @ingroup util
 */
bool loadImageBGR( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean=make_float3(0,0,0) );



#endif
