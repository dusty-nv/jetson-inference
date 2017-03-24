/*
 * inference-101
 */

#ifndef __CUDA_MAPPED_MEMORY_H_
#define __CUDA_MAPPED_MEMORY_H_


#include "cudaUtility.h"


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 * @ingroup util
 */
inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)) )
		return false;

	if( CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)) )
		return false;

	memset(*cpuPtr, 0, size);
	printf("[cuda]  cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
	return true;
}


#endif
