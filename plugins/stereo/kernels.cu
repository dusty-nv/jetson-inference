// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cuda_fp16.h>

// Check async error.
// Sync and get kernel status in Debug builds.
#ifndef NDEBUG
    #define SYNC_AND_CHECK_STREAM(stream) do {          \
    cudaError_t status = cudaStreamSynchronize(stream); \
    if (status != cudaSuccess)                          \
        return status;                                  \
}while(false)
#else
    #define SYNC_AND_CHECK_STREAM(stream)
#endif

#define CHECKK(stream) do {                  \
    cudaError_t status = cudaGetLastError(); \
    if (status != cudaSuccess)               \
        return status;                       \
    SYNC_AND_CHECK_STREAM(stream);           \
}while(false)

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

static const int kMaxGridSizeY = 65535;
static const int kMaxGridSizeZ = 65535;

// -----------------------------------------------------------------
// Helper function to get block count.
// -----------------------------------------------------------------
static uint32_t getBlockCount(uint32_t total_size, uint32_t block_size)
{
    uint32_t res = (total_size + block_size - 1) / block_size;
    assert(res > 0);
    assert((size_t)res * block_size >= total_size);
    return res;
}

// REVIEW alexeyk: kernels are not optimized for now.

// -----------------------------------------------------------------
// Cost volume kernels.
// -----------------------------------------------------------------
template<typename T>
__global__ void costVolumeCopyKernel(const T* src, int32_t c, int32_t h, int32_t w, int32_t disp, T* dst)
{
    assert(src != nullptr);
    assert(dst != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= w || iy >= h || iz >= c)
        return;
    const size_t isrc   = iz * h * w + iy * w + ix;
    const size_t stride = 2 * c * h * w;
    T  val  = src[isrc];
    T* pdst = dst + isrc;
    for (int32_t idst = 0; idst < disp; idst++)
    {
        *pdst = val;
        pdst += stride;
    }
}

template<typename T>
__global__ void costVolumeCopyPadKernel(const T* src, int32_t c, int32_t h, int32_t w, int32_t disp, T* dst)
{
    assert(src != nullptr);
    assert(dst != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= w || iy >= h || iz >= c)
        return;
    const size_t isrc    = iz * h * w + iy * w + ix;
    size_t       stride  = c * h * w;
    const size_t idst    = isrc + stride;
    stride *= 2;

    T* pdst = dst + idst;
    for (int32_t pad = 0; pad < disp; pad++)
    {
        if (ix < pad)
            *pdst = 0;
        else
            *pdst = src[isrc - pad];
        pdst += stride;
    }
}

template<typename T>
__global__ void costVolumeKernel(const T* left, const T* right, int32_t c, int32_t h, int32_t w, int32_t disp, T* dst)
{
    assert(left  != nullptr);
    assert(right != nullptr);
    assert(dst   != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= w || iy >= h || iz >= c)
        return;

    // Setup initial indices.
    size_t       stride = c * h * w;
    // Left and right source is the same.
    const size_t ileft  = iz * h * w + iy * w + ix;
    T*           pdst_l = dst + ileft;
    const size_t iright = ileft;
    // Right destination is offset by 1 in c dimension.
    T*           pdst_r = dst + iright + stride;
    // Final stride is 2 in c dimension.
    stride *= 2;

    T  val_l  = left[ileft];
    for (int32_t pad = 0; pad < disp; pad++)
    {
        if (ix < pad)
            *pdst_r = 0;
        else
            *pdst_r = right[iright - pad];
        *pdst_l = val_l;
        pdst_l += stride;
        pdst_r += stride;
    }
}

template<>
cudaError_t CudaKernels::computeCostVolume(DataType data_type, const float* left, const float* right, Dims in_dims, 
                                           float* cost_vol, Dims out_dims, cudaStream_t stream)
{
    assert(data_type == DataType::kFLOAT);
    assert(in_dims.nbDims  == 3);
    assert(out_dims.nbDims == 4);

    dim3 b_dim{16, 16, 1};
    dim3 g_dim;
    g_dim.x = getBlockCount(in_dims.d[2], b_dim.x);
    g_dim.y = getBlockCount(in_dims.d[1], b_dim.y);
    g_dim.z = getBlockCount(in_dims.d[0], b_dim.z);

    // REVIEW alexeyk: using 2 kernels instead of one as it's not yet optimized so 2 kernels are faster.
    // REVIEW alexeyk: optimize, see gld_efficiency,gst_efficiency,gld_transactions,gst_transactions.
    // costVolumeKernel<<<g_dim, b_dim, 0, stream>>>(left, right, in_dims.d[0], in_dims.d[1], in_dims.d[2], out_dims.d[0],
    //                                               cost_vol);
    costVolumeCopyKernel<<<g_dim, b_dim, 0, stream>>>(left, in_dims.d[0], in_dims.d[1], in_dims.d[2], out_dims.d[0],
                                                      cost_vol);
    CHECKK(stream);
    costVolumeCopyPadKernel<<<g_dim, b_dim, 0, stream>>>(right, in_dims.d[0], in_dims.d[1], in_dims.d[2], out_dims.d[0],
                                                         cost_vol);
    CHECKK(stream);
    return cudaSuccess;
}

// -----------------------------------------------------------------
// Correlation cost volume kernels.
// -----------------------------------------------------------------

// FP32, NCHW kernel.
template<typename T>
__global__ void corrCostVolumeKernel(const T* left, const T* right, int32_t c, int32_t h, int32_t w, int32_t disp, T* dst)
{
    assert(left  != nullptr);
    assert(right != nullptr);
    assert(dst   != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= w || iy >= h)
        return;

    uint32_t pad = blockIdx.z;
    assert(pad < disp);
    size_t stride = h * w;

    T val = 0;
    if (ix >= pad)
    {
        const T* pl = left  + iy * w + ix;
        const T* pr = right + iy * w + ix - pad;
        for (int32_t i = 0; i < c; i++)
        {
            val += *pl * (*pr);
            pl  += stride;
            pr  += stride;
        }
    }

    // Disparity feature maps are arranged from to min to max.
    size_t idst = pad * h * w + iy * w + ix;
    dst[idst] = val;
}

// FP16, NC2HW2 kernel.
__global__ void corrCostVolumeFP16NC2HW2Kernel(const float* left, const float* right, int32_t c, int32_t h, int32_t w, int32_t disp, float* dst)
{
    assert(left  != nullptr);
    assert(right != nullptr);
    assert(dst   != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= w || iy >= h)
        return;

    uint32_t pad = 2 * blockIdx.z;
    assert(pad < disp);
    size_t stride = h * w;

    // REVIEW alexeyk: using FP32 arithmetic for better precision. FP16 works fine too
    // but does not give any perf increase and causes slight loss in accuracy.
    // __half2 val1{0, 0};
    // __half2 val2{0, 0};
    float val1 = 0;
    float val2 = 0;
    if (ix >= pad)
    {
        const float* pl = left  + iy * w + ix;
        const float* pr = right + iy * w + ix - pad;
        for (int32_t i = 0; i < (c + 1) / 2; i++)
        {
            // auto l  = *(__half2*)pl;
            // auto r1 = *(__half2*)pr;
            // auto r2 = ix >= pad + 1 ? *(__half2*)(pr - 1) : __half2{0, 0};
            // val1 = __hfma2(l, r1, val1);
            // val2 = __hfma2(l, r2, val2);
            auto l  = __half22float2(*(__half2*)pl);
            auto r1 = __half22float2(*(__half2*)pr);
            auto r2 = ix >= pad + 1 ? __half22float2(*(__half2*)(pr - 1)) : float2{0, 0};
            val1 += l.x * r1.x + l.y * r1.y;
            val2 += l.x * r2.x + l.y * r2.y;
            pl  += stride;
            pr  += stride;
        }
    }

    // Disparity feature maps are arranged from to min to max.
    size_t idst = blockIdx.z * h * w + iy * w + ix;
    // auto val  = __half2(__hadd(val1.x, val1.y), __hadd(val2.x, val2.y));
    auto val  = __half2((__half)val1, (__half)val2);
    dst[idst] = *(float*)&val;
}

template<>
cudaError_t CudaKernels::computeCorrCostVolume(DataType data_type, const float* left, const float* right, Dims in_dims, 
                                               float* cost_vol, Dims out_dims, cudaStream_t stream)
{
    assert(data_type == DataType::kFLOAT || data_type == DataType::kHALF);
    assert(in_dims.nbDims  == 3);
    assert(out_dims.nbDims == 3);

    if (data_type == DataType::kFLOAT)
    {
        dim3 b_dim{16, 16, 1};
        dim3 g_dim;
        g_dim.x = getBlockCount(in_dims.d[2],  b_dim.x);
        g_dim.y = getBlockCount(in_dims.d[1],  b_dim.y);
        // Each block handles a particular disparity.
        g_dim.z = out_dims.d[0];

        corrCostVolumeKernel<<<g_dim, b_dim, 0, stream>>>(left, right, in_dims.d[0], in_dims.d[1], in_dims.d[2], out_dims.d[0],
                                                          cost_vol);
        CHECKK(stream);
    }
    else if (data_type == DataType::kHALF)
    {
        dim3 b_dim{16, 16, 1};
        dim3 g_dim;
        g_dim.x = getBlockCount(in_dims.d[2], b_dim.x);
        g_dim.y = getBlockCount(in_dims.d[1], b_dim.y);
        // Each block handles 2 disparity values.
        g_dim.z = (out_dims.d[0] + 1) / 2;

        corrCostVolumeFP16NC2HW2Kernel<<<g_dim, b_dim, 0, stream>>>(left, right, in_dims.d[0], in_dims.d[1], in_dims.d[2], out_dims.d[0],
                                                                    cost_vol);
        CHECKK(stream);
    }
    return cudaSuccess;
}

// -----------------------------------------------------------------
// Some convolution-related kernels.
// -----------------------------------------------------------------
template<typename T>
__global__ void addDBiasTo3DConvKernel(const T* bias, int32_t c, int32_t d, int32_t h, int32_t w, T* conv)
{
    assert(bias != nullptr);
    assert(conv != nullptr);

    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= w || iy >= h || iz >= d * c)
        return;

    int32_t cur_d     = iz % d;
    const size_t idst = iz * h * w + iy * w + ix;

    conv[idst] += bias[cur_d];
}

template<>
cudaError_t CudaKernels::addDBiasTo3DConv(const float* bias, Dims bias_dims, float* conv, Dims conv_dims, cudaStream_t stream)
{
    assert(bias_dims.nbDims == 5);
    assert(conv_dims.nbDims == 4);
    // REVIEW alexeyk: minibatch size 1 for now.
    assert(bias_dims.d[0] == 1);
    assert(bias_dims.d[2] == conv_dims.d[1]);
    UNUSEDR(bias_dims);

    dim3 b_dim{16, 16, 1};
    dim3 g_dim;
    g_dim.x = getBlockCount(conv_dims.d[3], b_dim.x);
    g_dim.y = getBlockCount(conv_dims.d[2], b_dim.y);
    g_dim.z = getBlockCount(conv_dims.d[0] * conv_dims.d[1], b_dim.z);
    // REVIEW alexeyk: no block striding for now.
    assert(g_dim.y <= kMaxGridSizeY);
    assert(g_dim.z <= kMaxGridSizeZ);
    UNUSEDR(kMaxGridSizeY);
    UNUSEDR(kMaxGridSizeZ);

    addDBiasTo3DConvKernel<<<g_dim, b_dim, 0, stream>>>(bias, conv_dims.d[0], conv_dims.d[1], conv_dims.d[2], conv_dims.d[3], conv);
    CHECKK(stream);

    return cudaSuccess;
}

// -----------------------------------------------------------------
// Conversion kernels.
// -----------------------------------------------------------------
__global__ void fp32Tofp16Kernel(const float* src, uint16_t* dst, size_t size)
{
    assert(src != nullptr);
    assert(dst != nullptr);

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    __half val(src[tid]);
    dst[tid] = *(uint16_t*)&val;
}

cudaError_t CudaKernels::fp32Tofp16(const float* src, uint16_t* dst, size_t size, cudaStream_t stream)
{
    dim3 b_dim{256, 1, 1};
    dim3 g_dim;
    g_dim.x = getBlockCount(size, b_dim.x);

    fp32Tofp16Kernel<<<g_dim, b_dim, 0, stream>>>(src, dst, size);
    CHECKK(stream);

    return cudaSuccess;
}

__global__ void fp16Tofp32Kernel(const uint16_t* src, float* dst, size_t size)
{
    assert(src != nullptr);
    assert(dst != nullptr);

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    dst[tid] = (float)(*(__half*)(src + tid));
}

cudaError_t CudaKernels::fp16Tofp32(const uint16_t* src, float* dst, size_t size, cudaStream_t stream)
{
    dim3 b_dim{256, 1, 1};
    dim3 g_dim;
    g_dim.x = getBlockCount(size, b_dim.x);

    fp16Tofp32Kernel<<<g_dim, b_dim, 0, stream>>>(src, dst, size);
    CHECKK(stream);

    return cudaSuccess;
}

} }