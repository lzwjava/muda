/*
 * mudart.h - CUDA runtime API shim on top of muda.
 *
 * This header lets host-side CUDA runtime code compile and run against muda
 * with minimal edits: cudaMalloc, cudaMemcpy, cudaStreams, cudaEvents,
 * cudaDevice* and the launch bridge all map onto the corresponding muda
 * calls.
 *
 * Kernels are still compiled from CUDA-style source by the muda driver (there
 * is no nvcc anywhere in the loop).  Load a module and look up a kernel with
 * mudaModuleLoadData()/mudaModuleGetFunction(), then launch it with either
 *
 *     cudaLaunchKernel((const void *)fn, grid, block, args, shared, stream);
 *
 * or the legacy configure/setup/launch sequence:
 *
 *     cudaConfigureCall(grid, block, shared, stream);
 *     cudaSetupArgument(&a, sizeof(a), 0);
 *     cudaSetupArgument(&b, sizeof(b), sizeof(a));
 *     cudaLaunch((const void *)fn);
 */
#ifndef MUDART_H
#define MUDART_H

#include "muda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Types                                                              */
/* ------------------------------------------------------------------ */
typedef mudaError_t    cudaError_t;
typedef mudaMemcpyKind cudaMemcpyKind;
typedef mudaStream_t   cudaStream_t;
typedef mudaEvent_t    cudaEvent_t;

typedef struct dim3 { unsigned int x, y, z; } dim3;

static inline dim3 make_dim3(unsigned int x, unsigned int y, unsigned int z)
{
    dim3 d; d.x = x; d.y = y; d.z = z; return d;
}
static inline dim3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
    return make_dim3(x, y, z);
}

typedef struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    size_t totalConstMem;
    int    major;
    int    minor;
    size_t textureAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
    int    kernelExecTimeoutEnabled;
    int    integrated;
    int    canMapHostMemory;
    int    computeMode;
    int    concurrentKernels;
    int    ECCEnabled;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    tccDriver;
    int    unifiedAddressing;
    int    memoryClockRate;
    int    memoryBusWidth;
    int    l2CacheSize;
    int    maxThreadsPerMultiProcessor;
} cudaDeviceProp;

/* ------------------------------------------------------------------ */
/* Error / flag aliases                                               */
/* ------------------------------------------------------------------ */
#define cudaSuccess                      mudaSuccess
#define cudaErrorInvalidValue            mudaErrorInvalidValue
#define cudaErrorMemoryAllocation        mudaErrorOutOfMemory
#define cudaErrorInitializationError     mudaErrorNotInitialized
#define cudaErrorProfilerDisabled        mudaErrorProfilerDisabled
#define cudaErrorNoDevice                mudaErrorNoDevice
#define cudaErrorInvalidDevice           mudaErrorInvalidDevice
#define cudaErrorInvalidHandle           mudaErrorInvalidHandle
#define cudaErrorFileNotFound            mudaErrorNotFound
#define cudaErrorNotReady                mudaErrorNotReady
#define cudaErrorLaunchFailure           mudaErrorLaunchFailure
#define cudaErrorNotSupported            mudaErrorNotSupported
#define cudaErrorInvalidDevicePointer    mudaErrorInvalidValue
#define cudaErrorInvalidMemcpyDirection  mudaErrorInvalidValue
#define cudaErrorUnknown                  mudaErrorUnknown

#define cudaMemcpyHostToHost             mudaMemcpyHostToHost
#define cudaMemcpyHostToDevice           mudaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost           mudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice         mudaMemcpyDeviceToDevice
#define cudaMemcpyDefault                mudaMemcpyDefault

#define cudaStreamDefault                mudaStreamDefault
#define cudaStreamNonBlocking            mudaStreamNonBlocking
#define cudaEventDefault                 mudaEventDefault
#define cudaEventDisableTiming           mudaEventDisableTiming

#define cudaHostAllocDefault             0x00
#define cudaHostAllocPortable            0x01
#define cudaHostAllocMapped              0x02
#define cudaHostAllocWriteCombined       0x04

/* ------------------------------------------------------------------ */
/* Device management                                                  */
/* ------------------------------------------------------------------ */
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);

/* ------------------------------------------------------------------ */
/* Memory                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemGetInfo(size_t *freeMem, size_t *totalMem);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/* ------------------------------------------------------------------ */
/* Launch bridge                                                      */
/* ------------------------------------------------------------------ */

/* func is a mudaFunction_t obtained from mudaModuleGetFunction(). */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream);

/* Legacy launch sequence. */
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                              size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void *func);

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorName(cudaError_t error);
const char *cudaGetErrorString(cudaError_t error);

/* Convenience macros for building argument packs. */
#define CUDA_ARGS(...) ((void *[]){ __VA_ARGS__ })
#define CUDA_LAUNCH(fn, grid, block, stream, ...)                            \
    cudaLaunchKernel((const void *)(fn), (grid), (block),                    \
                     CUDA_ARGS(__VA_ARGS__), 0, (stream))

#ifdef __cplusplus
}
#endif

#endif /* MUDART_H */
