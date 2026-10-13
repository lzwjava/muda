/*
 * muda.h - minimal CUDA-compatible runtime for the NVIDIA Tesla M40.
 *
 * muda implements a small, self-contained subset of the CUDA driver API.
 * It deliberately does NOT link or load the NVIDIA CUDA driver (libcuda).
 * Instead, a pluggable "muda driver" backend is responsible for talking to
 * the hardware.  The default backend translates CUDA-style kernels to
 * OpenCL C at runtime and dispatches them through the system OpenCL ICD.
 *
 * The API mirrors cuda.h/cu*.h closely enough that porting small programs
 * is mostly a matter of s/cuda/muda/ and s/cu/muda/.
 */
#ifndef MUDA_H
#define MUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MUDA_VERSION_MAJOR 0
#define MUDA_VERSION_MINOR 1
#define MUDA_VERSION_PATCH 0

/* ------------------------------------------------------------------ */
/* Error codes                                                        */
/* ------------------------------------------------------------------ */
typedef enum mudaError {
    mudaSuccess                   = 0,
    mudaErrorInvalidValue         = 1,
    mudaErrorOutOfMemory          = 2,
    mudaErrorNotInitialized       = 3,
    mudaErrorDeinitialized        = 4,
    mudaErrorNoDevice             = 100,
    mudaErrorInvalidDevice        = 101,
    mudaErrorInvalidContext       = 201,
    mudaErrorInvalidHandle        = 400,
    mudaErrorNotFound             = 500,
    mudaErrorNotSupported         = 801,
    mudaErrorLaunchFailure        = 719,
    mudaErrorUnknown              = 999
} mudaError_t;

/* ------------------------------------------------------------------ */
/* Memory copy kinds (kept numerically identical to CUDA)             */
/* ------------------------------------------------------------------ */
typedef enum mudaMemcpyKind {
    mudaMemcpyHostToHost     = 0,
    mudaMemcpyHostToDevice   = 1,
    mudaMemcpyDeviceToHost   = 2,
    mudaMemcpyDeviceToDevice = 3,
    mudaMemcpyDefault        = 4
} mudaMemcpyKind;

/* ------------------------------------------------------------------ */
/* Device properties                                                  */
/* ------------------------------------------------------------------ */
typedef struct mudaDeviceProp {
    char   name[256];
    char   driverVersion[64];
    char   deviceVersion[64];
    int    computeCapabilityMajor;
    int    computeCapabilityMinor;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    size_t sharedMemPerMultiprocessor;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    multiProcessorCount;
    int    warpSize;
    int    clockRateKHz;
    int    memoryClockRateKHz;
    int    memoryBusWidth;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
} mudaDeviceProp;

/* Opaque handles. */
typedef void *mudaContext_t;
typedef void *mudaStream_t;
typedef void *mudaEvent_t;
typedef void *mudaModule_t;
typedef void *mudaFunction_t;

/* ------------------------------------------------------------------ */
/* Runtime / context management                                       */
/* ------------------------------------------------------------------ */
mudaError_t mudaInit(unsigned int flags);
mudaError_t mudaShutdown(void);
const char *mudaGetDriverName(void);

mudaError_t mudaGetDeviceCount(int *count);
mudaError_t mudaGetDeviceProperties(mudaDeviceProp *prop, int device);

mudaError_t mudaCtxCreate(mudaContext_t *ctx, unsigned int flags, int device);
mudaError_t mudaCtxDestroy(mudaContext_t ctx);
mudaError_t mudaCtxPushCurrent(mudaContext_t ctx);
mudaError_t mudaCtxPopCurrent(mudaContext_t *ctx);
mudaError_t mudaCtxGetCurrent(mudaContext_t *ctx);

/* ------------------------------------------------------------------ */
/* Device memory                                                      */
/* ------------------------------------------------------------------ */
mudaError_t mudaMalloc(void **devPtr, size_t size);
mudaError_t mudaFree(void *devPtr);
mudaError_t mudaMemcpy(void *dst, const void *src, size_t count, mudaMemcpyKind kind);
mudaError_t mudaMemcpyAsync(void *dst, const void *src, size_t count,
                            mudaMemcpyKind kind, mudaStream_t stream);
mudaError_t mudaMemset(void *devPtr, int value, size_t count);

/* ------------------------------------------------------------------ */
/* Modules and kernels                                                */
/* ------------------------------------------------------------------ */

/* Load a module from CUDA-style source text.  The text is translated to
 * OpenCL C by the active driver and compiled for the device. */
mudaError_t mudaModuleLoadData(mudaModule_t *module, const char *source);
mudaError_t mudaModuleLoad(mudaModule_t *module, const char *path);
mudaError_t mudaModuleUnload(mudaModule_t module);
mudaError_t mudaModuleGetFunction(mudaFunction_t *func, mudaModule_t module,
                                  const char *name);

/* Launch, mirroring cuLaunchKernel.  kernelParams is an array of pointers,
 * one per kernel argument, each pointing at the argument value. */
mudaError_t mudaLaunchKernel(mudaFunction_t f,
                             unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
                             unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                             unsigned sharedMemBytes, mudaStream_t stream,
                             void **kernelParams, void **extra);

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
mudaError_t mudaStreamCreate(mudaStream_t *stream);
mudaError_t mudaStreamDestroy(mudaStream_t stream);
mudaError_t mudaStreamSynchronize(mudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
mudaError_t mudaEventCreate(mudaEvent_t *event);
mudaError_t mudaEventDestroy(mudaEvent_t event);
mudaError_t mudaEventRecord(mudaEvent_t event, mudaStream_t stream);
mudaError_t mudaEventSynchronize(mudaEvent_t event);
mudaError_t mudaEventElapsedTime(float *ms, mudaEvent_t start, mudaEvent_t end);

/* ------------------------------------------------------------------ */
/* Misc                                                               */
/* ------------------------------------------------------------------ */
mudaError_t mudaDeviceSynchronize(void);
mudaError_t mudaGetLastError(void);
const char *mudaGetErrorName(mudaError_t error);
const char *mudaGetErrorString(mudaError_t error);

#ifdef __cplusplus
}
#endif

#endif /* MUDA_H */
