/*
 * muda.h - minimal CUDA-compatible runtime for the NVIDIA Tesla M40.
 *
 * muda implements a self-contained subset of the CUDA runtime/driver API.
 * It deliberately does NOT link or load the NVIDIA CUDA driver (libcuda).
 * Instead, a pluggable "muda driver" backend is responsible for talking to
 * the hardware.  The default backend translates CUDA-style kernels to
 * OpenCL C at run time and dispatches them through the system OpenCL ICD.
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
#define MUDA_VERSION_MINOR 2
#define MUDA_VERSION_PATCH 0

/* Version integer in the same layout as CUDA: major*1000 + minor*10. */
#define MUDA_VERSION (MUDA_VERSION_MAJOR * 1000 + MUDA_VERSION_MINOR * 10)

/* ------------------------------------------------------------------ */
/* Error codes (kept numerically close to CUDA where practical)       */
/* ------------------------------------------------------------------ */
typedef enum mudaError {
    mudaSuccess                   = 0,
    mudaErrorInvalidValue         = 1,
    mudaErrorOutOfMemory          = 2,
    mudaErrorNotInitialized       = 3,
    mudaErrorDeinitialized        = 4,
    mudaErrorProfilerDisabled     = 5,
    mudaErrorProfilerNotInitialized = 6,
    mudaErrorProfilerAlreadyStarted = 7,
    mudaErrorProfilerAlreadyStopped = 8,
    mudaErrorNoDevice             = 100,
    mudaErrorInvalidDevice        = 101,
    mudaErrorInvalidContext       = 201,
    mudaErrorInvalidHandle        = 400,
    mudaErrorNotFound             = 500,
    mudaErrorNotReady             = 600,
    mudaErrorLaunchFailure        = 719,
    mudaErrorNotSupported         = 801,
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
/* Memory types / pointer attributes                                  */
/* ------------------------------------------------------------------ */
typedef enum mudaMemoryType {
    mudaMemoryTypeUnregistered = 0,
    mudaMemoryTypeHost         = 1,
    mudaMemoryTypeDevice       = 2,
    mudaMemoryTypeManaged      = 3
} mudaMemoryType;

typedef struct mudaPointerAttributes {
    mudaMemoryType type;
    int            device;
    void          *devicePointer;
    void          *hostPointer;
} mudaPointerAttributes;

/* ------------------------------------------------------------------ */
/* Device attributes                                                  */
/* ------------------------------------------------------------------ */
typedef enum mudaDeviceAttr {
    mudaDevAttrMaxThreadsPerBlock       = 1,
    mudaDevAttrMaxBlockDimX             = 2,
    mudaDevAttrMaxBlockDimY             = 3,
    mudaDevAttrMaxBlockDimZ             = 4,
    mudaDevAttrMaxGridDimX              = 5,
    mudaDevAttrMaxGridDimY              = 6,
    mudaDevAttrMaxGridDimZ              = 7,
    mudaDevAttrMaxSharedMemoryPerBlock  = 8,
    mudaDevAttrTotalConstantMemory      = 9,
    mudaDevAttrWarpSize                 = 10,
    mudaDevAttrMaxPitch                 = 11,
    mudaDevAttrMaxRegistersPerBlock     = 12,
    mudaDevAttrClockRate                = 13,
    mudaDevAttrMultiProcessorCount      = 16,
    mudaDevAttrConcurrentKernels        = 31,
    mudaDevAttrMemoryClockRate          = 36,
    mudaDevAttrGlobalMemoryBusWidth     = 37,
    mudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    mudaDevAttrComputeCapabilityMajor   = 75,
    mudaDevAttrComputeCapabilityMinor   = 76,
    mudaDevAttrPciBusId                 = 60,
    mudaDevAttrPciDeviceId              = 61,
    mudaDevAttrPciDomainId              = 62
} mudaDeviceAttr;

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
    int    maxRegistersPerBlock;
    int    multiProcessorCount;
    int    warpSize;
    int    clockRateKHz;
    int    memoryClockRateKHz;
    int    memoryBusWidth;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    concurrentKernels;
} mudaDeviceProp;

/* ------------------------------------------------------------------ */
/* Function (kernel) attributes                                       */
/* ------------------------------------------------------------------ */
typedef struct mudaFuncAttributes {
    int    maxThreadsPerBlock;
    int    numRegs;
    size_t sharedSizeBytes;
    size_t localSizeBytes;
    size_t constSizeBytes;
    int    binaryVersion;
    int    cacheModeCA;
    int    maxDynamicSharedSizeBytes;
    int    preferredShmemCarveout;
} mudaFuncAttributes;

/* ------------------------------------------------------------------ */
/* Opaque handles                                                     */
/* ------------------------------------------------------------------ */
typedef void *mudaContext_t;
typedef void *mudaStream_t;
typedef void *mudaEvent_t;
typedef void *mudaModule_t;
typedef void *mudaFunction_t;

/* ------------------------------------------------------------------ */
/* Runtime / device management                                        */
/* ------------------------------------------------------------------ */
mudaError_t mudaInit(unsigned int flags);
mudaError_t mudaShutdown(void);
const char *mudaGetDriverName(void);

mudaError_t mudaGetDeviceCount(int *count);
mudaError_t mudaSetDevice(int device);
mudaError_t mudaGetDevice(int *device);
mudaError_t mudaGetDeviceProperties(mudaDeviceProp *prop, int device);
mudaError_t mudaDeviceGetName(char *name, int len, int device);
mudaError_t mudaDeviceGetAttribute(int *value, mudaDeviceAttr attr, int device);
mudaError_t mudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
mudaError_t mudaDeviceReset(void);

mudaError_t mudaDriverGetVersion(int *version);
mudaError_t mudaRuntimeGetVersion(int *version);

/* ------------------------------------------------------------------ */
/* Contexts                                                           */
/* ------------------------------------------------------------------ */
mudaError_t mudaCtxCreate(mudaContext_t *ctx, unsigned int flags, int device);
mudaError_t mudaCtxDestroy(mudaContext_t ctx);
mudaError_t mudaCtxPushCurrent(mudaContext_t ctx);
mudaError_t mudaCtxPopCurrent(mudaContext_t *ctx);
mudaError_t mudaCtxGetCurrent(mudaContext_t *ctx);
mudaError_t mudaCtxGetDevice(int *device);

/* ------------------------------------------------------------------ */
/* Device memory                                                      */
/* ------------------------------------------------------------------ */
mudaError_t mudaMalloc(void **devPtr, size_t size);
mudaError_t mudaFree(void *devPtr);
mudaError_t mudaMallocHost(void **ptr, size_t size);
mudaError_t mudaFreeHost(void *ptr);
mudaError_t mudaHostGetDevicePointer(void **devPtr, void *hostPtr, unsigned int flags);
mudaError_t mudaMemGetInfo(size_t *freeBytes, size_t *totalBytes);
mudaError_t mudaPointerGetAttributes(mudaPointerAttributes *attr, const void *ptr);

mudaError_t mudaMemcpy(void *dst, const void *src, size_t count, mudaMemcpyKind kind);
mudaError_t mudaMemcpyAsync(void *dst, const void *src, size_t count,
                            mudaMemcpyKind kind, mudaStream_t stream);
mudaError_t mudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, mudaMemcpyKind kind);
mudaError_t mudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                              size_t width, size_t height, mudaMemcpyKind kind,
                              mudaStream_t stream);
mudaError_t mudaMemset(void *devPtr, int value, size_t count);
mudaError_t mudaMemsetD8(void *devPtr, unsigned char value, size_t count);
mudaError_t mudaMemsetD32(void *devPtr, unsigned int value, size_t count);

/* Advisory calls for managed memory.  No-ops on this driver. */
mudaError_t mudaMemAdvise(const void *devPtr, size_t count, int advice, int device);
mudaError_t mudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                                 mudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Modules and kernels                                                */
/* ------------------------------------------------------------------ */
mudaError_t mudaModuleLoadData(mudaModule_t *module, const char *source);
mudaError_t mudaModuleLoad(mudaModule_t *module, const char *path);
mudaError_t mudaModuleUnload(mudaModule_t module);
mudaError_t mudaModuleGetFunction(mudaFunction_t *func, mudaModule_t module,
                                  const char *name);
mudaError_t mudaFuncGetAttributes(mudaFuncAttributes *attr, mudaFunction_t func);
mudaError_t mudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          mudaFunction_t func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize);

/* Launch, mirroring cuLaunchKernel.  kernelParams is an array of pointers,
 * one per kernel argument, each pointing at the argument value.  When
 * kernelParams is NULL, extra may carry a packed argument buffer. */
mudaError_t mudaLaunchKernel(mudaFunction_t f,
                             unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
                             unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                             unsigned sharedMemBytes, mudaStream_t stream,
                             void **kernelParams, void **extra);

/* CUDA-style packed launch configuration. */
typedef struct mudaLaunchConfig {
    unsigned     gridDimX, gridDimY, gridDimZ;
    unsigned     blockDimX, blockDimY, blockDimZ;
    unsigned     sharedMemBytes;
    mudaStream_t stream;
    /* attributes are not supported yet */
    const int   *attrs;
    unsigned     numAttrs;
} mudaLaunchConfig;

#define mudaLaunchConfigDefault                                                 \
    { 1, 1, 1, 1, 1, 1, 0, NULL, NULL, 0 }

mudaError_t mudaLaunchKernelEx(const mudaLaunchConfig *config, mudaFunction_t f,
                               void **kernelParams, void **extra);

/* Packed-argument sentinels, matching CU_LAUNCH_PARAM_*. */
#define MUDA_LAUNCH_PARAM_END            ((void *)0x00)
#define MUDA_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define MUDA_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
typedef enum mudaStreamFlags {
    mudaStreamDefault     = 0x00,
    mudaStreamNonBlocking = 0x01
} mudaStreamFlags;

mudaError_t mudaStreamCreate(mudaStream_t *stream);
mudaError_t mudaStreamCreateWithFlags(mudaStream_t *stream, unsigned int flags);
mudaError_t mudaStreamDestroy(mudaStream_t stream);
mudaError_t mudaStreamSynchronize(mudaStream_t stream);
mudaError_t mudaStreamQuery(mudaStream_t stream);
mudaError_t mudaStreamWaitEvent(mudaStream_t stream, mudaEvent_t event,
                                unsigned int flags);

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
typedef enum mudaEventFlags {
    mudaEventDefault        = 0x00,
    mudaEventDisableTiming  = 0x01
} mudaEventFlags;

mudaError_t mudaEventCreate(mudaEvent_t *event);
mudaError_t mudaEventCreateWithFlags(mudaEvent_t *event, unsigned int flags);
mudaError_t mudaEventDestroy(mudaEvent_t event);
mudaError_t mudaEventRecord(mudaEvent_t event, mudaStream_t stream);
mudaError_t mudaEventSynchronize(mudaEvent_t event);
mudaError_t mudaEventQuery(mudaEvent_t event);
mudaError_t mudaEventElapsedTime(float *ms, mudaEvent_t start, mudaEvent_t end);

/* ------------------------------------------------------------------ */
/* Misc                                                               */
/* ------------------------------------------------------------------ */
mudaError_t mudaDeviceSynchronize(void);
mudaError_t mudaGetLastError(void);
mudaError_t mudaPeekAtLastError(void);
const char *mudaGetErrorName(mudaError_t error);
const char *mudaGetErrorString(mudaError_t error);

#ifdef __cplusplus
}
#endif

#endif /* MUDA_H */
