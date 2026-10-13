/*
 * mudart.c - CUDA runtime API shim on top of muda.
 */

#include "mudart.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Device                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaGetDeviceCount(int *count)
{
    return mudaGetDeviceCount(count);
}

cudaError_t cudaGetDevice(int *device)
{
    return mudaGetDevice(device);
}

cudaError_t cudaSetDevice(int device)
{
    return mudaSetDevice(device);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (!prop)
        return mudaErrorInvalidValue;

    mudaDeviceProp m;
    cudaError_t e = mudaGetDeviceProperties(&m, device);
    if (e != cudaSuccess)
        return e;

    memset(prop, 0, sizeof(*prop));
    snprintf(prop->name, sizeof(prop->name), "%s", m.name);
    prop->totalGlobalMem         = m.totalGlobalMem;
    prop->sharedMemPerBlock      = m.sharedMemPerBlock;
    prop->regsPerBlock           = m.maxRegistersPerBlock;
    prop->warpSize               = m.warpSize;
    prop->memPitch               = m.memPitch;
    prop->maxThreadsPerBlock     = m.maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        prop->maxThreadsDim[i] = m.maxThreadsDim[i];
        prop->maxGridSize[i]   = m.maxGridSize[i];
    }
    prop->clockRate              = m.clockRateKHz;
    prop->totalConstMem          = 65536;
    prop->major                  = m.computeCapabilityMajor;
    prop->minor                  = m.computeCapabilityMinor;
    prop->multiProcessorCount    = m.multiProcessorCount;
    prop->concurrentKernels      = m.concurrentKernels;
    prop->memoryClockRate        = m.memoryClockRateKHz;
    prop->memoryBusWidth         = m.memoryBusWidth;
    prop->pciBusID               = m.pciBusID;
    prop->pciDeviceID            = m.pciDeviceID;
    prop->pciDomainID            = m.pciDomainID;
    prop->canMapHostMemory       = 1;
    prop->maxThreadsPerMultiProcessor = m.maxThreadsPerBlock;
    prop->computeMode            = 0; /* cudaComputeModeDefault */
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void)
{
    return mudaDeviceSynchronize();
}

cudaError_t cudaDeviceReset(void)
{
    return mudaDeviceReset();
}

/* ------------------------------------------------------------------ */
/* Memory                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    return mudaMalloc(devPtr, size);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    return mudaMallocHost(ptr, size);
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    (void)flags;
    return mudaMallocHost(ptr, size);
}

cudaError_t cudaFree(void *devPtr)
{
    return mudaFree(devPtr);
}

cudaError_t cudaFreeHost(void *ptr)
{
    return mudaFreeHost(ptr);
}

cudaError_t cudaMemGetInfo(size_t *freeMem, size_t *totalMem)
{
    return mudaMemGetInfo(freeMem, totalMem);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    return mudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream)
{
    return mudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind kind)
{
    return mudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    return mudaMemset(devPtr, value, count);
}

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    return mudaStreamCreate(stream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    return mudaStreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    return mudaStreamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return mudaStreamSynchronize(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    return mudaStreamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags)
{
    return mudaStreamWaitEvent(stream, event, flags);
}

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return mudaEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    return mudaEventCreateWithFlags(event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    return mudaEventDestroy(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return mudaEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return mudaEventSynchronize(event);
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return mudaEventQuery(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    return mudaEventElapsedTime(ms, start, end);
}

/* ------------------------------------------------------------------ */
/* Launch bridge                                                      */
/* ------------------------------------------------------------------ */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream)
{
    return mudaLaunchKernel((mudaFunction_t)(uintptr_t)func,
                            gridDim.x, gridDim.y, gridDim.z,
                            blockDim.x, blockDim.y, blockDim.z,
                            (unsigned)sharedMem, stream, args, NULL);
}

/* Legacy configure/setup/launch. */
#define MUDART_MAX_ARGS 64

static struct {
    int            active;
    unsigned       gx, gy, gz;
    unsigned       bx, by, bz;
    size_t         shared;
    cudaStream_t   stream;
    unsigned char *buf;
    size_t         cap;
    void          *argptr[MUDART_MAX_ARGS];
    int            nargs;
} g_cfg;

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                              size_t sharedMem, cudaStream_t stream)
{
    g_cfg.active = 1;
    g_cfg.gx = gridDim.x; g_cfg.gy = gridDim.y; g_cfg.gz = gridDim.z;
    g_cfg.bx = blockDim.x; g_cfg.by = blockDim.y; g_cfg.bz = blockDim.z;
    g_cfg.shared = sharedMem;
    g_cfg.stream = stream;
    g_cfg.nargs = 0;
    return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    if (!g_cfg.active)
        return mudaErrorInvalidValue;
    if (g_cfg.nargs >= MUDART_MAX_ARGS)
        return mudaErrorInvalidValue;

    size_t needed = offset + size;
    if (needed > g_cfg.cap) {
        size_t cap = g_cfg.cap ? g_cfg.cap : 256;
        while (cap < needed)
            cap *= 2;
        unsigned char *nb = (unsigned char *)realloc(g_cfg.buf, cap);
        if (!nb)
            return mudaErrorOutOfMemory;
        g_cfg.buf = nb;
        g_cfg.cap = cap;
    }
    if (arg && size)
        memcpy(g_cfg.buf + offset, arg, size);
    g_cfg.argptr[g_cfg.nargs++] = g_cfg.buf + offset;
    return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func)
{
    if (!g_cfg.active)
        return mudaErrorInvalidValue;

    cudaError_t e = mudaLaunchKernel((mudaFunction_t)(uintptr_t)func,
                                     g_cfg.gx, g_cfg.gy, g_cfg.gz,
                                     g_cfg.bx, g_cfg.by, g_cfg.bz,
                                     (unsigned)g_cfg.shared, g_cfg.stream,
                                     g_cfg.argptr, NULL);
    g_cfg.active = 0;
    return e;
}

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */
cudaError_t cudaGetLastError(void)
{
    return mudaGetLastError();
}

cudaError_t cudaPeekAtLastError(void)
{
    return mudaPeekAtLastError();
}

const char *cudaGetErrorName(cudaError_t error)
{
    return mudaGetErrorName(error);
}

const char *cudaGetErrorString(cudaError_t error)
{
    return mudaGetErrorString(error);
}
