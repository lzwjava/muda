/*
 * muda.c - public API dispatcher.
 *
 * Holds the process-wide runtime state (selected driver, current context,
 * allocation registry) and forwards each call to the active muda driver.
 */

#include "muda_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>

static pthread_mutex_t    g_lock = PTHREAD_MUTEX_INITIALIZER;
static const muda_driver *g_drv = NULL;
static int                 g_initialized = 0;
static int                 g_last_error = mudaSuccess;
static int                 g_preferred_device = 0;

static muda_context_impl *g_current = NULL;
static muda_context_impl *g_primary = NULL;

/* ------------------------------------------------------------------ */
/* Allocation registry (for memGetInfo / pointerGetAttributes)        */
/* ------------------------------------------------------------------ */
typedef struct {
    void  *ptr;
    size_t size;
    int    type;    /* mudaMemoryType */
    int    device;
} alloc_rec;

static alloc_rec *g_allocs = NULL;
static int        g_nallocs = 0;
static int        g_acap = 0;

static void reg_add(void *ptr, size_t size, int type, int device)
{
    if (g_nallocs == g_acap) {
        g_acap = g_acap ? g_acap * 2 : 64;
        g_allocs = (alloc_rec *)realloc(g_allocs, (size_t)g_acap * sizeof(alloc_rec));
    }
    if (!g_allocs)
        return;
    g_allocs[g_nallocs].ptr = ptr;
    g_allocs[g_nallocs].size = size;
    g_allocs[g_nallocs].type = type;
    g_allocs[g_nallocs].device = device;
    g_nallocs++;
}

static int reg_find(const void *ptr, alloc_rec *out)
{
    for (int i = 0; i < g_nallocs; i++) {
        if (g_allocs[i].ptr == ptr) {
            if (out)
                *out = g_allocs[i];
            return 1;
        }
    }
    return 0;
}

static void reg_remove(const void *ptr)
{
    for (int i = 0; i < g_nallocs; i++) {
        if (g_allocs[i].ptr == ptr) {
            g_allocs[i] = g_allocs[g_nallocs - 1];
            g_nallocs--;
            return;
        }
    }
}

static size_t reg_device_used(void)
{
    size_t used = 0;
    for (int i = 0; i < g_nallocs; i++)
        if (g_allocs[i].type == mudaMemoryTypeDevice)
            used += g_allocs[i].size;
    return used;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */
static mudaError_t fail(mudaError_t e)
{
    g_last_error = e;
    return e;
}

static mudaError_t ok(void)
{
    g_last_error = mudaSuccess;
    return mudaSuccess;
}

static int file_read(const char *path, char **out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return -1;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *out = buf;
    return 0;
}

static muda_context_impl *make_context(int device)
{
    muda_context_impl *impl = (muda_context_impl *)calloc(1, sizeof(*impl));
    if (!impl)
        return NULL;
    impl->drv = g_drv;
    impl->device = device;
    if (g_drv->ctx_create(&impl->drv_ctx, device) != mudaSuccess) {
        free(impl);
        return NULL;
    }
    return impl;
}

/* Returns the context to use, lazily creating a primary one. */
static muda_context_impl *get_ctx(void)
{
    if (g_current)
        return g_current;
    if (g_primary)
        return g_primary;

    pthread_mutex_lock(&g_lock);
    if (!g_primary && g_drv && g_initialized)
        g_primary = make_context(g_preferred_device);
    muda_context_impl *result = g_current ? g_current : g_primary;
    pthread_mutex_unlock(&g_lock);
    return result;
}

/* ------------------------------------------------------------------ */
/* Runtime / devices                                                  */
/* ------------------------------------------------------------------ */
mudaError_t mudaInit(unsigned int flags)
{
    (void)flags;
    pthread_mutex_lock(&g_lock);
    if (g_initialized) {
        pthread_mutex_unlock(&g_lock);
        return ok();
    }

    const char *want = getenv("MUDA_DRIVER");
    if (want && strcmp(want, "opencl") != 0) {
        fprintf(stderr, "muda: unknown driver '%s' (only 'opencl' is available)\n", want);
        pthread_mutex_unlock(&g_lock);
        return fail(mudaErrorNotSupported);
    }

    g_drv = &muda_driver_opencl;
    if (g_drv->init() != mudaSuccess) {
        g_drv = NULL;
        pthread_mutex_unlock(&g_lock);
        return fail(mudaErrorNoDevice);
    }
    g_initialized = 1;
    pthread_mutex_unlock(&g_lock);
    return ok();
}

mudaError_t mudaShutdown(void)
{
    pthread_mutex_lock(&g_lock);
    if (g_primary && g_primary != g_current) {
        g_primary->drv->ctx_destroy(g_primary->drv_ctx);
        free(g_primary);
    }
    if (g_primary == g_current)
        g_current = NULL;
    g_primary = NULL;
    if (g_drv && g_drv->shutdown)
        g_drv->shutdown();
    g_drv = NULL;
    g_initialized = 0;
    pthread_mutex_unlock(&g_lock);
    return ok();
}

const char *mudaGetDriverName(void)
{
    return g_drv ? g_drv->name : "none";
}

mudaError_t mudaGetDeviceCount(int *count)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    return g_drv->get_device_count(count) == mudaSuccess ? ok() : fail(mudaErrorNoDevice);
}

mudaError_t mudaSetDevice(int device)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    int count = 0;
    if (g_drv->get_device_count(&count) != mudaSuccess)
        return fail(mudaErrorNoDevice);
    if (device < 0 || device >= count)
        return fail(mudaErrorInvalidDevice);
    pthread_mutex_lock(&g_lock);
    g_preferred_device = device;
    if (g_primary && g_primary->device != device && g_primary != g_current) {
        g_primary->drv->ctx_destroy(g_primary->drv_ctx);
        free(g_primary);
        g_primary = NULL;
    }
    pthread_mutex_unlock(&g_lock);
    return ok();
}

mudaError_t mudaGetDevice(int *device)
{
    if (!device)
        return fail(mudaErrorInvalidValue);
    if (g_current)
        *device = g_current->device;
    else if (g_primary)
        *device = g_primary->device;
    else
        *device = g_preferred_device;
    return ok();
}

mudaError_t mudaGetDeviceProperties(mudaDeviceProp *prop, int device)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    return g_drv->get_device_props(device, prop) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidDevice);
}

mudaError_t mudaDeviceGetName(char *name, int len, int device)
{
    if (!name || len <= 0)
        return fail(mudaErrorInvalidValue);
    mudaDeviceProp p;
    if (mudaGetDeviceProperties(&p, device) != mudaSuccess)
        return fail(mudaErrorInvalidDevice);
    snprintf(name, (size_t)len, "%s", p.name);
    return ok();
}

mudaError_t mudaDeviceGetAttribute(int *value, mudaDeviceAttr attr, int device)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    if (!value)
        return fail(mudaErrorInvalidValue);
    return g_drv->get_device_attr(device, (int)attr, value) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidValue);
}

mudaError_t mudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    return g_drv->get_device_pci_bus_id(device, pciBusId, len) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidValue);
}

mudaError_t mudaDeviceReset(void)
{
    pthread_mutex_lock(&g_lock);
    if (g_primary) {
        if (g_primary != g_current)
            g_primary->drv->ctx_destroy(g_primary->drv_ctx);
        if (g_primary == g_current)
            g_current = NULL;
        free(g_primary);
        g_primary = NULL;
    }
    pthread_mutex_unlock(&g_lock);
    return ok();
}

mudaError_t mudaDriverGetVersion(int *version)
{
    if (!version)
        return fail(mudaErrorInvalidValue);
    *version = MUDA_VERSION;
    return ok();
}

mudaError_t mudaRuntimeGetVersion(int *version)
{
    if (!version)
        return fail(mudaErrorInvalidValue);
    *version = MUDA_VERSION;
    return ok();
}

/* ------------------------------------------------------------------ */
/* Context                                                            */
/* ------------------------------------------------------------------ */
mudaError_t mudaCtxCreate(mudaContext_t *ctx, unsigned int flags, int device)
{
    (void)flags;
    if (!ctx)
        return fail(mudaErrorInvalidValue);
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);

    muda_context_impl *impl = make_context(device);
    if (!impl)
        return fail(mudaErrorInvalidDevice);
    g_current = impl;
    *ctx = impl;
    return ok();
}

mudaError_t mudaCtxDestroy(mudaContext_t ctx)
{
    muda_context_impl *impl = (muda_context_impl *)ctx;
    if (!impl)
        return fail(mudaErrorInvalidContext);
    impl->drv->ctx_destroy(impl->drv_ctx);
    if (g_current == impl)
        g_current = NULL;
    if (g_primary == impl)
        g_primary = NULL;
    free(impl);
    return ok();
}

mudaError_t mudaCtxPushCurrent(mudaContext_t ctx)
{
    if (!ctx)
        return fail(mudaErrorInvalidContext);
    g_current = (muda_context_impl *)ctx;
    return ok();
}

mudaError_t mudaCtxPopCurrent(mudaContext_t *ctx)
{
    if (ctx)
        *ctx = g_current;
    g_current = NULL;
    return ok();
}

mudaError_t mudaCtxGetCurrent(mudaContext_t *ctx)
{
    if (!ctx)
        return fail(mudaErrorInvalidValue);
    *ctx = g_current;
    return ok();
}

mudaError_t mudaCtxGetDevice(int *device)
{
    if (!device)
        return fail(mudaErrorInvalidValue);
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    *device = c->device;
    return ok();
}

/* ------------------------------------------------------------------ */
/* Memory                                                             */
/* ------------------------------------------------------------------ */
mudaError_t mudaMalloc(void **devPtr, size_t size)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!devPtr)
        return fail(mudaErrorInvalidValue);
    void *p = NULL;
    if (c->drv->malloc(c->drv_ctx, &p, size) != mudaSuccess)
        return fail(mudaErrorOutOfMemory);
    pthread_mutex_lock(&g_lock);
    reg_add(p, size, mudaMemoryTypeDevice, c->device);
    pthread_mutex_unlock(&g_lock);
    *devPtr = p;
    return ok();
}

mudaError_t mudaFree(void *devPtr)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    pthread_mutex_lock(&g_lock);
    reg_remove(devPtr);
    pthread_mutex_unlock(&g_lock);
    return c->drv->free(c->drv_ctx, devPtr) == mudaSuccess ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMallocHost(void **ptr, size_t size)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!ptr)
        return fail(mudaErrorInvalidValue);
    void *p = NULL;
    if (c->drv->malloc_host(c->drv_ctx, &p, size) != mudaSuccess)
        return fail(mudaErrorOutOfMemory);
    pthread_mutex_lock(&g_lock);
    reg_add(p, size, mudaMemoryTypeHost, c->device);
    pthread_mutex_unlock(&g_lock);
    *ptr = p;
    return ok();
}

mudaError_t mudaFreeHost(void *ptr)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    pthread_mutex_lock(&g_lock);
    reg_remove(ptr);
    pthread_mutex_unlock(&g_lock);
    return c->drv->free_host(c->drv_ctx, ptr) == mudaSuccess ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaHostGetDevicePointer(void **devPtr, void *hostPtr, unsigned int flags)
{
    (void)flags;
    if (!devPtr)
        return fail(mudaErrorInvalidValue);
    *devPtr = hostPtr; /* regular pageable memory is used as-is */
    return ok();
}

mudaError_t mudaMemGetInfo(size_t *freeBytes, size_t *totalBytes)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    size_t total = 0, freeb = 0;
    if (c->drv->mem_get_info(c->drv_ctx, &freeb, &total) != mudaSuccess)
        return fail(mudaErrorUnknown);
    pthread_mutex_lock(&g_lock);
    size_t used = reg_device_used();
    pthread_mutex_unlock(&g_lock);
    if (freeBytes)
        *freeBytes = total > used ? total - used : 0;
    if (totalBytes)
        *totalBytes = total;
    return ok();
}

mudaError_t mudaPointerGetAttributes(mudaPointerAttributes *attr, const void *ptr)
{
    if (!attr || !ptr)
        return fail(mudaErrorInvalidValue);
    memset(attr, 0, sizeof(*attr));

    pthread_mutex_lock(&g_lock);
    alloc_rec rec;
    int found = reg_find(ptr, &rec);
    pthread_mutex_unlock(&g_lock);

    if (found) {
        attr->type = (mudaMemoryType)rec.type;
        attr->device = rec.device;
        attr->devicePointer = rec.type == mudaMemoryTypeDevice ? (void *)ptr : NULL;
        attr->hostPointer = rec.type == mudaMemoryTypeHost ? (void *)ptr : NULL;
    } else {
        attr->type = mudaMemoryTypeUnregistered;
        attr->device = g_preferred_device;
        attr->devicePointer = (void *)ptr;
        attr->hostPointer = (void *)ptr;
    }
    return ok();
}

mudaError_t mudaMemcpy(void *dst, const void *src, size_t count, mudaMemcpyKind kind)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memcpy(c->drv_ctx, dst, src, count, (int)kind) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemcpyAsync(void *dst, const void *src, size_t count,
                            mudaMemcpyKind kind, mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memcpy_async(c->drv_ctx, dst, src, count, (int)kind, stream) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, mudaMemcpyKind kind)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memcpy_2d(c->drv_ctx, dst, dpitch, src, spitch, width, height,
                             (int)kind, 0, NULL) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                              size_t width, size_t height, mudaMemcpyKind kind,
                              mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memcpy_2d(c->drv_ctx, dst, dpitch, src, spitch, width, height,
                             (int)kind, 1, stream) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemset(void *devPtr, int value, size_t count)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memset(c->drv_ctx, devPtr, value, count) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemsetD8(void *devPtr, unsigned char value, size_t count)
{
    return mudaMemset(devPtr, (int)value, count);
}

mudaError_t mudaMemsetD32(void *devPtr, unsigned int value, size_t count)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memset_d32(c->drv_ctx, devPtr, value, count) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaMemAdvise(const void *devPtr, size_t count, int advice, int device)
{
    (void)devPtr; (void)count; (void)advice; (void)device;
    return ok(); /* no-op */
}

mudaError_t mudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                                 mudaStream_t stream)
{
    (void)devPtr; (void)count; (void)dstDevice; (void)stream;
    return ok(); /* no-op */
}

/* ------------------------------------------------------------------ */
/* Modules / kernels                                                  */
/* ------------------------------------------------------------------ */
mudaError_t mudaModuleLoadData(mudaModule_t *module, const char *source)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!module || !source)
        return fail(mudaErrorInvalidValue);
    return c->drv->module_load(c->drv_ctx, source, module) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidValue);
}

mudaError_t mudaModuleLoad(mudaModule_t *module, const char *path)
{
    if (!path)
        return fail(mudaErrorInvalidValue);
    char *src = NULL;
    if (file_read(path, &src) != 0)
        return fail(mudaErrorNotFound);
    mudaError_t e = mudaModuleLoadData(module, src);
    free(src);
    return e;
}

mudaError_t mudaModuleUnload(mudaModule_t module)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    c->drv->module_unload(c->drv_ctx, module);
    return ok();
}

mudaError_t mudaModuleGetFunction(mudaFunction_t *func, mudaModule_t module, const char *name)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!func || !module || !name)
        return fail(mudaErrorInvalidValue);
    return c->drv->func_get(c->drv_ctx, module, name, func) == mudaSuccess
               ? ok() : fail(mudaErrorNotFound);
}

mudaError_t mudaFuncGetAttributes(mudaFuncAttributes *attr, mudaFunction_t func)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!attr || !func)
        return fail(mudaErrorInvalidValue);
    return c->drv->func_get_attributes(c->drv_ctx, func, attr) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidValue);
}

mudaError_t mudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          mudaFunction_t func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!numBlocks || !func || blockSize <= 0)
        return fail(mudaErrorInvalidValue);

    mudaDeviceProp p;
    if (c->drv->get_device_props(c->device, &p) != mudaSuccess)
        return fail(mudaErrorInvalidDevice);

    mudaFuncAttributes fa;
    memset(&fa, 0, sizeof(fa));
    c->drv->func_get_attributes(c->drv_ctx, func, &fa);

    int maxThreads = p.maxThreadsPerBlock;
    if (fa.maxThreadsPerBlock > 0 && fa.maxThreadsPerBlock < maxThreads)
        maxThreads = fa.maxThreadsPerBlock;

    if (blockSize > maxThreads) {
        *numBlocks = 0;
        return ok();
    }

    int byThreads = maxThreads / blockSize;
    int byShared = INT_MAX;
    size_t perBlock = fa.sharedSizeBytes + dynamicSMemSize;
    if (perBlock > 0 && p.sharedMemPerMultiprocessor > 0)
        byShared = (int)(p.sharedMemPerMultiprocessor / perBlock);

    *numBlocks = byThreads < byShared ? byThreads : byShared;
    if (*numBlocks < 1)
        *numBlocks = 1;
    return ok();
}

static mudaError_t launch_impl(mudaFunction_t f,
                               unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz,
                               unsigned shared, mudaStream_t stream,
                               void **kernelParams, void **extra)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!f)
        return fail(mudaErrorInvalidHandle);

    void *packed = NULL;
    size_t packed_size = 0;
    if (!kernelParams && extra) {
        for (void **p = extra; *p != MUDA_LAUNCH_PARAM_END; p += 2) {
            if (p[0] == MUDA_LAUNCH_PARAM_BUFFER_POINTER)
                packed = p[1];
            else if (p[0] == MUDA_LAUNCH_PARAM_BUFFER_SIZE)
                packed_size = p[1] ? *(size_t *)p[1] : 0;
        }
        if (!packed)
            return fail(mudaErrorInvalidValue);
    }

    unsigned grid[3]  = { gx, gy, gz };
    unsigned block[3] = { bx, by, bz };

    return c->drv->launch(c->drv_ctx, f, grid, block, shared, stream,
                          kernelParams, packed, packed_size) == mudaSuccess
               ? ok() : fail(mudaErrorLaunchFailure);
}

mudaError_t mudaLaunchKernel(mudaFunction_t f,
                             unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
                             unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                             unsigned sharedMemBytes, mudaStream_t stream,
                             void **kernelParams, void **extra)
{
    return launch_impl(f, gridDimX, gridDimY, gridDimZ,
                       blockDimX, blockDimY, blockDimZ,
                       sharedMemBytes, stream, kernelParams, extra);
}

mudaError_t mudaLaunchKernelEx(const mudaLaunchConfig *config, mudaFunction_t f,
                               void **kernelParams, void **extra)
{
    if (!config)
        return fail(mudaErrorInvalidValue);
    return launch_impl(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                       config->blockDimX, config->blockDimY, config->blockDimZ,
                       config->sharedMemBytes, config->stream, kernelParams, extra);
}

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
mudaError_t mudaStreamCreateWithFlags(mudaStream_t *stream, unsigned int flags)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!stream)
        return fail(mudaErrorInvalidValue);
    return c->drv->stream_create(c->drv_ctx, stream, flags) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaStreamCreate(mudaStream_t *stream)
{
    return mudaStreamCreateWithFlags(stream, mudaStreamDefault);
}

mudaError_t mudaStreamDestroy(mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    c->drv->stream_destroy(c->drv_ctx, stream);
    return ok();
}

mudaError_t mudaStreamSynchronize(mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->stream_sync(c->drv_ctx, stream) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaStreamQuery(mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->stream_query(c->drv_ctx, stream) == mudaSuccess
               ? ok() : fail(mudaErrorNotReady);
}

mudaError_t mudaStreamWaitEvent(mudaStream_t stream, mudaEvent_t event, unsigned int flags)
{
    (void)flags;
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->stream_wait_event(c->drv_ctx, stream, event) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
mudaError_t mudaEventCreateWithFlags(mudaEvent_t *event, unsigned int flags)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!event)
        return fail(mudaErrorInvalidValue);
    return c->drv->event_create(c->drv_ctx, event, flags) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaEventCreate(mudaEvent_t *event)
{
    return mudaEventCreateWithFlags(event, mudaEventDefault);
}

mudaError_t mudaEventDestroy(mudaEvent_t event)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    c->drv->event_destroy(c->drv_ctx, event);
    return ok();
}

mudaError_t mudaEventRecord(mudaEvent_t event, mudaStream_t stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->event_record(c->drv_ctx, event, stream) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaEventSynchronize(mudaEvent_t event)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->event_sync(c->drv_ctx, event) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaEventQuery(mudaEvent_t event)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->event_query(c->drv_ctx, event) == mudaSuccess
               ? ok() : fail(mudaErrorNotReady);
}

mudaError_t mudaEventElapsedTime(float *ms, mudaEvent_t start, mudaEvent_t end)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->event_elapsed(c->drv_ctx, ms, start, end) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

/* ------------------------------------------------------------------ */
/* Misc                                                               */
/* ------------------------------------------------------------------ */
mudaError_t mudaDeviceSynchronize(void)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->device_sync(c->drv_ctx) == mudaSuccess ? ok() : fail(mudaErrorUnknown);
}

mudaError_t mudaGetLastError(void)
{
    mudaError_t e = (mudaError_t)g_last_error;
    g_last_error = mudaSuccess;
    return e;
}

mudaError_t mudaPeekAtLastError(void)
{
    return (mudaError_t)g_last_error;
}

const char *mudaGetErrorName(mudaError_t error)
{
    switch (error) {
    case mudaSuccess:                   return "mudaSuccess";
    case mudaErrorInvalidValue:         return "mudaErrorInvalidValue";
    case mudaErrorOutOfMemory:          return "mudaErrorOutOfMemory";
    case mudaErrorNotInitialized:       return "mudaErrorNotInitialized";
    case mudaErrorDeinitialized:        return "mudaErrorDeinitialized";
    case mudaErrorProfilerDisabled:     return "mudaErrorProfilerDisabled";
    case mudaErrorProfilerNotInitialized: return "mudaErrorProfilerNotInitialized";
    case mudaErrorProfilerAlreadyStarted: return "mudaErrorProfilerAlreadyStarted";
    case mudaErrorProfilerAlreadyStopped: return "mudaErrorProfilerAlreadyStopped";
    case mudaErrorNoDevice:             return "mudaErrorNoDevice";
    case mudaErrorInvalidDevice:        return "mudaErrorInvalidDevice";
    case mudaErrorInvalidContext:       return "mudaErrorInvalidContext";
    case mudaErrorInvalidHandle:        return "mudaErrorInvalidHandle";
    case mudaErrorNotFound:             return "mudaErrorNotFound";
    case mudaErrorNotReady:             return "mudaErrorNotReady";
    case mudaErrorLaunchFailure:        return "mudaErrorLaunchFailure";
    case mudaErrorNotSupported:         return "mudaErrorNotSupported";
    default:                            return "mudaErrorUnknown";
    }
}

const char *mudaGetErrorString(mudaError_t error)
{
    switch (error) {
    case mudaSuccess:                   return "no error";
    case mudaErrorInvalidValue:         return "invalid value";
    case mudaErrorOutOfMemory:          return "out of memory";
    case mudaErrorNotInitialized:       return "runtime not initialized (call mudaInit)";
    case mudaErrorDeinitialized:        return "runtime was deinitialized";
    case mudaErrorProfilerDisabled:     return "profiler is disabled";
    case mudaErrorProfilerNotInitialized: return "profiler is not initialized";
    case mudaErrorProfilerAlreadyStarted: return "profiler already started";
    case mudaErrorProfilerAlreadyStopped: return "profiler already stopped";
    case mudaErrorNoDevice:             return "no capable device found";
    case mudaErrorInvalidDevice:        return "invalid device ordinal";
    case mudaErrorInvalidContext:       return "invalid context";
    case mudaErrorInvalidHandle:        return "invalid handle";
    case mudaErrorNotFound:             return "not found";
    case mudaErrorNotReady:             return "operation not yet complete";
    case mudaErrorLaunchFailure:        return "kernel launch failed";
    case mudaErrorNotSupported:         return "operation not supported by this driver";
    default:                            return "unknown error";
    }
}
