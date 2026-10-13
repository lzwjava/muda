/*
 * muda.c - public API dispatcher.
 *
 * Holds the process-wide runtime state (selected driver, current context)
 * and forwards each call to the active muda driver.
 */

#include "muda_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static pthread_mutex_t    g_lock = PTHREAD_MUTEX_INITIALIZER;
static const muda_driver *g_drv = NULL;
static int                 g_initialized = 0;
static int                 g_last_error = mudaSuccess;

static muda_context_impl *g_current = NULL;
static muda_context_impl *g_primary = NULL;

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
        g_primary = make_context(0);
    muda_context_impl *result = g_current ? g_current : g_primary;
    pthread_mutex_unlock(&g_lock);
    return result;
}

/* ------------------------------------------------------------------ */
/* Runtime                                                            */
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

mudaError_t mudaGetDeviceProperties(mudaDeviceProp *prop, int device)
{
    if (!g_initialized)
        return fail(mudaErrorNotInitialized);
    return g_drv->get_device_props(device, prop) == mudaSuccess
               ? ok() : fail(mudaErrorInvalidDevice);
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

/* ------------------------------------------------------------------ */
/* Memory                                                             */
/* ------------------------------------------------------------------ */
mudaError_t mudaMalloc(void **devPtr, size_t size)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->malloc(c->drv_ctx, devPtr, size) == mudaSuccess
               ? ok() : fail(mudaErrorOutOfMemory);
}

mudaError_t mudaFree(void *devPtr)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->free(c->drv_ctx, devPtr) == mudaSuccess ? ok() : fail(mudaErrorUnknown);
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

mudaError_t mudaMemset(void *devPtr, int value, size_t count)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    return c->drv->memset(c->drv_ctx, devPtr, value, count) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
}

/* ------------------------------------------------------------------ */
/* Modules                                                            */
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

mudaError_t mudaLaunchKernel(mudaFunction_t f,
                             unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
                             unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                             unsigned sharedMemBytes, mudaStream_t stream,
                             void **kernelParams, void **extra)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!f)
        return fail(mudaErrorInvalidHandle);
    if (extra)
        return fail(mudaErrorNotSupported); /* packed-buffer launch not implemented */

    unsigned grid[3]  = { gridDimX, gridDimY, gridDimZ };
    unsigned block[3] = { blockDimX, blockDimY, blockDimZ };

    return c->drv->launch(c->drv_ctx, f, grid, block, sharedMemBytes, stream, kernelParams)
               == mudaSuccess ? ok() : fail(mudaErrorLaunchFailure);
}

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
mudaError_t mudaStreamCreate(mudaStream_t *stream)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!stream)
        return fail(mudaErrorInvalidValue);
    return c->drv->stream_create(c->drv_ctx, stream) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
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

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
mudaError_t mudaEventCreate(mudaEvent_t *event)
{
    muda_context_impl *c = get_ctx();
    if (!c)
        return fail(mudaErrorNotInitialized);
    if (!event)
        return fail(mudaErrorInvalidValue);
    return c->drv->event_create(c->drv_ctx, event) == mudaSuccess
               ? ok() : fail(mudaErrorUnknown);
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

const char *mudaGetErrorName(mudaError_t error)
{
    switch (error) {
    case mudaSuccess:              return "mudaSuccess";
    case mudaErrorInvalidValue:    return "mudaErrorInvalidValue";
    case mudaErrorOutOfMemory:     return "mudaErrorOutOfMemory";
    case mudaErrorNotInitialized:  return "mudaErrorNotInitialized";
    case mudaErrorDeinitialized:   return "mudaErrorDeinitialized";
    case mudaErrorNoDevice:        return "mudaErrorNoDevice";
    case mudaErrorInvalidDevice:   return "mudaErrorInvalidDevice";
    case mudaErrorInvalidContext:  return "mudaErrorInvalidContext";
    case mudaErrorInvalidHandle:   return "mudaErrorInvalidHandle";
    case mudaErrorNotFound:        return "mudaErrorNotFound";
    case mudaErrorNotSupported:    return "mudaErrorNotSupported";
    case mudaErrorLaunchFailure:   return "mudaErrorLaunchFailure";
    default:                       return "mudaErrorUnknown";
    }
}

const char *mudaGetErrorString(mudaError_t error)
{
    switch (error) {
    case mudaSuccess:              return "no error";
    case mudaErrorInvalidValue:    return "invalid value";
    case mudaErrorOutOfMemory:     return "out of memory";
    case mudaErrorNotInitialized:  return "runtime not initialized (call mudaInit)";
    case mudaErrorDeinitialized:   return "runtime was deinitialized";
    case mudaErrorNoDevice:        return "no capable device found";
    case mudaErrorInvalidDevice:   return "invalid device ordinal";
    case mudaErrorInvalidContext:  return "invalid context";
    case mudaErrorInvalidHandle:   return "invalid handle";
    case mudaErrorNotFound:        return "not found";
    case mudaErrorNotSupported:    return "operation not supported by this driver";
    case mudaErrorLaunchFailure:   return "kernel launch failed";
    default:                       return "unknown error";
    }
}
