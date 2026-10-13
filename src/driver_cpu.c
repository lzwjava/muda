/*
 * driver_cpu.c - a reference/no-kernel backend.
 *
 * This driver implements the full memory, context, stream and event surface
 * using ordinary host memory.  It cannot compile or execute device kernels
 * (launch returns mudaErrorNotSupported), but it lets everything else run on
 * a machine without a GPU - useful for CI and for exercising the API.
 *
 * Select it with MUDA_DRIVER=cpu.  The OpenCL backend is the default.
 */

#include "muda_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CPU_MEM_BYTES ((size_t)4 * 1024 * 1024 * 1024)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void fill_prop(mudaDeviceProp *p)
{
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "muda CPU reference (memory only)");
    snprintf(p->driverVersion, sizeof(p->driverVersion), "muda-cpu");
    snprintf(p->deviceVersion, sizeof(p->deviceVersion), "cpu");
    p->totalGlobalMem            = CPU_MEM_BYTES;
    p->sharedMemPerBlock         = 65536;
    p->sharedMemPerMultiprocessor = 65536;
    p->memPitch                  = 2147483647UL;
    p->maxThreadsPerBlock        = 1024;
    p->maxThreadsDim[0]          = 1024;
    p->maxThreadsDim[1]          = 1024;
    p->maxThreadsDim[2]          = 64;
    p->maxGridSize[0]            = 2147483647;
    p->maxGridSize[1]            = 65535;
    p->maxGridSize[2]            = 65535;
    p->multiProcessorCount       = 1;
    p->warpSize                  = 1;
    p->concurrentKernels         = 0;
}

/* ------------------------------------------------------------------ */
/* lifecycle / device                                                 */
/* ------------------------------------------------------------------ */
static int cpu_init(void) { return mudaSuccess; }
static void cpu_shutdown(void) {}

static int cpu_get_device_count(int *count)
{
    if (!count)
        return mudaErrorInvalidValue;
    *count = 1;
    return mudaSuccess;
}

static int cpu_get_device_props(int device, mudaDeviceProp *prop)
{
    if (!prop || device != 0)
        return mudaErrorInvalidDevice;
    fill_prop(prop);
    return mudaSuccess;
}

static int cpu_get_device_attr(int device, int attr, int *value)
{
    if (!value || device != 0)
        return mudaErrorInvalidValue;
    mudaDeviceProp p;
    fill_prop(&p);
    switch (attr) {
    case mudaDevAttrMaxThreadsPerBlock:      *value = p.maxThreadsPerBlock; break;
    case mudaDevAttrMaxBlockDimX:            *value = p.maxThreadsDim[0]; break;
    case mudaDevAttrMaxBlockDimY:            *value = p.maxThreadsDim[1]; break;
    case mudaDevAttrMaxBlockDimZ:            *value = p.maxThreadsDim[2]; break;
    case mudaDevAttrMaxGridDimX:             *value = p.maxGridSize[0]; break;
    case mudaDevAttrMaxGridDimY:             *value = p.maxGridSize[1]; break;
    case mudaDevAttrMaxGridDimZ:             *value = p.maxGridSize[2]; break;
    case mudaDevAttrMaxSharedMemoryPerBlock: *value = (int)p.sharedMemPerBlock; break;
    case mudaDevAttrMaxSharedMemoryPerMultiprocessor:
        *value = (int)p.sharedMemPerMultiprocessor; break;
    case mudaDevAttrWarpSize:                *value = p.warpSize; break;
    case mudaDevAttrMultiProcessorCount:     *value = p.multiProcessorCount; break;
    case mudaDevAttrConcurrentKernels:       *value = p.concurrentKernels; break;
    case mudaDevAttrComputeCapabilityMajor:  *value = 0; break;
    case mudaDevAttrComputeCapabilityMinor:  *value = 0; break;
    default:                                 return mudaErrorInvalidValue;
    }
    return mudaSuccess;
}

static int cpu_get_device_pci_bus_id(int device, char *buf, int len)
{
    if (device != 0 || !buf || len <= 0)
        return mudaErrorInvalidValue;
    snprintf(buf, (size_t)len, "cpu");
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* context                                                            */
/* ------------------------------------------------------------------ */
static int cpu_ctx_create(void **ctx, int device)
{
    if (device != 0)
        return mudaErrorInvalidDevice;
    void *c = calloc(1, 1);
    if (!c)
        return mudaErrorOutOfMemory;
    *ctx = c;
    return mudaSuccess;
}

static void cpu_ctx_destroy(void *ctx) { free(ctx); }

/* ------------------------------------------------------------------ */
/* memory                                                             */
/* ------------------------------------------------------------------ */
static int cpu_malloc(void *c, void **ptr, size_t size)
{
    (void)c;
    if (!ptr || size == 0)
        return mudaErrorInvalidValue;
    void *p = malloc(size);
    if (!p)
        return mudaErrorOutOfMemory;
    *ptr = p;
    return mudaSuccess;
}

static int cpu_malloc_host(void *c, void **ptr, size_t size) { return cpu_malloc(c, ptr, size); }
static int cpu_free(void *c, void *ptr) { (void)c; free(ptr); return mudaSuccess; }
static int cpu_free_host(void *c, void *ptr) { return cpu_free(c, ptr); }

static int cpu_mem_get_info(void *c, size_t *freeBytes, size_t *totalBytes)
{
    (void)c;
    if (freeBytes)  *freeBytes = CPU_MEM_BYTES;
    if (totalBytes) *totalBytes = CPU_MEM_BYTES;
    return mudaSuccess;
}

static int cpu_memcpy(void *c, void *dst, const void *src, size_t size, int kind)
{
    (void)c; (void)kind;
    if (dst && src && size)
        memcpy(dst, src, size);
    return mudaSuccess;
}

static int cpu_memcpy_async(void *c, void *dst, const void *src, size_t size,
                            int kind, void *stream)
{
    (void)stream;
    return cpu_memcpy(c, dst, src, size, kind);
}

static int cpu_memcpy_2d(void *c, void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height, int kind,
                         int async, void *stream)
{
    (void)c; (void)kind; (void)async; (void)stream;
    for (size_t r = 0; r < height; r++)
        memcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, width);
    return mudaSuccess;
}

static int cpu_memset(void *c, void *ptr, int value, size_t size)
{
    (void)c;
    if (ptr)
        memset(ptr, value, size);
    return mudaSuccess;
}

static int cpu_memset_d32(void *c, void *ptr, unsigned value, size_t count)
{
    (void)c;
    for (size_t i = 0; i < count; i++)
        ((unsigned *)ptr)[i] = value;
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* modules / kernels (not supported)                                  */
/* ------------------------------------------------------------------ */
static int cpu_module_load(void *c, const char *src, void **mod)
{
    (void)c; (void)src; (void)mod;
    return mudaErrorNotSupported;
}
static int cpu_module_unload(void *c, void *mod) { (void)c; (void)mod; return mudaSuccess; }
static int cpu_func_get(void *c, void *mod, const char *name, void **fn)
{
    (void)c; (void)mod; (void)name; (void)fn;
    return mudaErrorNotSupported;
}
static int cpu_func_get_attributes(void *c, void *fn, mudaFuncAttributes *a)
{
    (void)c; (void)fn; (void)a;
    return mudaErrorNotSupported;
}
static int cpu_launch(void *c, void *fn, unsigned grid[3], unsigned block[3],
                      unsigned shared, void *stream, void **args,
                      void *packed, size_t packed_size)
{
    (void)c; (void)fn; (void)grid; (void)block; (void)shared; (void)stream;
    (void)args; (void)packed; (void)packed_size;
    return mudaErrorNotSupported;
}

/* ------------------------------------------------------------------ */
/* streams                                                            */
/* ------------------------------------------------------------------ */
static int cpu_stream_create(void *c, void **stream, unsigned flags)
{
    (void)c; (void)flags;
    void *s = calloc(1, 1);
    if (!s)
        return mudaErrorOutOfMemory;
    *stream = s;
    return mudaSuccess;
}
static void cpu_stream_destroy(void *c, void *stream) { (void)c; free(stream); }
static int cpu_stream_sync(void *c, void *stream) { (void)c; (void)stream; return mudaSuccess; }
static int cpu_stream_query(void *c, void *stream) { (void)c; (void)stream; return mudaSuccess; }
static int cpu_stream_wait_event(void *c, void *stream, void *event)
{
    (void)c; (void)stream; (void)event;
    return mudaSuccess;
}
static int cpu_device_sync(void *c) { (void)c; return mudaSuccess; }

/* ------------------------------------------------------------------ */
/* events                                                             */
/* ------------------------------------------------------------------ */
typedef struct { double t0, t1; } cpu_event;

static int cpu_event_create(void *c, void **event, unsigned flags)
{
    (void)c; (void)flags;
    cpu_event *e = (cpu_event *)calloc(1, sizeof(cpu_event));
    if (!e)
        return mudaErrorOutOfMemory;
    *event = e;
    return mudaSuccess;
}
static void cpu_event_destroy(void *c, void *event) { (void)c; free(event); }
static int cpu_event_record(void *c, void *event, void *stream)
{
    (void)c; (void)stream;
    ((cpu_event *)event)->t0 = now_ms();
    ((cpu_event *)event)->t1 = ((cpu_event *)event)->t0;
    return mudaSuccess;
}
static int cpu_event_sync(void *c, void *event)
{
    (void)c;
    ((cpu_event *)event)->t1 = now_ms();
    return mudaSuccess;
}
static int cpu_event_query(void *c, void *event) { (void)c; (void)event; return mudaSuccess; }
static int cpu_event_elapsed(void *c, float *ms, void *start, void *end)
{
    (void)c;
    if (!ms)
        return mudaErrorInvalidValue;
    *ms = (float)(((cpu_event *)end)->t1 - ((cpu_event *)start)->t0);
    return mudaSuccess;
}

const muda_driver muda_driver_cpu = {
    .name                  = "cpu",
    .init                  = cpu_init,
    .shutdown              = cpu_shutdown,
    .get_device_count      = cpu_get_device_count,
    .get_device_props      = cpu_get_device_props,
    .get_device_attr       = cpu_get_device_attr,
    .get_device_pci_bus_id = cpu_get_device_pci_bus_id,
    .ctx_create            = cpu_ctx_create,
    .ctx_destroy           = cpu_ctx_destroy,
    .malloc                = cpu_malloc,
    .malloc_host           = cpu_malloc_host,
    .free                  = cpu_free,
    .free_host             = cpu_free_host,
    .mem_get_info          = cpu_mem_get_info,
    .memcpy                = cpu_memcpy,
    .memcpy_async          = cpu_memcpy_async,
    .memcpy_2d             = cpu_memcpy_2d,
    .memset                = cpu_memset,
    .memset_d32            = cpu_memset_d32,
    .module_load           = cpu_module_load,
    .module_unload         = cpu_module_unload,
    .func_get              = cpu_func_get,
    .func_get_attributes   = cpu_func_get_attributes,
    .launch                = cpu_launch,
    .stream_create         = cpu_stream_create,
    .stream_destroy        = cpu_stream_destroy,
    .stream_sync           = cpu_stream_sync,
    .stream_query          = cpu_stream_query,
    .stream_wait_event     = cpu_stream_wait_event,
    .device_sync           = cpu_device_sync,
    .event_create          = cpu_event_create,
    .event_destroy         = cpu_event_destroy,
    .event_record          = cpu_event_record,
    .event_sync            = cpu_event_sync,
    .event_query           = cpu_event_query,
    .event_elapsed         = cpu_event_elapsed,
};
