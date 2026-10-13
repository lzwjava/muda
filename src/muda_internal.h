/*
 * muda_internal.h - internal driver interface.
 *
 * A "muda driver" is a backend that knows how to execute kernels on some
 * device.  Backends are selected at run time (MUDA_DRIVER env var) and
 * registered here.  The public API in muda.h is a thin dispatcher on top
 * of this interface.
 */
#ifndef MUDA_INTERNAL_H
#define MUDA_INTERNAL_H

#include "muda.h"

typedef struct muda_driver {
    const char *name;

    /* lifecycle */
    int  (*init)(void);
    void (*shutdown)(void);

    /* device enumeration */
    int  (*get_device_count)(int *count);
    int  (*get_device_props)(int device, mudaDeviceProp *prop);

    /* context */
    int  (*ctx_create)(void **drv_ctx, int device);
    void (*ctx_destroy)(void *drv_ctx);

    /* memory */
    int  (*malloc)(void *drv_ctx, void **ptr, size_t size);
    int  (*free)(void *drv_ctx, void *ptr);
    int  (*memcpy)(void *drv_ctx, void *dst, const void *src, size_t size, int kind);
    int  (*memcpy_async)(void *drv_ctx, void *dst, const void *src, size_t size,
                         int kind, void *stream);
    int  (*memset)(void *drv_ctx, void *ptr, int value, size_t size);

    /* modules / kernels */
    int  (*module_load)(void *drv_ctx, const char *source, void **mod);
    int  (*module_unload)(void *drv_ctx, void *mod);
    int  (*func_get)(void *drv_ctx, void *mod, const char *name, void **fn);
    int  (*launch)(void *drv_ctx, void *fn, unsigned grid[3], unsigned block[3],
                   unsigned shared, void *stream, void **args);

    /* streams */
    int  (*stream_create)(void *drv_ctx, void **stream);
    void (*stream_destroy)(void *drv_ctx, void *stream);
    int  (*stream_sync)(void *drv_ctx, void *stream);
    int  (*device_sync)(void *drv_ctx);

    /* events */
    int  (*event_create)(void *drv_ctx, void **event);
    void (*event_destroy)(void *drv_ctx, void *event);
    int  (*event_record)(void *drv_ctx, void *event, void *stream);
    int  (*event_sync)(void *drv_ctx, void *event);
    int  (*event_elapsed)(void *drv_ctx, float *ms, void *start, void *end);
} muda_driver;

/* Context wrapper that binds a driver to a driver-private context. */
typedef struct muda_context_impl {
    const muda_driver *drv;
    void              *drv_ctx;
    int                device;
} muda_context_impl;

/* Registered backends. */
extern const muda_driver muda_driver_opencl;

/* Utilities shared by backends. */
char *muda_translate_cuda_to_opencl(const char *source);

#endif /* MUDA_INTERNAL_H */
