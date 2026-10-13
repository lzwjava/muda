/*
 * driver_nouveau.c - a native DRM/nouveau backend (experimental).
 *
 * This driver talks straight to the open nouveau kernel driver through
 * /dev/dri/renderD* using the DRM ioctl ABI.  It provides device information
 * and GPU memory management (GEM allocation + mmap) without NVIDIA's
 * user-space stack.
 *
 * Kernel execution is not implemented yet: module_load/launch return
 * mudaErrorNotSupported.  The plan is to grow a compute channel on top of
 * this, reusing OpenCL only as a compiler if necessary.
 *
 * Requires the nouveau kernel driver to be bound to the M40:
 *
 *     sudo tools/muda-gpu-mode.sh nouveau
 *     MUDA_DRIVER=nouveau build/mudainfo
 *
 * The ioctl numbers and structs below are hand-written (like cl_min.h) so
 * there is no dependency on the libdrm development headers.
 */

#include "muda_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>

/* ------------------------------------------------------------------ */
/* Minimal DRM / nouveau ABI                                          */
/* ------------------------------------------------------------------ */
#define DRM_IOCTL_BASE   'd'
#define DRM_COMMAND_BASE 0x40

struct drm_version {
    int      version_major;
    int      version_minor;
    int      version_patchlevel;
    size_t   name_len;
    char    *name;
    size_t   date_len;
    char    *date;
    size_t   desc_len;
    char    *desc;
};
#define DRM_IOCTL_VERSION _IOWR(DRM_IOCTL_BASE, 0x00, struct drm_version)

struct drm_gem_close {
    uint32_t handle;
    uint32_t pad;
};
#define DRM_IOCTL_GEM_CLOSE _IOW(DRM_IOCTL_BASE, 0x09, struct drm_gem_close)

struct drm_nouveau_getparam {
    uint64_t param;
    uint64_t value;
};
#define DRM_IOCTL_NOUVEAU_GETPARAM \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct drm_nouveau_getparam)

#define NOUVEAU_GETPARAM_PCI_VENDOR 0x00
#define NOUVEAU_GETPARAM_PCI_DEVICE 0x01
#define NOUVEAU_GETPARAM_FB_SIZE    0x04
#define NOUVEAU_GETPARAM_CHIPSET_ID 0x10

struct drm_nouveau_gem_info {
    uint32_t handle;
    uint32_t domain;
    uint64_t size;
    uint64_t offset;
    uint64_t map_handle;
    uint32_t tile_mode;
    uint32_t tile_flags;
};

struct drm_nouveau_gem_new {
    union {
        struct drm_nouveau_gem_info info;
        uint64_t align;
    } u;
    uint32_t domain;
    uint32_t tile_flags;
    uint32_t tile_mode;
};
#define DRM_IOCTL_NOUVEAU_GEM_NEW \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x40, struct drm_nouveau_gem_new)

#define NOUVEAU_GEM_DOMAIN_CPU   (1 << 0)
#define NOUVEAU_GEM_DOMAIN_VRAM  (1 << 1)
#define NOUVEAU_GEM_DOMAIN_GART  (1 << 2)

/* ------------------------------------------------------------------ */
/* Driver state                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    int fd;   /* render node file descriptor (-1 if unavailable) */
    int card;
    uint64_t vram_bytes;
    uint32_t chipset;
    uint32_t pci_device;
} nv_state;

typedef struct {
    nv_state *st;
    uint32_t  handle;
    uint64_t  map_handle;
    size_t    size;
    void     *map;
} nv_buf;

static nv_state g_nv = { -1, 0, 0, 0, 0 };

static uint64_t read_sysfs_size(const char *bdf, int bar)
{
    char path[512];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char line[256];
    int idx = 0;
    uint64_t size = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long start = 0, end = 0, flags = 0;
        if (idx == bar && sscanf(line, "%llx %llx %llx", &start, &end, &flags) == 3)
            size = (end >= start) ? (end - start + 1) : 0;
        idx++;
    }
    fclose(f);
    return size;
}

static int nv_getparam(int fd, uint64_t param, uint64_t *value)
{
    struct drm_nouveau_getparam p = { param, 0 };
    if (ioctl(fd, DRM_IOCTL_NOUVEAU_GETPARAM, &p) != 0)
        return -1;
    *value = p.value;
    return 0;
}

static int nv_open(void)
{
    for (int i = 0; i < 8; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0)
            continue;

        char name[64] = {0};
        struct drm_version v;
        memset(&v, 0, sizeof(v));
        v.name_len = sizeof(name);
        v.name = name;
        if (ioctl(fd, DRM_IOCTL_VERSION, &v) != 0 || strcmp(name, "nouveau") != 0) {
            close(fd);
            continue;
        }

        uint64_t vendor = 0, dev = 0, chipset = 0, fb = 0;
        nv_getparam(fd, NOUVEAU_GETPARAM_PCI_VENDOR, &vendor);
        nv_getparam(fd, NOUVEAU_GETPARAM_PCI_DEVICE, &dev);
        nv_getparam(fd, NOUVEAU_GETPARAM_CHIPSET_ID, &chipset);
        nv_getparam(fd, NOUVEAU_GETPARAM_FB_SIZE, &fb);

        g_nv.fd = fd;
        g_nv.card = i;
        g_nv.chipset = (uint32_t)chipset;
        g_nv.pci_device = (uint32_t)dev;
        g_nv.vram_bytes = fb;
        return 0;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* lifecycle / device                                                 */
/* ------------------------------------------------------------------ */
static int nv_init(void)
{
    if (nv_open() != 0) {
        fprintf(stderr, "muda: no nouveau DRM device found "
                        "(is nouveau bound? see tools/muda-gpu-mode.sh)\n");
        return mudaErrorNoDevice;
    }
    return mudaSuccess;
}

static void nv_shutdown(void)
{
    if (g_nv.fd >= 0)
        close(g_nv.fd);
    g_nv.fd = -1;
}

static int nv_get_device_count(int *count)
{
    if (!count)
        return mudaErrorInvalidValue;
    *count = (g_nv.fd >= 0) ? 1 : 0;
    return mudaSuccess;
}

static void fill_prop(mudaDeviceProp *p)
{
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "NVIDIA (nouveau) chipset 0x%04x",
             g_nv.chipset);
    snprintf(p->driverVersion, sizeof(p->driverVersion), "nouveau");
    snprintf(p->deviceVersion, sizeof(p->deviceVersion), "drm-nouveau");
    p->totalGlobalMem = g_nv.vram_bytes
                            ? g_nv.vram_bytes
                            : read_sysfs_size("0000:01:00.0", 1);
    p->sharedMemPerBlock = 49152;
    p->sharedMemPerMultiprocessor = 49152;
    p->memPitch = 2147483647UL;
    p->maxThreadsPerBlock = 1024;
    p->maxThreadsDim[0] = 1024;
    p->maxThreadsDim[1] = 1024;
    p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 2147483647;
    p->maxGridSize[1] = 65535;
    p->maxGridSize[2] = 65535;
    p->multiProcessorCount = 24;
    p->warpSize = 32;
    p->computeCapabilityMajor = 5;
    p->computeCapabilityMinor = 2;
}

static int nv_get_device_props(int device, mudaDeviceProp *prop)
{
    if (!prop || device != 0 || g_nv.fd < 0)
        return mudaErrorInvalidDevice;
    fill_prop(prop);
    return mudaSuccess;
}

static int nv_get_device_attr(int device, int attr, int *value)
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
    case mudaDevAttrMaxSharedMemoryPerBlock: *value = (int)p.sharedMemPerBlock; break;
    case mudaDevAttrWarpSize:                *value = p.warpSize; break;
    case mudaDevAttrMultiProcessorCount:     *value = p.multiProcessorCount; break;
    case mudaDevAttrComputeCapabilityMajor:  *value = p.computeCapabilityMajor; break;
    case mudaDevAttrComputeCapabilityMinor:  *value = p.computeCapabilityMinor; break;
    default:                                 return mudaErrorInvalidValue;
    }
    return mudaSuccess;
}

static int nv_get_device_pci_bus_id(int device, char *buf, int len)
{
    if (device != 0 || !buf || len <= 0)
        return mudaErrorInvalidValue;
    snprintf(buf, (size_t)len, "0000:01:00.0");
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* context                                                            */
/* ------------------------------------------------------------------ */
static int nv_ctx_create(void **ctx, int device)
{
    if (device != 0 || g_nv.fd < 0)
        return mudaErrorInvalidDevice;
    void *c = calloc(1, 1);
    if (!c)
        return mudaErrorOutOfMemory;
    *ctx = c;
    return mudaSuccess;
}

static void nv_ctx_destroy(void *ctx) { free(ctx); }

/* ------------------------------------------------------------------ */
/* memory (GEM)                                                       */
/* ------------------------------------------------------------------ */
static int nv_malloc(void *drv_ctx, void **ptr, size_t size)
{
    (void)drv_ctx;
    if (!ptr || size == 0 || g_nv.fd < 0)
        return mudaErrorInvalidValue;

    nv_buf *b = (nv_buf *)calloc(1, sizeof(nv_buf));
    if (!b)
        return mudaErrorOutOfMemory;

    struct drm_nouveau_gem_new req;
    memset(&req, 0, sizeof(req));
    req.domain = NOUVEAU_GEM_DOMAIN_GART;
    req.u.info.size = size;

    if (ioctl(g_nv.fd, DRM_IOCTL_NOUVEAU_GEM_NEW, &req) != 0) {
        free(b);
        return mudaErrorOutOfMemory;
    }

    b->st = &g_nv;
    b->handle = req.u.info.handle;
    b->map_handle = req.u.info.map_handle;
    b->size = size;
    b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                  g_nv.fd, (off_t)req.u.info.map_handle);
    if (b->map == MAP_FAILED) {
        struct drm_gem_close c = { req.u.info.handle, 0 };
        ioctl(g_nv.fd, DRM_IOCTL_GEM_CLOSE, &c);
        free(b);
        return mudaErrorOutOfMemory;
    }

    *ptr = b;
    return mudaSuccess;
}

static int nv_malloc_host(void *drv_ctx, void **ptr, size_t size)
{
    (void)drv_ctx;
    if (!ptr || size == 0)
        return mudaErrorInvalidValue;
    void *p = malloc(size);
    if (!p)
        return mudaErrorOutOfMemory;
    *ptr = p;
    return mudaSuccess;
}

static int nv_free(void *drv_ctx, void *ptr)
{
    (void)drv_ctx;
    nv_buf *b = (nv_buf *)ptr;
    if (!b)
        return mudaSuccess;
    if (b->map && b->map != MAP_FAILED)
        munmap(b->map, b->size);
    if (g_nv.fd >= 0) {
        struct drm_gem_close c = { b->handle, 0 };
        ioctl(g_nv.fd, DRM_IOCTL_GEM_CLOSE, &c);
    }
    free(b);
    return mudaSuccess;
}

static int nv_free_host(void *drv_ctx, void *ptr)
{
    (void)drv_ctx;
    free(ptr);
    return mudaSuccess;
}

static int nv_mem_get_info(void *drv_ctx, size_t *freeBytes, size_t *totalBytes)
{
    (void)drv_ctx;
    size_t total = g_nv.vram_bytes ? g_nv.vram_bytes : read_sysfs_size("0000:01:00.0", 1);
    if (totalBytes)
        *totalBytes = total;
    if (freeBytes)
        *freeBytes = total;
    return mudaSuccess;
}

static int nv_memcpy(void *drv_ctx, void *dst, const void *src, size_t size, int kind)
{
    (void)drv_ctx;
    (void)kind;
    if (size)
        memcpy(dst, src, size);
    return mudaSuccess;
}

static int nv_memcpy_async(void *drv_ctx, void *dst, const void *src, size_t size,
                           int kind, void *stream)
{
    (void)stream;
    return nv_memcpy(drv_ctx, dst, src, size, kind);
}

static int nv_memcpy_2d(void *drv_ctx, void *dst, size_t dpitch, const void *src,
                        size_t spitch, size_t width, size_t height, int kind,
                        int async, void *stream)
{
    (void)drv_ctx; (void)kind; (void)async; (void)stream;
    for (size_t r = 0; r < height; r++)
        memcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, width);
    return mudaSuccess;
}

static int nv_memset(void *drv_ctx, void *ptr, int value, size_t size)
{
    (void)drv_ctx;
    if (ptr)
        memset(ptr, value, size);
    return mudaSuccess;
}

static int nv_memset_d32(void *drv_ctx, void *ptr, unsigned value, size_t count)
{
    (void)drv_ctx;
    for (size_t i = 0; i < count; i++)
        ((unsigned *)ptr)[i] = value;
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* modules / kernels (not implemented yet)                            */
/* ------------------------------------------------------------------ */
static int nv_module_load(void *c, const char *src, void **mod)
{
    (void)c; (void)src; (void)mod;
    return mudaErrorNotSupported;
}
static int nv_module_unload(void *c, void *mod) { (void)c; (void)mod; return mudaSuccess; }
static int nv_func_get(void *c, void *mod, const char *n, void **f)
{
    (void)c; (void)mod; (void)n; (void)f;
    return mudaErrorNotSupported;
}
static int nv_func_get_attributes(void *c, void *f, mudaFuncAttributes *a)
{
    (void)c; (void)f; (void)a;
    return mudaErrorNotSupported;
}
static int nv_launch(void *c, void *f, unsigned g[3], unsigned b[3], unsigned s,
                     void *st, void **a, void *p, size_t ps)
{
    (void)c; (void)f; (void)g; (void)b; (void)s; (void)st; (void)a; (void)p; (void)ps;
    return mudaErrorNotSupported;
}

/* ------------------------------------------------------------------ */
/* streams / events                                                   */
/* ------------------------------------------------------------------ */
static int nv_stream_create(void *c, void **stream, unsigned flags)
{
    (void)c; (void)flags;
    void *s = calloc(1, 1);
    if (!s)
        return mudaErrorOutOfMemory;
    *stream = s;
    return mudaSuccess;
}
static void nv_stream_destroy(void *c, void *s) { (void)c; free(s); }
static int nv_stream_sync(void *c, void *s) { (void)c; (void)s; return mudaSuccess; }
static int nv_stream_query(void *c, void *s) { (void)c; (void)s; return mudaSuccess; }
static int nv_stream_wait_event(void *c, void *s, void *e)
{
    (void)c; (void)s; (void)e;
    return mudaSuccess;
}
static int nv_device_sync(void *c) { (void)c; return mudaSuccess; }

typedef struct { double t0, t1; } nv_event;

static int nv_event_create(void *c, void **e, unsigned flags)
{
    (void)c; (void)flags;
    nv_event *ev = (nv_event *)calloc(1, sizeof(nv_event));
    if (!ev)
        return mudaErrorOutOfMemory;
    *e = ev;
    return mudaSuccess;
}
static void nv_event_destroy(void *c, void *e) { (void)c; free(e); }
static int nv_event_record(void *c, void *e, void *s)
{
    (void)c; (void)s;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    ((nv_event *)e)->t0 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
    ((nv_event *)e)->t1 = ((nv_event *)e)->t0;
    return mudaSuccess;
}
static int nv_event_sync(void *c, void *e)
{
    (void)c;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    ((nv_event *)e)->t1 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
    return mudaSuccess;
}
static int nv_event_query(void *c, void *e) { (void)c; (void)e; return mudaSuccess; }
static int nv_event_elapsed(void *c, float *ms, void *a, void *b)
{
    (void)c;
    if (!ms)
        return mudaErrorInvalidValue;
    *ms = (float)(((nv_event *)b)->t1 - ((nv_event *)a)->t0);
    return mudaSuccess;
}

const muda_driver muda_driver_nouveau = {
    .name                  = "nouveau",
    .init                  = nv_init,
    .shutdown              = nv_shutdown,
    .get_device_count      = nv_get_device_count,
    .get_device_props      = nv_get_device_props,
    .get_device_attr       = nv_get_device_attr,
    .get_device_pci_bus_id = nv_get_device_pci_bus_id,
    .ctx_create            = nv_ctx_create,
    .ctx_destroy           = nv_ctx_destroy,
    .malloc                = nv_malloc,
    .malloc_host           = nv_malloc_host,
    .free                  = nv_free,
    .free_host             = nv_free_host,
    .mem_get_info          = nv_mem_get_info,
    .memcpy                = nv_memcpy,
    .memcpy_async          = nv_memcpy_async,
    .memcpy_2d             = nv_memcpy_2d,
    .memset                = nv_memset,
    .memset_d32            = nv_memset_d32,
    .module_load           = nv_module_load,
    .module_unload         = nv_module_unload,
    .func_get              = nv_func_get,
    .func_get_attributes   = nv_func_get_attributes,
    .launch                = nv_launch,
    .stream_create         = nv_stream_create,
    .stream_destroy        = nv_stream_destroy,
    .stream_sync           = nv_stream_sync,
    .stream_query          = nv_stream_query,
    .stream_wait_event     = nv_stream_wait_event,
    .device_sync           = nv_device_sync,
    .event_create          = nv_event_create,
    .event_destroy         = nv_event_destroy,
    .event_record          = nv_event_record,
    .event_sync            = nv_event_sync,
    .event_query           = nv_event_query,
    .event_elapsed         = nv_event_elapsed,
};
