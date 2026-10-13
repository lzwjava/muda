/*
 * driver_opencl.c - the default muda device driver.
 *
 * This backend executes kernels on the GPU by translating CUDA-style source
 * to OpenCL C and dispatching through the system OpenCL installable client
 * driver (ICD).  It never touches libcuda.
 *
 * On the Tesla M40 the NVIDIA OpenCL runtime exposes Maxwell (sm_52) even
 * though recent CUDA toolkits have dropped it, which is precisely why this
 * backend exists.
 */

#include "muda_internal.h"
#include "cl_min.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dlfcn.h>
#include <time.h>

#define OCL_MAX_DEVICES 16
#define ARG_UNKNOWN ((size_t)-1)

static int is_ident_start_char(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static int is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

/* ------------------------------------------------------------------ */
/* OpenCL runtime loader                                              */
/* ------------------------------------------------------------------ */
cl_api g_cl;
int    g_cl_loaded = 0;

static void *g_cl_lib = NULL;

#define CL_LOAD_SYM(field)                                                    \
    do {                                                                      \
        *(void **)(&g_cl.field) = dlsym(g_cl_lib, "cl" #field);               \
        if (!g_cl.field) {                                                    \
            fprintf(stderr, "muda: libOpenCL missing symbol cl%s\n", #field); \
            goto fail;                                                        \
        }                                                                     \
    } while (0)

int muda_cl_load(void)
{
    if (g_cl_loaded)
        return 0;

    const char *names[] = { "libOpenCL.so.1", "libOpenCL.so", NULL };
    for (int i = 0; names[i]; i++) {
        g_cl_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (g_cl_lib)
            break;
    }
    if (!g_cl_lib) {
        fprintf(stderr, "muda: cannot load libOpenCL: %s\n", dlerror());
        return -1;
    }

    memset(&g_cl, 0, sizeof(g_cl));
    CL_LOAD_SYM(GetPlatformIDs);
    CL_LOAD_SYM(GetPlatformInfo);
    CL_LOAD_SYM(GetDeviceIDs);
    CL_LOAD_SYM(GetDeviceInfo);
    CL_LOAD_SYM(CreateContext);
    CL_LOAD_SYM(CreateCommandQueue);
    CL_LOAD_SYM(CreateProgramWithSource);
    CL_LOAD_SYM(BuildProgram);
    CL_LOAD_SYM(GetProgramBuildInfo);
    CL_LOAD_SYM(CreateKernel);
    CL_LOAD_SYM(ReleaseKernel);
    CL_LOAD_SYM(ReleaseProgram);
    CL_LOAD_SYM(CreateBuffer);
    CL_LOAD_SYM(ReleaseMemObject);
    CL_LOAD_SYM(SetKernelArg);
    CL_LOAD_SYM(EnqueueNDRangeKernel);
    CL_LOAD_SYM(EnqueueReadBuffer);
    CL_LOAD_SYM(EnqueueWriteBuffer);
    CL_LOAD_SYM(EnqueueCopyBuffer);
    CL_LOAD_SYM(EnqueueFillBuffer);
    CL_LOAD_SYM(EnqueueMarkerWithWaitList);
    CL_LOAD_SYM(Finish);
    CL_LOAD_SYM(Flush);
    CL_LOAD_SYM(ReleaseCommandQueue);
    CL_LOAD_SYM(ReleaseContext);
    CL_LOAD_SYM(WaitForEvents);
    CL_LOAD_SYM(ReleaseEvent);
    CL_LOAD_SYM(GetEventProfilingInfo);
    CL_LOAD_SYM(GetEventInfo);
    CL_LOAD_SYM(GetKernelWorkGroupInfo);

    g_cl_loaded = 1;
    return 0;

fail:
    dlclose(g_cl_lib);
    g_cl_lib = NULL;
    return -1;
}

void muda_cl_unload(void)
{
    if (g_cl_lib) {
        dlclose(g_cl_lib);
        g_cl_lib = NULL;
    }
    g_cl_loaded = 0;
}

/* ------------------------------------------------------------------ */
/* Driver-private objects                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    size_t size;        /* bytes, or ARG_UNKNOWN */
    int    is_local;    /* dynamic __local memory */
    int    is_pointer;  /* buffer / global pointer */
} ocl_arg;

typedef struct {
    char     *name;
    ocl_arg  *args;
    int       nargs;
} ocl_func_meta;

typedef struct {
    cl_program     prog;
    cl_device_id   dev;
    ocl_func_meta *funcs;
    int            nfuncs;
} ocl_module;

typedef struct {
    cl_kernel            kernel;
    const ocl_func_meta *meta;
} ocl_func;

typedef struct {
    cl_context       ctx;
    cl_command_queue q;
    cl_event         last;
    cl_device_id     dev;
} ocl_ctx;

typedef struct {
    cl_event ev;
    double   host_start_ms;
    double   host_end_ms;
    int      disable_timing;
} ocl_event;

typedef struct {
    cl_command_queue q;
    cl_event         last;
    unsigned         flags;
} ocl_stream;

static cl_platform_id g_platform;
static cl_device_id   g_devices[OCL_MAX_DEVICES];
static int            g_ndev;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Kernel signature parser                                            */
/* ------------------------------------------------------------------ */
static size_t match_paren(const char *s, size_t open)
{
    int depth = 0;
    for (size_t i = open; s[i]; i++) {
        if (s[i] == '(')
            depth++;
        else if (s[i] == ')') {
            depth--;
            if (depth == 0)
                return i;
        }
    }
    return 0;
}

static size_t scalar_type_size(const char *param, size_t len)
{
    static const struct { const char *name; size_t size; } kTypes[] = {
        { "char", 1 }, { "uchar", 1 }, { "bool", 1 },
        { "short", 2 }, { "ushort", 2 },
        { "int", 4 }, { "uint", 4 }, { "unsigned", 4 }, { "signed", 4 },
        { "float", 4 },
        { "long", 8 }, { "ulong", 8 }, { "size_t", 8 }, { "double", 8 },
        { "float2", 8 }, { "int2", 8 }, { "uint2", 8 }, { "double2", 16 },
        { "float3", 16 }, { "int3", 16 }, { "uint3", 16 },
        { "float4", 16 }, { "int4", 16 }, { "uint4", 16 }, { "double4", 32 },
        { "float8", 32 }, { "float16", 64 },
    };
    size_t best = ARG_UNKNOWN;

    for (size_t i = 0; i < len;) {
        if (is_ident_start_char(param[i])) {
            size_t s = i;
            while (i < len && is_ident_char(param[i]))
                i++;
            size_t n = i - s;
            for (size_t k = 0; k < sizeof(kTypes) / sizeof(kTypes[0]); k++) {
                if (strlen(kTypes[k].name) == n &&
                    strncmp(param + s, kTypes[k].name, n) == 0) {
                    if (best == ARG_UNKNOWN || kTypes[k].size > best)
                        best = kTypes[k].size;
                }
            }
        } else {
            i++;
        }
    }
    return best;
}

static int contains_word(const char *s, size_t len, const char *word)
{
    size_t wl = strlen(word);
    for (size_t i = 0; i + wl <= len; i++) {
        if (strncmp(s + i, word, wl) == 0) {
            int left = (i == 0) || !is_ident_char(s[i - 1]);
            int right = (i + wl == len) || !is_ident_char(s[i + wl]);
            if (left && right)
                return 1;
        }
    }
    return 0;
}

static void parse_kernels(const char *src, ocl_module *mod)
{
    const char *p = src;
    int cap = 8;
    mod->funcs = (ocl_func_meta *)calloc(cap, sizeof(ocl_func_meta));
    mod->nfuncs = 0;

    while ((p = strstr(p, "__kernel")) != NULL) {
        if (p != src && is_ident_char(p[-1])) {
            p += 8;
            continue;
        }
        const char *after = p + 8;
        if (is_ident_char(*after)) {
            p = after;
            continue;
        }

        const char *paren = strchr(after, '(');
        if (!paren)
            break;
        const char *name_end = paren;
        while (name_end > after && isspace((unsigned char)name_end[-1]))
            name_end--;
        const char *name_start = name_end;
        while (name_start > after && is_ident_char(name_start[-1]))
            name_start--;
        size_t name_len = (size_t)(name_end - name_start);

        size_t open = (size_t)(paren - src);
        size_t close = match_paren(src, open);
        if (close == 0)
            break;

        if (mod->nfuncs == cap) {
            cap *= 2;
            mod->funcs = (ocl_func_meta *)realloc(mod->funcs, cap * sizeof(ocl_func_meta));
        }
        ocl_func_meta *meta = &mod->funcs[mod->nfuncs++];
        memset(meta, 0, sizeof(*meta));
        meta->name = (char *)malloc(name_len + 1);
        memcpy(meta->name, name_start, name_len);
        meta->name[name_len] = '\0';

        int nargs = 0;
        for (size_t i = open + 1; i < close; i++)
            if (src[i] == ',')
                nargs++;
        if (close > open + 1) {
            int only_space = 1;
            for (size_t i = open + 1; i < close; i++)
                if (!isspace((unsigned char)src[i]))
                    only_space = 0;
            if (!only_space)
                nargs++;
        }
        meta->nargs = nargs;
        meta->args = nargs ? (ocl_arg *)calloc(nargs, sizeof(ocl_arg)) : NULL;

        size_t seg = open + 1;
        int idx = 0;
        for (size_t i = open + 1; i <= close && idx < nargs; i++) {
            if (i == close || src[i] == ',') {
                const char *a = src + seg;
                size_t alen = i - seg;
                while (alen && isspace((unsigned char)*a)) { a++; alen--; }
                while (alen && isspace((unsigned char)a[alen - 1])) alen--;

                ocl_arg *arg = &meta->args[idx++];
                if (contains_word(a, alen, "__local")) {
                    arg->is_local = 1;
                    arg->size = 0;
                } else if (memchr(a, '*', alen)) {
                    arg->is_pointer = 1;
                    arg->size = sizeof(cl_mem);
                } else {
                    arg->size = scalar_type_size(a, alen);
                }
                seg = i + 1;
            }
        }
        p = src + close + 1;
    }
}

/* ------------------------------------------------------------------ */
/* Stream helpers                                                     */
/* ------------------------------------------------------------------ */
static cl_command_queue stream_queue(ocl_ctx *c, void *stream)
{
    return stream ? ((ocl_stream *)stream)->q : c->q;
}

static void stream_set_last(ocl_ctx *c, void *stream, cl_event ev)
{
    cl_event *slot = stream ? &((ocl_stream *)stream)->last : &c->last;
    if (*slot)
        g_cl.ReleaseEvent(*slot);
    *slot = ev;
}

static cl_event stream_get_last(ocl_ctx *c, void *stream)
{
    return stream ? ((ocl_stream *)stream)->last : c->last;
}

/* ------------------------------------------------------------------ */
/* Device queries                                                     */
/* ------------------------------------------------------------------ */
static int opencl_init(void)
{
    if (muda_cl_load() != 0)
        return mudaErrorNoDevice;

    cl_uint nplat = 0;
    if (g_cl.GetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0)
        return mudaErrorNoDevice;

    cl_platform_id plats[8];
    if (nplat > 8) nplat = 8;
    g_cl.GetPlatformIDs(nplat, plats, NULL);

    g_platform = plats[0];
    for (cl_uint i = 0; i < nplat; i++) {
        char name[256] = {0};
        g_cl.GetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        if (strstr(name, "NVIDIA")) {
            g_platform = plats[i];
            break;
        }
        if (!g_platform)
            g_platform = plats[i];
    }

    cl_uint ndev = 0;
    if (g_cl.GetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, OCL_MAX_DEVICES,
                          g_devices, &ndev) != CL_SUCCESS || ndev == 0) {
        if (g_cl.GetDeviceIDs(g_platform, CL_DEVICE_TYPE_ALL, OCL_MAX_DEVICES,
                              g_devices, &ndev) != CL_SUCCESS)
            return mudaErrorNoDevice;
    }
    g_ndev = (int)ndev;
    return g_ndev > 0 ? mudaSuccess : mudaErrorNoDevice;
}

static void opencl_shutdown(void)
{
    g_ndev = 0;
    muda_cl_unload();
}

static int opencl_get_device_count(int *count)
{
    if (!count)
        return mudaErrorInvalidValue;
    *count = g_ndev;
    return mudaSuccess;
}

static void get_str(cl_device_id d, cl_uint param, char *dst, size_t cap)
{
    if (cap == 0)
        return;
    dst[0] = '\0';
    g_cl.GetDeviceInfo(d, param, cap, dst, NULL);
}

static int fill_prop(int device, mudaDeviceProp *prop)
{
    if (device < 0 || device >= g_ndev)
        return mudaErrorInvalidDevice;

    memset(prop, 0, sizeof(*prop));
    cl_device_id d = g_devices[device];

    get_str(d, CL_DEVICE_NAME, prop->name, sizeof(prop->name));
    get_str(d, CL_DRIVER_VERSION, prop->driverVersion, sizeof(prop->driverVersion));
    get_str(d, CL_DEVICE_VERSION, prop->deviceVersion, sizeof(prop->deviceVersion));

    cl_uint u = 0;
    g_cl.GetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(u), &u, NULL);
    prop->multiProcessorCount = (int)u;

    size_t sz = 0;
    g_cl.GetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(sz), &sz, NULL);
    prop->totalGlobalMem = sz;
    g_cl.GetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(sz), &sz, NULL);
    prop->sharedMemPerBlock = sz;
    prop->sharedMemPerMultiprocessor = sz;

    g_cl.GetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sz), &sz, NULL);
    prop->maxThreadsPerBlock = (int)sz;

    size_t dims[3] = {0, 0, 0};
    if (g_cl.GetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, NULL) == CL_SUCCESS) {
        for (int i = 0; i < 3; i++)
            prop->maxThreadsDim[i] = (int)dims[i];
    }
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;

    g_cl.GetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(u), &u, NULL);
    prop->clockRateKHz = (int)u * 1000;

    prop->maxRegistersPerBlock = 65536;
    prop->memoryClockRateKHz = 0;
    prop->memoryBusWidth = 0;
    prop->memPitch = 2147483647UL;
    prop->concurrentKernels = 1;

    cl_uint cc_major = 0, cc_minor = 0, warp = 0;
    g_cl.GetDeviceInfo(d, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cc_major), &cc_major, NULL);
    g_cl.GetDeviceInfo(d, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cc_minor), &cc_minor, NULL);
    g_cl.GetDeviceInfo(d, CL_DEVICE_WARP_SIZE_NV, sizeof(warp), &warp, NULL);
    prop->computeCapabilityMajor = (int)cc_major;
    prop->computeCapabilityMinor = (int)cc_minor;
    prop->warpSize = warp ? (int)warp : 32;

    cl_uint v = 0;
    if (g_cl.GetDeviceInfo(d, CL_DEVICE_PCI_BUS_ID_NV, sizeof(v), &v, NULL) == CL_SUCCESS)
        prop->pciBusID = (int)v;
    if (g_cl.GetDeviceInfo(d, CL_DEVICE_PCI_SLOT_ID_NV, sizeof(v), &v, NULL) == CL_SUCCESS)
        prop->pciDeviceID = (int)v;
    if (g_cl.GetDeviceInfo(d, CL_DEVICE_PCI_DOMAIN_ID_NV, sizeof(v), &v, NULL) == CL_SUCCESS)
        prop->pciDomainID = (int)v;

    return mudaSuccess;
}

static int opencl_get_device_props(int device, mudaDeviceProp *prop)
{
    if (!prop)
        return mudaErrorInvalidValue;
    return fill_prop(device, prop);
}

static int opencl_get_device_attr(int device, int attr, int *value)
{
    if (!value)
        return mudaErrorInvalidValue;
    mudaDeviceProp p;
    int e = fill_prop(device, &p);
    if (e != mudaSuccess)
        return e;

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
    case mudaDevAttrTotalConstantMemory:     *value = 65536; break;
    case mudaDevAttrWarpSize:                *value = p.warpSize; break;
    case mudaDevAttrMaxPitch:                *value = (int)p.memPitch; break;
    case mudaDevAttrMaxRegistersPerBlock:    *value = p.maxRegistersPerBlock; break;
    case mudaDevAttrClockRate:               *value = p.clockRateKHz; break;
    case mudaDevAttrMultiProcessorCount:     *value = p.multiProcessorCount; break;
    case mudaDevAttrConcurrentKernels:       *value = p.concurrentKernels; break;
    case mudaDevAttrMemoryClockRate:         *value = p.memoryClockRateKHz; break;
    case mudaDevAttrGlobalMemoryBusWidth:    *value = p.memoryBusWidth; break;
    case mudaDevAttrComputeCapabilityMajor:  *value = p.computeCapabilityMajor; break;
    case mudaDevAttrComputeCapabilityMinor:  *value = p.computeCapabilityMinor; break;
    case mudaDevAttrPciBusId:                *value = p.pciBusID; break;
    case mudaDevAttrPciDeviceId:             *value = p.pciDeviceID; break;
    case mudaDevAttrPciDomainId:             *value = p.pciDomainID; break;
    default:                                 return mudaErrorInvalidValue;
    }
    return mudaSuccess;
}

static int opencl_get_device_pci_bus_id(int device, char *buf, int len)
{
    if (!buf || len <= 0 || device < 0 || device >= g_ndev)
        return mudaErrorInvalidValue;
    mudaDeviceProp p;
    if (fill_prop(device, &p) != mudaSuccess)
        return mudaErrorInvalidDevice;
    snprintf(buf, (size_t)len, "%04x:%02x:%02x.%x",
             p.pciDomainID, p.pciBusID, p.pciDeviceID >> 3, p.pciDeviceID & 7);
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* Context                                                            */
/* ------------------------------------------------------------------ */
static int opencl_ctx_create(void **drv_ctx, int device)
{
    if (device < 0 || device >= g_ndev)
        return mudaErrorInvalidDevice;

    ocl_ctx *c = (ocl_ctx *)calloc(1, sizeof(ocl_ctx));
    if (!c)
        return mudaErrorOutOfMemory;
    c->dev = g_devices[device];

    cl_int err = CL_SUCCESS;
    c->ctx = g_cl.CreateContext(NULL, 1, &c->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !c->ctx) {
        free(c);
        return mudaErrorInvalidContext;
    }
    c->q = g_cl.CreateCommandQueue(c->ctx, c->dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !c->q) {
        g_cl.ReleaseContext(c->ctx);
        free(c);
        return mudaErrorInvalidContext;
    }
    *drv_ctx = c;
    return mudaSuccess;
}

static void opencl_ctx_destroy(void *drv_ctx)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    if (!c)
        return;
    if (c->last)
        g_cl.ReleaseEvent(c->last);
    if (c->q)
        g_cl.ReleaseCommandQueue(c->q);
    if (c->ctx)
        g_cl.ReleaseContext(c->ctx);
    free(c);
}

/* ------------------------------------------------------------------ */
/* Memory                                                             */
/* ------------------------------------------------------------------ */
static int opencl_malloc(void *drv_ctx, void **ptr, size_t size)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    if (!ptr || size == 0)
        return mudaErrorInvalidValue;
    cl_int err;
    cl_mem m = g_cl.CreateBuffer(c->ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS || !m)
        return mudaErrorOutOfMemory;
    *ptr = m;
    return mudaSuccess;
}

static int opencl_malloc_host(void *drv_ctx, void **ptr, size_t size)
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

static int opencl_free(void *drv_ctx, void *ptr)
{
    (void)drv_ctx;
    if (ptr)
        g_cl.ReleaseMemObject((cl_mem)ptr);
    return mudaSuccess;
}

static int opencl_free_host(void *drv_ctx, void *ptr)
{
    (void)drv_ctx;
    free(ptr);
    return mudaSuccess;
}

static int opencl_mem_get_info(void *drv_ctx, size_t *freeBytes, size_t *totalBytes)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_ulong total = 0;
    g_cl.GetDeviceInfo(c->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total), &total, NULL);
    if (totalBytes)
        *totalBytes = (size_t)total;
    if (freeBytes)
        *freeBytes = (size_t)total;
    return mudaSuccess;
}

static int opencl_memcpy(void *drv_ctx, void *dst, const void *src, size_t size, int kind)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_int err = CL_SUCCESS;

    switch (kind) {
    case mudaMemcpyHostToHost:
        memcpy(dst, src, size);
        break;
    case mudaMemcpyHostToDevice:
        err = g_cl.EnqueueWriteBuffer(c->q, (cl_mem)dst, CL_TRUE, 0, size, src, 0, NULL, NULL);
        break;
    case mudaMemcpyDeviceToHost:
        err = g_cl.EnqueueReadBuffer(c->q, (cl_mem)src, CL_TRUE, 0, size, dst, 0, NULL, NULL);
        break;
    case mudaMemcpyDeviceToDevice:
        err = g_cl.EnqueueCopyBuffer(c->q, (cl_mem)src, (cl_mem)dst, 0, 0, size, 0, NULL, NULL);
        break;
    default:
        return mudaErrorInvalidValue;
    }
    return err == CL_SUCCESS ? mudaSuccess : mudaErrorUnknown;
}

static int opencl_memcpy_async(void *drv_ctx, void *dst, const void *src, size_t size,
                               int kind, void *stream)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_command_queue q = stream_queue(c, stream);
    cl_event ev = NULL;
    cl_int err = CL_SUCCESS;

    switch (kind) {
    case mudaMemcpyHostToHost:
        memcpy(dst, src, size);
        return mudaSuccess;
    case mudaMemcpyHostToDevice:
        err = g_cl.EnqueueWriteBuffer(q, (cl_mem)dst, CL_FALSE, 0, size, src, 0, NULL, &ev);
        break;
    case mudaMemcpyDeviceToHost:
        err = g_cl.EnqueueReadBuffer(q, (cl_mem)src, CL_FALSE, 0, size, dst, 0, NULL, &ev);
        break;
    case mudaMemcpyDeviceToDevice:
        err = g_cl.EnqueueCopyBuffer(q, (cl_mem)src, (cl_mem)dst, 0, 0, size, 0, NULL, &ev);
        break;
    default:
        return mudaErrorInvalidValue;
    }
    if (err != CL_SUCCESS)
        return mudaErrorUnknown;
    stream_set_last(c, stream, ev);
    return mudaSuccess;
}

static int opencl_memcpy_2d(void *drv_ctx, void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height, int kind,
                            int async, void *stream)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_command_queue q = stream_queue(c, stream);
    cl_uint blocking = async ? CL_FALSE : CL_TRUE;
    cl_event ev = NULL;
    cl_int err = CL_SUCCESS;

    for (size_t row = 0; row < height; row++) {
        const char *s = (const char *)src + row * spitch;
        char *d = (char *)dst + row * dpitch;
        switch (kind) {
        case mudaMemcpyHostToHost:
            memcpy(d, s, width);
            break;
        case mudaMemcpyHostToDevice:
            err = g_cl.EnqueueWriteBuffer(q, (cl_mem)dst, CL_TRUE,
                                          row * dpitch, width, s, 0, NULL, NULL);
            break;
        case mudaMemcpyDeviceToHost:
            err = g_cl.EnqueueReadBuffer(q, (cl_mem)src, CL_TRUE,
                                         row * spitch, width, d, 0, NULL, NULL);
            break;
        case mudaMemcpyDeviceToDevice:
            err = g_cl.EnqueueCopyBuffer(q, (cl_mem)src, (cl_mem)dst,
                                         row * spitch, row * dpitch, width, 0, NULL, NULL);
            break;
        default:
            return mudaErrorInvalidValue;
        }
        if (err != CL_SUCCESS)
            return mudaErrorUnknown;
    }

    /* For the async host<->device cases we issued blocking per-row copies
     * above; convert to a single async marker so the stream is queryable. */
    if (async && kind != mudaMemcpyHostToHost) {
        err = g_cl.EnqueueMarkerWithWaitList(q, 0, NULL, &ev);
        if (err == CL_SUCCESS)
            stream_set_last(c, stream, ev);
    }
    (void)blocking;
    return mudaSuccess;
}

static int opencl_memset(void *drv_ctx, void *ptr, int value, size_t size)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    unsigned char byte = (unsigned char)value;
    cl_int err = CL_SUCCESS;

    if (g_cl.EnqueueFillBuffer) {
        err = g_cl.EnqueueFillBuffer(c->q, (cl_mem)ptr, &byte, 1, 0, size, 0, NULL, NULL);
        if (err == CL_SUCCESS)
            return mudaSuccess;
    }
    unsigned char *tmp = (unsigned char *)malloc(size);
    if (!tmp)
        return mudaErrorOutOfMemory;
    memset(tmp, value, size);
    err = g_cl.EnqueueWriteBuffer(c->q, (cl_mem)ptr, CL_TRUE, 0, size, tmp, 0, NULL, NULL);
    free(tmp);
    return err == CL_SUCCESS ? mudaSuccess : mudaErrorUnknown;
}

static int opencl_memset_d32(void *drv_ctx, void *ptr, unsigned value, size_t count)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_int err = g_cl.EnqueueFillBuffer(c->q, (cl_mem)ptr, &value, sizeof(value),
                                        0, count * sizeof(unsigned), 0, NULL, NULL);
    return err == CL_SUCCESS ? mudaSuccess : mudaErrorUnknown;
}

/* ------------------------------------------------------------------ */
/* Modules / kernels                                                  */
/* ------------------------------------------------------------------ */
static int opencl_module_load(void *drv_ctx, const char *source, void **mod_out)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    char *opencl_src = muda_translate_cuda_to_opencl(source);
    if (!opencl_src)
        return mudaErrorOutOfMemory;

    cl_int err = CL_SUCCESS;
    const char *strings[1] = { opencl_src };
    cl_program prog = g_cl.CreateProgramWithSource(c->ctx, 1, strings, NULL, &err);
    free(opencl_src);
    if (err != CL_SUCCESS || !prog)
        return mudaErrorInvalidValue;

    err = g_cl.BuildProgram(prog, 1, &c->dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS || getenv("MUDA_VERBOSE")) {
        size_t log_size = 0;
        g_cl.GetProgramBuildInfo(prog, c->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 0) {
            char *log = (char *)malloc(log_size + 1);
            if (log) {
                g_cl.GetProgramBuildInfo(prog, c->dev, CL_PROGRAM_BUILD_LOG,
                                         log_size, log, NULL);
                log[log_size] = '\0';
                if (err != CL_SUCCESS)
                    fprintf(stderr, "muda: OpenCL build failed:\n%s\n", log);
                else
                    fprintf(stderr, "muda: OpenCL build log:\n%s\n", log);
                free(log);
            }
        }
    }
    if (err != CL_SUCCESS) {
        g_cl.ReleaseProgram(prog);
        return mudaErrorLaunchFailure;
    }

    ocl_module *mod = (ocl_module *)calloc(1, sizeof(ocl_module));
    if (!mod) {
        g_cl.ReleaseProgram(prog);
        return mudaErrorOutOfMemory;
    }
    mod->prog = prog;
    mod->dev = c->dev;

    char *opencl_src2 = muda_translate_cuda_to_opencl(source);
    if (opencl_src2) {
        parse_kernels(opencl_src2, mod);
        free(opencl_src2);
    }

    *mod_out = mod;
    return mudaSuccess;
}

static int opencl_module_unload(void *drv_ctx, void *mod_ptr)
{
    (void)drv_ctx;
    ocl_module *mod = (ocl_module *)mod_ptr;
    if (!mod)
        return mudaSuccess;
    if (mod->funcs) {
        for (int i = 0; i < mod->nfuncs; i++) {
            free(mod->funcs[i].name);
            free(mod->funcs[i].args);
        }
        free(mod->funcs);
    }
    if (mod->prog)
        g_cl.ReleaseProgram(mod->prog);
    free(mod);
    return mudaSuccess;
}

static int opencl_func_get(void *drv_ctx, void *mod_ptr, const char *name, void **fn_out)
{
    ocl_module *mod = (ocl_module *)mod_ptr;
    (void)drv_ctx;

    const ocl_func_meta *meta = NULL;
    for (int i = 0; i < mod->nfuncs; i++) {
        if (strcmp(mod->funcs[i].name, name) == 0) {
            meta = &mod->funcs[i];
            break;
        }
    }
    if (!meta)
        return mudaErrorNotFound;

    cl_int err = CL_SUCCESS;
    cl_kernel k = g_cl.CreateKernel(mod->prog, name, &err);
    if (err != CL_SUCCESS || !k)
        return mudaErrorNotFound;

    ocl_func *f = (ocl_func *)calloc(1, sizeof(ocl_func));
    if (!f) {
        g_cl.ReleaseKernel(k);
        return mudaErrorOutOfMemory;
    }
    f->kernel = k;
    f->meta = meta;
    *fn_out = f;
    return mudaSuccess;
}

static int opencl_func_get_attributes(void *drv_ctx, void *fn_ptr, mudaFuncAttributes *attr)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    ocl_func *f = (ocl_func *)fn_ptr;
    if (!attr || !f)
        return mudaErrorInvalidValue;

    memset(attr, 0, sizeof(*attr));
    size_t v = 0;
    if (g_cl.GetKernelWorkGroupInfo(f->kernel, c->dev, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(v), &v, NULL) == CL_SUCCESS)
        attr->maxThreadsPerBlock = (int)v;
    if (g_cl.GetKernelWorkGroupInfo(f->kernel, c->dev, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(v), &v, NULL) == CL_SUCCESS)
        attr->sharedSizeBytes = v;
    if (g_cl.GetKernelWorkGroupInfo(f->kernel, c->dev, CL_KERNEL_PRIVATE_MEM_SIZE,
                                    sizeof(v), &v, NULL) == CL_SUCCESS)
        attr->localSizeBytes = v;
    attr->numRegs = 0;
    return mudaSuccess;
}

static size_t align_up(size_t v, size_t a)
{
    return a ? ((v + a - 1) / a) * a : v;
}

static int opencl_launch(void *drv_ctx, void *fn_ptr, unsigned grid[3], unsigned block[3],
                         unsigned shared, void *stream, void **args,
                         void *packed, size_t packed_size)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    ocl_func *f = (ocl_func *)fn_ptr;
    cl_command_queue q = stream_queue(c, stream);

    size_t offset = 0;
    for (int i = 0; i < f->meta->nargs; i++) {
        const ocl_arg *a = &f->meta->args[i];
        cl_int err;
        if (a->is_local) {
            err = g_cl.SetKernelArg(f->kernel, (cl_uint)i, shared, NULL);
        } else if (packed) {
            size_t align = a->size > 16 ? 16 : a->size;
            offset = align_up(offset, align);
            if (offset + a->size > packed_size)
                return mudaErrorInvalidValue;
            err = g_cl.SetKernelArg(f->kernel, (cl_uint)i, a->size,
                                    (const char *)packed + offset);
            offset += a->size;
        } else {
            if (!args || !args[i])
                return mudaErrorInvalidValue;
            if (a->size == ARG_UNKNOWN)
                return mudaErrorNotSupported;
            err = g_cl.SetKernelArg(f->kernel, (cl_uint)i, a->size, args[i]);
        }
        if (err != CL_SUCCESS)
            return mudaErrorInvalidValue;
    }

    size_t global[3] = {
        (size_t)grid[0] * block[0],
        (size_t)grid[1] * block[1],
        (size_t)grid[2] * block[2]
    };
    size_t local[3] = { block[0], block[1], block[2] };

    cl_event ev = NULL;
    cl_int err = g_cl.EnqueueNDRangeKernel(q, f->kernel, 3, NULL, global, local,
                                           0, NULL, &ev);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "muda: enqueue failed (cl err %d)\n", err);
        return mudaErrorLaunchFailure;
    }
    stream_set_last(c, stream, ev);
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* Streams                                                            */
/* ------------------------------------------------------------------ */
static int opencl_stream_create(void *drv_ctx, void **stream, unsigned flags)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    ocl_stream *s = (ocl_stream *)calloc(1, sizeof(ocl_stream));
    if (!s)
        return mudaErrorOutOfMemory;
    cl_int err;
    s->q = g_cl.CreateCommandQueue(c->ctx, c->dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !s->q) {
        free(s);
        return mudaErrorUnknown;
    }
    s->flags = flags;
    *stream = s;
    return mudaSuccess;
}

static void opencl_stream_destroy(void *drv_ctx, void *stream)
{
    (void)drv_ctx;
    ocl_stream *s = (ocl_stream *)stream;
    if (!s)
        return;
    if (s->last)
        g_cl.ReleaseEvent(s->last);
    if (s->q)
        g_cl.ReleaseCommandQueue(s->q);
    free(s);
}

static int opencl_stream_sync(void *drv_ctx, void *stream)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    return g_cl.Finish(stream_queue(c, stream)) == CL_SUCCESS ? mudaSuccess : mudaErrorUnknown;
}

static int event_status(cl_event ev)
{
    if (!ev)
        return CL_COMPLETE;
    cl_int status = CL_QUEUED;
    if (g_cl.GetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                          sizeof(status), &status, NULL) != CL_SUCCESS)
        return CL_COMPLETE;
    return status;
}

static int opencl_stream_query(void *drv_ctx, void *stream)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    cl_event ev = stream_get_last(c, stream);
    if (!ev)
        return mudaSuccess;
    return event_status(ev) == CL_COMPLETE ? mudaSuccess : mudaErrorNotReady;
}

static int opencl_stream_wait_event(void *drv_ctx, void *stream, void *event)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    ocl_event *e = (ocl_event *)event;
    cl_command_queue q = stream_queue(c, stream);
    cl_event marker = NULL;

    if (!e || !e->ev)
        return mudaSuccess;

    cl_int err = g_cl.EnqueueMarkerWithWaitList(q, 1, &e->ev, &marker);
    if (err != CL_SUCCESS)
        return mudaErrorUnknown;
    stream_set_last(c, stream, marker);
    return mudaSuccess;
}

static int opencl_device_sync(void *drv_ctx)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    return g_cl.Finish(c->q) == CL_SUCCESS ? mudaSuccess : mudaErrorUnknown;
}

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */
static int opencl_event_create(void *drv_ctx, void **event, unsigned flags)
{
    (void)drv_ctx;
    ocl_event *e = (ocl_event *)calloc(1, sizeof(ocl_event));
    if (!e)
        return mudaErrorOutOfMemory;
    e->disable_timing = (flags & mudaEventDisableTiming) ? 1 : 0;
    *event = e;
    return mudaSuccess;
}

static void opencl_event_destroy(void *drv_ctx, void *event)
{
    (void)drv_ctx;
    ocl_event *e = (ocl_event *)event;
    if (!e)
        return;
    if (e->ev)
        g_cl.ReleaseEvent(e->ev);
    free(e);
}

static int opencl_event_record(void *drv_ctx, void *event, void *stream)
{
    ocl_ctx *c = (ocl_ctx *)drv_ctx;
    ocl_event *e = (ocl_event *)event;
    cl_command_queue q = stream_queue(c, stream);

    if (e->ev) {
        g_cl.ReleaseEvent(e->ev);
        e->ev = NULL;
    }
    e->host_start_ms = now_ms();
    e->host_end_ms = e->host_start_ms;

    cl_int err = g_cl.EnqueueMarkerWithWaitList(q, 0, NULL, &e->ev);
    if (err != CL_SUCCESS)
        e->ev = NULL;
    return mudaSuccess;
}

static int opencl_event_sync(void *drv_ctx, void *event)
{
    (void)drv_ctx;
    ocl_event *e = (ocl_event *)event;
    if (e && e->ev) {
        if (g_cl.WaitForEvents(1, &e->ev) != CL_SUCCESS)
            return mudaErrorUnknown;
        e->host_end_ms = now_ms();
    }
    return mudaSuccess;
}

static int opencl_event_query(void *drv_ctx, void *event)
{
    (void)drv_ctx;
    ocl_event *e = (ocl_event *)event;
    if (!e || !e->ev)
        return mudaSuccess;
    return event_status(e->ev) == CL_COMPLETE ? mudaSuccess : mudaErrorNotReady;
}

static int opencl_event_elapsed(void *drv_ctx, float *ms, void *start, void *end)
{
    (void)drv_ctx;
    ocl_event *a = (ocl_event *)start;
    ocl_event *b = (ocl_event *)end;
    if (!ms || !a || !b)
        return mudaErrorInvalidValue;

    if (a->ev && b->ev && !a->disable_timing && !b->disable_timing) {
        cl_ulong ta = 0, tb = 0;
        cl_int e1 = g_cl.GetEventProfilingInfo(a->ev, CL_PROFILING_COMMAND_END,
                                               sizeof(ta), &ta, NULL);
        cl_int e2 = g_cl.GetEventProfilingInfo(b->ev, CL_PROFILING_COMMAND_END,
                                               sizeof(tb), &tb, NULL);
        if (e1 == CL_SUCCESS && e2 == CL_SUCCESS && tb >= ta) {
            *ms = (float)((double)(tb - ta) / 1e6);
            return mudaSuccess;
        }
    }
    *ms = (float)(b->host_end_ms - a->host_start_ms);
    return mudaSuccess;
}

const muda_driver muda_driver_opencl = {
    .name                     = "opencl",
    .init                     = opencl_init,
    .shutdown                 = opencl_shutdown,
    .get_device_count         = opencl_get_device_count,
    .get_device_props         = opencl_get_device_props,
    .get_device_attr          = opencl_get_device_attr,
    .get_device_pci_bus_id    = opencl_get_device_pci_bus_id,
    .ctx_create               = opencl_ctx_create,
    .ctx_destroy              = opencl_ctx_destroy,
    .malloc                   = opencl_malloc,
    .malloc_host              = opencl_malloc_host,
    .free                     = opencl_free,
    .free_host                = opencl_free_host,
    .mem_get_info             = opencl_mem_get_info,
    .memcpy                   = opencl_memcpy,
    .memcpy_async             = opencl_memcpy_async,
    .memcpy_2d                = opencl_memcpy_2d,
    .memset                   = opencl_memset,
    .memset_d32               = opencl_memset_d32,
    .module_load              = opencl_module_load,
    .module_unload            = opencl_module_unload,
    .func_get                 = opencl_func_get,
    .func_get_attributes      = opencl_func_get_attributes,
    .launch                   = opencl_launch,
    .stream_create            = opencl_stream_create,
    .stream_destroy           = opencl_stream_destroy,
    .stream_sync              = opencl_stream_sync,
    .stream_query             = opencl_stream_query,
    .stream_wait_event        = opencl_stream_wait_event,
    .device_sync              = opencl_device_sync,
    .event_create             = opencl_event_create,
    .event_destroy            = opencl_event_destroy,
    .event_record             = opencl_event_record,
    .event_sync               = opencl_event_sync,
    .event_query              = opencl_event_query,
    .event_elapsed            = opencl_event_elapsed,
};
