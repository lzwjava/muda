/*
 * test_muda.c - self-contained test suite for muda.
 *
 * Run with: make test
 */
#include "muda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0;
static int g_fail = 0;
static const char *g_current_test = "";

#define OK(cond)                                                             \
    do {                                                                     \
        if (cond) {                                                          \
            g_pass++;                                                        \
        } else {                                                             \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s\n", g_current_test,              \
                   __FILE__, __LINE__, #cond);                              \
        }                                                                    \
    } while (0)

#define MUDA(expr)                                                           \
    do {                                                                     \
        mudaError_t _e = (expr);                                            \
        if (_e != mudaSuccess) {                                            \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s -> %s\n", g_current_test,        \
                   __FILE__, __LINE__, #expr, mudaGetErrorString(_e));      \
        }                                                                    \
    } while (0)

#define TEST(name)                                                           \
    do {                                                                     \
        g_current_test = name;                                              \
        printf("== %s ==\n", name);                                         \
    } while (0)

/* ------------------------------------------------------------------ */
/* Kernels                                                            */
/* ------------------------------------------------------------------ */
static const char *kKernels =
    "__global__ void vector_add(const float *a, const float *b, float *c, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) c[i] = a[i] + b[i];\n"
    "}\n"
    "\n"
    "__global__ void saxpy(float a, const float *x, float *y, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) y[i] = a * x[i] + y[i];\n"
    "}\n"
    "\n"
    "__global__ void reverse_blocks(const float *in, float *out, int n)\n"
    "{\n"
    "    __shared__ float s[256];\n"
    "    int t = threadIdx.x;\n"
    "    int base = blockIdx.x * blockDim.x;\n"
    "    int i = base + t;\n"
    "    s[t] = (i < n) ? in[i] : 0.0f;\n"
    "    __syncthreads();\n"
    "    int j = base + (blockDim.x - 1 - t);\n"
    "    if (j < n) out[j] = s[t];\n"
    "}\n"
    "\n"
    "__global__ void fill_2d(int *out, int w, int h)\n"
    "{\n"
    "    int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "    if (x < w && y < h) out[y * w + x] = y * w + x;\n"
    "}\n";

static mudaModule_t g_mod;

static mudaFunction_t get_kernel(const char *name)
{
    mudaFunction_t fn = NULL;
    if (mudaModuleGetFunction(&fn, g_mod, name) != mudaSuccess) {
        printf("  [FAIL] cannot find kernel %s\n", name);
        g_fail++;
    }
    return fn;
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */
static void test_device(void)
{
    TEST("device");
    int count = 0;
    MUDA(mudaGetDeviceCount(&count));
    OK(count >= 1);

    mudaDeviceProp p;
    MUDA(mudaGetDeviceProperties(&p, 0));
    OK(strlen(p.name) > 0);
    OK(p.totalGlobalMem > 0);
    OK(p.maxThreadsPerBlock >= 64);
    printf("  device 0: %s sm_%d%d, %.0f MiB, %d SMs\n",
           p.name, p.computeCapabilityMajor, p.computeCapabilityMinor,
           (double)p.totalGlobalMem / 1048576.0, p.multiProcessorCount);
}

static void test_memory(void)
{
    TEST("memory");
    const size_t N = 4096;
    unsigned char *h = (unsigned char *)malloc(N);
    unsigned char *back = (unsigned char *)malloc(N);

    void *d = NULL;
    MUDA(mudaMalloc(&d, N));
    OK(d != NULL);

    for (size_t i = 0; i < N; i++)
        h[i] = (unsigned char)(i * 7 + 3);

    MUDA(mudaMemcpy(d, h, N, mudaMemcpyHostToDevice));

    MUDA(mudaMemset(d, 0xAB, N));
    memset(back, 0, N);
    MUDA(mudaMemcpy(back, d, N, mudaMemcpyDeviceToHost));
    int all_ab = 1;
    for (size_t i = 0; i < N; i++)
        if (back[i] != 0xAB)
            all_ab = 0;
    OK(all_ab);

    /* device to device */
    void *d2 = NULL;
    MUDA(mudaMalloc(&d2, N));
    MUDA(mudaMemcpy(d, h, N, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(d2, d, N, mudaMemcpyDeviceToDevice));
    memset(back, 0, N);
    MUDA(mudaMemcpy(back, d2, N, mudaMemcpyDeviceToHost));
    OK(memcmp(back, h, N) == 0);

    MUDA(mudaFree(d2));
    MUDA(mudaFree(d));
    free(h);
    free(back);
}

static void test_vector_add(void)
{
    TEST("vector_add");
    const int N = 1 << 16;
    const size_t bytes = N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i % 13);
    }

    void *da = NULL, *db = NULL, *dc = NULL;
    MUDA(mudaMalloc(&da, bytes));
    MUDA(mudaMalloc(&db, bytes));
    MUDA(mudaMalloc(&dc, bytes));
    MUDA(mudaMemcpy(da, a, bytes, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(db, b, bytes, mudaMemcpyHostToDevice));

    mudaFunction_t fn = get_kernel("vector_add");
    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    MUDA(mudaLaunchKernel(fn, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(c, dc, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-4f)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(da));
    MUDA(mudaFree(db));
    MUDA(mudaFree(dc));
    free(a);
    free(b);
    free(c);
}

static void test_saxpy(void)
{
    TEST("saxpy");
    const int N = 4096;
    const size_t bytes = N * sizeof(float);
    float *x = malloc(bytes), *y = malloc(bytes), *back = malloc(bytes);
    for (int i = 0; i < N; i++) {
        x[i] = (float)i * 0.5f;
        y[i] = 1.0f;
    }

    void *dx = NULL, *dy = NULL;
    MUDA(mudaMalloc(&dx, bytes));
    MUDA(mudaMalloc(&dy, bytes));
    MUDA(mudaMemcpy(dx, x, bytes, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dy, y, bytes, mudaMemcpyHostToDevice));

    mudaFunction_t fn = get_kernel("saxpy");
    float a = 2.5f;
    int n = N;
    void *args[] = { &a, &dx, &dy, &n };
    MUDA(mudaLaunchKernel(fn, N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(back, dy, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(back[i] - (a * x[i] + y[i])) > 1e-4f)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(dx));
    MUDA(mudaFree(dy));
    free(x);
    free(y);
    free(back);
}

static void test_shared_syncthreads(void)
{
    TEST("shared_memory_+_syncthreads");
    const int N = 1024;
    const int block = 256;
    const size_t bytes = N * sizeof(float);
    float *in = malloc(bytes), *out = malloc(bytes);
    for (int i = 0; i < N; i++)
        in[i] = (float)i;

    void *din = NULL, *dout = NULL;
    MUDA(mudaMalloc(&din, bytes));
    MUDA(mudaMalloc(&dout, bytes));
    MUDA(mudaMemcpy(din, in, bytes, mudaMemcpyHostToDevice));

    mudaFunction_t fn = get_kernel("reverse_blocks");
    int n = N;
    void *args[] = { &din, &dout, &n };
    MUDA(mudaLaunchKernel(fn, N / block, 1, 1, block, 1, 1, 0, NULL, args, NULL));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(out, dout, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int base = 0; base < N; base += block)
        for (int t = 0; t < block; t++)
            if (out[base + (block - 1 - t)] != in[base + t])
                ok = 0;
    OK(ok);

    MUDA(mudaFree(din));
    MUDA(mudaFree(dout));
    free(in);
    free(out);
}

static void test_2d(void)
{
    TEST("2d_grid");
    const int W = 64, H = 32;
    const size_t bytes = (size_t)W * H * sizeof(int);
    int *out = malloc(bytes);

    void *dout = NULL;
    MUDA(mudaMalloc(&dout, bytes));

    mudaFunction_t fn = get_kernel("fill_2d");
    int w = W, h = H;
    void *args[] = { &dout, &w, &h };
    MUDA(mudaLaunchKernel(fn, (W + 15) / 16, (H + 15) / 16, 1, 16, 16, 1, 0, NULL, args, NULL));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(out, dout, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < W * H; i++)
        if (out[i] != i)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(dout));
    free(out);
}

static void test_streams_and_events(void)
{
    TEST("streams_and_events");
    const int N = 1 << 14;
    const size_t bytes = N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    void *da = NULL, *db = NULL, *dc = NULL;
    MUDA(mudaMalloc(&da, bytes));
    MUDA(mudaMalloc(&db, bytes));
    MUDA(mudaMalloc(&dc, bytes));

    mudaStream_t s;
    MUDA(mudaStreamCreate(&s));

    MUDA(mudaMemcpyAsync(da, a, bytes, mudaMemcpyHostToDevice, s));
    MUDA(mudaMemcpyAsync(db, b, bytes, mudaMemcpyHostToDevice, s));

    mudaFunction_t fn = get_kernel("vector_add");
    int n = N;
    void *args[] = { &da, &db, &dc, &n };

    mudaEvent_t t0, t1;
    MUDA(mudaEventCreate(&t0));
    MUDA(mudaEventCreate(&t1));
    MUDA(mudaEventRecord(t0, s));
    MUDA(mudaLaunchKernel(fn, N / 256, 1, 1, 256, 1, 1, 0, s, args, NULL));
    MUDA(mudaEventRecord(t1, s));
    MUDA(mudaStreamSynchronize(s));

    float ms = 0;
    MUDA(mudaEventElapsedTime(&ms, t0, t1));
    OK(ms >= 0.0f);

    MUDA(mudaMemcpyAsync(c, dc, bytes, mudaMemcpyDeviceToHost, s));
    MUDA(mudaStreamSynchronize(s));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - 3.0f) > 1e-4f)
            ok = 0;
    OK(ok);
    printf("  kernel time: %.3f ms\n", ms);

    MUDA(mudaEventDestroy(t0));
    MUDA(mudaEventDestroy(t1));
    MUDA(mudaStreamDestroy(s));
    MUDA(mudaFree(da));
    MUDA(mudaFree(db));
    MUDA(mudaFree(dc));
    free(a);
    free(b);
    free(c);
}

static void test_device_queries(void)
{
    TEST("device_queries");
    int v = 0;
    MUDA(mudaDeviceGetAttribute(&v, mudaDevAttrComputeCapabilityMajor, 0));
    OK(v >= 5);
    MUDA(mudaDeviceGetAttribute(&v, mudaDevAttrComputeCapabilityMinor, 0));
    OK(v >= 0);
    MUDA(mudaDeviceGetAttribute(&v, mudaDevAttrWarpSize, 0));
    OK(v == 32);
    MUDA(mudaDeviceGetAttribute(&v, mudaDevAttrMultiProcessorCount, 0));
    OK(v > 0);
    MUDA(mudaDeviceGetAttribute(&v, mudaDevAttrMaxThreadsPerBlock, 0));
    OK(v >= 256);

    char name[256];
    MUDA(mudaDeviceGetName(name, sizeof(name), 0));
    OK(strlen(name) > 0);

    char pci[64];
    MUDA(mudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
    OK(strlen(pci) > 0);
    printf("  name=%s pci=%s\n", name, pci);

    int dev = -1;
    MUDA(mudaGetDevice(&dev));
    OK(dev == 0);
    MUDA(mudaSetDevice(0));

    int ver = 0;
    MUDA(mudaRuntimeGetVersion(&ver));
    OK(ver == MUDA_VERSION);
    MUDA(mudaDriverGetVersion(&ver));
    OK(ver == MUDA_VERSION);

    size_t fr = 0, tot = 0;
    MUDA(mudaMemGetInfo(&fr, &tot));
    OK(tot > 0 && fr <= tot);
    printf("  free %.0f MiB / total %.0f MiB\n",
           (double)fr / 1048576.0, (double)tot / 1048576.0);
}

static void test_pointer_attributes(void)
{
    TEST("pointer_attributes");
    void *d = NULL;
    MUDA(mudaMalloc(&d, 1024));
    mudaPointerAttributes pa;
    MUDA(mudaPointerGetAttributes(&pa, d));
    OK(pa.type == mudaMemoryTypeDevice);
    OK(pa.devicePointer == d);

    void *h = NULL;
    MUDA(mudaMallocHost(&h, 1024));
    OK(h != NULL);
    MUDA(mudaPointerGetAttributes(&pa, h));
    OK(pa.type == mudaMemoryTypeHost);

    void *dp = NULL;
    MUDA(mudaHostGetDevicePointer(&dp, h, 0));
    OK(dp == h);

    MUDA(mudaFreeHost(h));
    MUDA(mudaFree(d));
}

static void test_memcpy_2d(void)
{
    TEST("memcpy_2d");
    const int W = 32, H = 16;
    const size_t spitch = (size_t)W * sizeof(float);
    const size_t dpitch = (size_t)(W + 8) * sizeof(float);
    float *src = (float *)malloc(spitch * H);
    float *dst = (float *)malloc(dpitch * H);
    for (int i = 0; i < W * H; i++)
        src[i] = (float)i;
    memset(dst, 0, dpitch * H);

    void *ds = NULL, *dd = NULL;
    MUDA(mudaMalloc(&ds, spitch * H));
    MUDA(mudaMalloc(&dd, dpitch * H));

    MUDA(mudaMemcpy2D(ds, spitch, src, spitch, W * sizeof(float), H, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy2D(dd, dpitch, ds, spitch, W * sizeof(float), H, mudaMemcpyDeviceToDevice));
    MUDA(mudaMemcpy2D(dst, dpitch, dd, dpitch, W * sizeof(float), H, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (dst[y * (dpitch / 4) + x] != src[y * (spitch / 4) + x])
                ok = 0;
    OK(ok);

    MUDA(mudaFree(ds));
    MUDA(mudaFree(dd));
    free(src);
    free(dst);
}

static void test_memset_d32(void)
{
    TEST("memset_d32");
    const int N = 256;
    void *d = NULL;
    MUDA(mudaMalloc(&d, N * sizeof(unsigned)));
    MUDA(mudaMemsetD32(d, 0xDEADBEEF, N));
    unsigned out[256];
    MUDA(mudaMemcpy(out, d, sizeof(out), mudaMemcpyDeviceToHost));
    int ok = 1;
    for (int i = 0; i < N; i++)
        if (out[i] != 0xDEADBEEF)
            ok = 0;
    OK(ok);
    MUDA(mudaFree(d));
}

static void test_packed_launch(void)
{
    TEST("packed_launch");
    const int N = 2048;
    const size_t bytes = N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }
    void *da = NULL, *db = NULL, *dc = NULL;
    MUDA(mudaMalloc(&da, bytes));
    MUDA(mudaMalloc(&db, bytes));
    MUDA(mudaMalloc(&dc, bytes));
    MUDA(mudaMemcpy(da, a, bytes, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(db, b, bytes, mudaMemcpyHostToDevice));

    /* Manually pack the arguments the way CUDA's extra[] interface does. */
    unsigned char packed[64];
    size_t off = 0;
    int n = N;
    memcpy(packed + off, &da, sizeof(void *)); off += sizeof(void *);
    memcpy(packed + off, &db, sizeof(void *)); off += sizeof(void *);
    memcpy(packed + off, &dc, sizeof(void *)); off += sizeof(void *);
    memcpy(packed + off, &n, sizeof(int));        off += sizeof(int);
    size_t packed_size = off;

    void *extra[] = {
        MUDA_LAUNCH_PARAM_BUFFER_POINTER, packed,
        MUDA_LAUNCH_PARAM_BUFFER_SIZE,    &packed_size,
        MUDA_LAUNCH_PARAM_END
    };

    mudaFunction_t fn = get_kernel("vector_add");
    MUDA(mudaLaunchKernel(fn, N / 256, 1, 1, 256, 1, 1, 0, NULL, NULL, extra));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(c, dc, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-4f)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(da));
    MUDA(mudaFree(db));
    MUDA(mudaFree(dc));
    free(a);
    free(b);
    free(c);
}

static void test_launch_ex(void)
{
    TEST("launch_ex");
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = 2.0f;
        b[i] = 3.0f;
    }
    void *da = NULL, *db = NULL, *dc = NULL;
    MUDA(mudaMalloc(&da, bytes));
    MUDA(mudaMalloc(&db, bytes));
    MUDA(mudaMalloc(&dc, bytes));
    MUDA(mudaMemcpy(da, a, bytes, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(db, b, bytes, mudaMemcpyHostToDevice));

    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    mudaLaunchConfig cfg = mudaLaunchConfigDefault;
    cfg.gridDimX = N / 128;
    cfg.blockDimX = 128;

    mudaFunction_t fn = get_kernel("vector_add");
    MUDA(mudaLaunchKernelEx(&cfg, fn, args, NULL));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(c, dc, bytes, mudaMemcpyDeviceToHost));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - 5.0f) > 1e-4f)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(da));
    MUDA(mudaFree(db));
    MUDA(mudaFree(dc));
    free(a);
    free(b);
    free(c);
}

static void test_func_attributes(void)
{
    TEST("func_attributes");
    mudaFunction_t fn = get_kernel("vector_add");
    mudaFuncAttributes fa;
    MUDA(mudaFuncGetAttributes(&fa, fn));
    OK(fa.maxThreadsPerBlock > 0);

    int nb = 0;
    MUDA(mudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, 256, 0));
    OK(nb >= 1);
    printf("  maxThreadsPerBlock=%d occupancy@256=%d blocks/SM\n",
           fa.maxThreadsPerBlock, nb);
}

static void test_stream_event_query(void)
{
    TEST("stream_event_query");
    mudaStream_t s;
    MUDA(mudaStreamCreateWithFlags(&s, mudaStreamNonBlocking));
    mudaEvent_t ev;
    MUDA(mudaEventCreateWithFlags(&ev, mudaEventDisableTiming));

    MUDA(mudaEventRecord(ev, s));
    MUDA(mudaEventSynchronize(ev));
    MUDA(mudaEventQuery(ev));

    MUDA(mudaStreamWaitEvent(s, ev, 0));
    MUDA(mudaStreamSynchronize(s));
    MUDA(mudaStreamQuery(s));

    MUDA(mudaEventDestroy(ev));
    MUDA(mudaStreamDestroy(s));
}

int main(void)
{
    printf("muda %d.%d.%d test suite\n\n",
           MUDA_VERSION_MAJOR, MUDA_VERSION_MINOR, MUDA_VERSION_PATCH);

    if (mudaInit(0) != mudaSuccess) {
        printf("mudaInit failed: %s\n", mudaGetErrorString(mudaGetLastError()));
        return 1;
    }

    int count = 0;
    mudaGetDeviceCount(&count);
    if (count == 0) {
        printf("no device available\n");
        return 1;
    }

    printf("driver: %s\n\n", mudaGetDriverName());

    if (mudaModuleLoadData(&g_mod, kKernels) != mudaSuccess) {
        printf("module load failed\n");
        return 1;
    }

    test_device();
    test_device_queries();
    test_memory();
    test_pointer_attributes();
    test_memcpy_2d();
    test_memset_d32();
    test_vector_add();
    test_saxpy();
    test_shared_syncthreads();
    test_2d();
    test_packed_launch();
    test_launch_ex();
    test_func_attributes();
    test_streams_and_events();
    test_stream_event_query();

    mudaModuleUnload(g_mod);
    mudaShutdown();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
