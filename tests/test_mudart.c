/*
 * test_mudart.c - tests for the CUDA runtime shim.
 */
#include "mudart.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;
static const char *g_test = "";

#define OK(cond)                                                             \
    do {                                                                     \
        if (cond) {                                                          \
            g_pass++;                                                        \
        } else {                                                             \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s\n", g_test, __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

#define CUDA(expr)                                                           \
    do {                                                                     \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s -> %s\n", g_test, __FILE__,      \
                   __LINE__, #expr, cudaGetErrorString(_e));                \
        }                                                                    \
    } while (0)

#define TEST(name) do { g_test = name; printf("== %s ==\n", name); } while (0)

static const char *kVectorAdd =
    "__global__ void vector_add(const float *a, const float *b, float *c, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) c[i] = a[i] + b[i];\n"
    "}\n";

static void test_device(void)
{
    TEST("device");
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    OK(count >= 1);

    cudaDeviceProp p;
    CUDA(cudaGetDeviceProperties(&p, 0));
    OK(strlen(p.name) > 0);
    OK(p.major == 5 || p.major > 0);
    OK(p.maxThreadsPerBlock >= 64);
    OK(p.totalGlobalMem > 0);
    printf("  %s sm_%d%d, %d SMs, %.0f MiB\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           (double)p.totalGlobalMem / 1048576.0);
}

static void test_memory(void)
{
    TEST("memory");
    const size_t N = 1024;
    unsigned *h = (unsigned *)malloc(N * 4), *back = (unsigned *)malloc(N * 4);
    for (size_t i = 0; i < N; i++)
        h[i] = (unsigned)(i * 2654435761u);

    void *d = NULL;
    CUDA(cudaMalloc(&d, N * 4));
    CUDA(cudaMemcpy(d, h, N * 4, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(d, 0, N * 4));
    CUDA(cudaMemcpy(back, d, N * 4, cudaMemcpyDeviceToHost));
    int ok = 1;
    for (size_t i = 0; i < N; i++)
        if (back[i] != 0)
            ok = 0;
    OK(ok);

    size_t fr = 0, tot = 0;
    CUDA(cudaMemGetInfo(&fr, &tot));
    OK(tot > 0);

    CUDA(cudaFree(d));
    free(h); free(back);
}

static void test_launch(void)
{
    TEST("launch");
    const int N = 1 << 14;
    const size_t bytes = (size_t)N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = 2.0f; }

    void *da = NULL, *db = NULL, *dc = NULL;
    CUDA(cudaMalloc(&da, bytes));
    CUDA(cudaMalloc(&db, bytes));
    CUDA(cudaMalloc(&dc, bytes));
    CUDA(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    mudaModule_t mod;
    mudaFunction_t fn;
    CUDA(mudaModuleLoadData(&mod, kVectorAdd));
    CUDA(mudaModuleGetFunction(&fn, mod, "vector_add"));

    const int block = 256, grid = N / block;
    int n = N;

    /* cudaLaunchKernel */
    void *args[] = { &da, &db, &dc, &n };
    CUDA(cudaLaunchKernel((const void *)fn, make_dim3(grid, 1, 1),
                          make_dim3(block, 1, 1), args, 0, NULL));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-4f)
            ok = 0;
    OK(ok);

    /* legacy configure/setup/launch */
    CUDA(cudaMemset(dc, 0, bytes));
    CUDA(cudaConfigureCall(make_dim3(grid, 1, 1), make_dim3(block, 1, 1), 0, NULL));
    CUDA(cudaSetupArgument(&da, sizeof(da), 0));
    CUDA(cudaSetupArgument(&db, sizeof(db), sizeof(da)));
    CUDA(cudaSetupArgument(&dc, sizeof(dc), 2 * sizeof(da)));
    CUDA(cudaSetupArgument(&n, sizeof(n), 3 * sizeof(da)));
    CUDA(cudaLaunch((const void *)fn));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-4f)
            ok = 0;
    OK(ok);

    CUDA(mudaModuleUnload(mod));
    CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
    free(a); free(b); free(c);
}

static void test_streams_events(void)
{
    TEST("streams_events");
    const int N = 512;
    const size_t bytes = (size_t)N * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 3.0f; }

    void *da = NULL, *db = NULL, *dc = NULL;
    CUDA(cudaMalloc(&da, bytes));
    CUDA(cudaMalloc(&db, bytes));
    CUDA(cudaMalloc(&dc, bytes));

    cudaStream_t s;
    CUDA(cudaStreamCreate(&s));
    CUDA(cudaMemcpyAsync(da, a, bytes, cudaMemcpyHostToDevice, s));
    CUDA(cudaMemcpyAsync(db, b, bytes, cudaMemcpyHostToDevice, s));

    mudaModule_t mod;
    mudaFunction_t fn;
    CUDA(mudaModuleLoadData(&mod, kVectorAdd));
    CUDA(mudaModuleGetFunction(&fn, mod, "vector_add"));

    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    cudaEvent_t t0, t1;
    CUDA(cudaEventCreate(&t0));
    CUDA(cudaEventCreate(&t1));
    CUDA(cudaEventRecord(t0, s));
    CUDA(cudaLaunchKernel((const void *)fn, make_dim3(2, 1, 1),
                          make_dim3(256, 1, 1), args, 0, s));
    CUDA(cudaEventRecord(t1, s));
    CUDA(cudaStreamSynchronize(s));

    float ms = 0;
    CUDA(cudaEventElapsedTime(&ms, t0, t1));
    OK(ms >= 0.0f);

    CUDA(cudaMemcpyAsync(c, dc, bytes, cudaMemcpyDeviceToHost, s));
    CUDA(cudaStreamSynchronize(s));
    int ok = 1;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - 4.0f) > 1e-4f)
            ok = 0;
    OK(ok);

    CUDA(cudaEventDestroy(t0));
    CUDA(cudaEventDestroy(t1));
    CUDA(cudaStreamDestroy(s));
    CUDA(mudaModuleUnload(mod));
    CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
    free(a); free(b); free(c);
}

int main(void)
{
    printf("mudart (CUDA runtime shim) test suite\n\n");
    if (mudaInit(0) != mudaSuccess) {
        printf("mudaInit failed\n");
        return 1;
    }

    test_device();
    test_memory();
    test_launch();
    test_streams_events();

    mudaShutdown();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
