/*
 * vector_add.c - the "hello world" of muda.
 *
 * The kernel is written in CUDA C.  muda translates it to OpenCL C at run
 * time and executes it on the Tesla M40 without ever touching libcuda.
 */
#include "muda.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK(expr)                                                          \
    do {                                                                     \
        mudaError_t _e = (expr);                                            \
        if (_e != mudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__,        \
                    #expr, mudaGetErrorString(_e));                         \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static const char *kVectorAdd =
    "__global__ void vector_add(const float *a, const float *b, float *c, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n)\n"
    "        c[i] = a[i] + b[i];\n"
    "}\n";

int main(void)
{
    const int N = 1 << 20; /* 1M elements */
    const size_t bytes = (size_t)N * sizeof(float);

    CHECK(mudaInit(0));

    mudaDeviceProp prop;
    CHECK(mudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d, %d SMs, %.0f MiB)\n",
           prop.name, prop.computeCapabilityMajor, prop.computeCapabilityMinor,
           prop.multiProcessorCount, (double)prop.totalGlobalMem / 1048576.0);

    /* Allocate and fill host memory. */
    float *ha = (float *)malloc(bytes);
    float *hb = (float *)malloc(bytes);
    float *hc = (float *)malloc(bytes);
    if (!ha || !hb || !hc)
        return 1;
    for (int i = 0; i < N; i++) {
        ha[i] = (float)i;
        hb[i] = (float)(2 * i);
    }

    /* Device memory + upload. */
    void *da = NULL, *db = NULL, *dc = NULL;
    CHECK(mudaMalloc(&da, bytes));
    CHECK(mudaMalloc(&db, bytes));
    CHECK(mudaMalloc(&dc, bytes));
    CHECK(mudaMemcpy(da, ha, bytes, mudaMemcpyHostToDevice));
    CHECK(mudaMemcpy(db, hb, bytes, mudaMemcpyHostToDevice));

    /* Compile and look up the kernel. */
    mudaModule_t mod;
    CHECK(mudaModuleLoadData(&mod, kVectorAdd));
    mudaFunction_t fn;
    CHECK(mudaModuleGetFunction(&fn, mod, "vector_add"));

    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    const int block = 256;
    const int grid = (N + block - 1) / block;

    /* Time the launch with muda events. */
    mudaEvent_t t0, t1;
    CHECK(mudaEventCreate(&t0));
    CHECK(mudaEventCreate(&t1));
    CHECK(mudaEventRecord(t0, NULL));
    CHECK(mudaLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL));
    CHECK(mudaEventRecord(t1, NULL));
    CHECK(mudaDeviceSynchronize());

    float ms = 0;
    CHECK(mudaEventElapsedTime(&ms, t0, t1));

    /* Verify. */
    CHECK(mudaMemcpy(hc, dc, bytes, mudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(hc[i] - (ha[i] + hb[i])) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "mismatch at %d: got %f expected %f\n",
                        i, hc[i], ha[i] + hb[i]);
            errors++;
        }
    }

    printf("vector_add of %d elements: %s (%.3f ms, %.1f GB/s)\n",
           N, errors ? "FAILED" : "OK", ms,
           errors ? 0.0 : (3.0 * bytes / (ms / 1000.0)) / 1e9);

    CHECK(mudaEventDestroy(t0));
    CHECK(mudaEventDestroy(t1));
    CHECK(mudaModuleUnload(mod));
    CHECK(mudaFree(da));
    CHECK(mudaFree(db));
    CHECK(mudaFree(dc));
    free(ha);
    free(hb);
    free(hc);
    mudaShutdown();
    return errors ? 1 : 0;
}
