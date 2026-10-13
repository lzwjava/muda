/*
 * saxpy.c - single-precision A*X+Y using muda.
 *
 * Demonstrates scalar kernel arguments, streams and event timing.
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

static const char *kSaxpy =
    "__global__ void saxpy(float a, const float *x, float *y, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) y[i] = a * x[i] + y[i];\n"
    "}\n";

int main(void)
{
    const int N = 1 << 22;
    const size_t bytes = (size_t)N * sizeof(float);
    const float a = 2.0f;

    CHECK(mudaInit(0));

    float *x = (float *)malloc(bytes);
    float *y = (float *)malloc(bytes);
    float *back = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = (float)i;
    }

    void *dx = NULL, *dy = NULL;
    CHECK(mudaMalloc(&dx, bytes));
    CHECK(mudaMalloc(&dy, bytes));

    mudaStream_t stream;
    CHECK(mudaStreamCreate(&stream));
    CHECK(mudaMemcpyAsync(dx, x, bytes, mudaMemcpyHostToDevice, stream));
    CHECK(mudaMemcpyAsync(dy, y, bytes, mudaMemcpyHostToDevice, stream));

    mudaModule_t mod;
    mudaFunction_t fn;
    CHECK(mudaModuleLoadData(&mod, kSaxpy));
    CHECK(mudaModuleGetFunction(&fn, mod, "saxpy"));

    int n = N;
    void *args[] = { (void *)&a, &dx, &dy, &n };
    const int block = 256;
    const int grid = (N + block - 1) / block;

    mudaEvent_t t0, t1;
    CHECK(mudaEventCreate(&t0));
    CHECK(mudaEventCreate(&t1));
    CHECK(mudaEventRecord(t0, stream));
    CHECK(mudaLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, args, NULL));
    CHECK(mudaEventRecord(t1, stream));
    CHECK(mudaStreamSynchronize(stream));

    CHECK(mudaMemcpyAsync(back, dy, bytes, mudaMemcpyDeviceToHost, stream));
    CHECK(mudaStreamSynchronize(stream));

    float ms = 0;
    CHECK(mudaEventElapsedTime(&ms, t0, t1));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(back[i] - (a * x[i] + y[i])) > 1e-3f)
            errors++;

    printf("saxpy of %d elements: %s (%.3f ms, %.1f GB/s)\n",
           N, errors ? "FAILED" : "OK", ms,
           errors ? 0.0 : (3.0 * bytes / (ms / 1000.0)) / 1e9);

    CHECK(mudaEventDestroy(t0));
    CHECK(mudaEventDestroy(t1));
    CHECK(mudaStreamDestroy(stream));
    CHECK(mudaModuleUnload(mod));
    CHECK(mudaFree(dx));
    CHECK(mudaFree(dy));
    free(x);
    free(y);
    free(back);
    mudaShutdown();
    return errors ? 1 : 0;
}
