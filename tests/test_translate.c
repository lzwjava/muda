/*
 * test_translate.c - tests for the CUDA source compatibility layer:
 * math intrinsics, atomics, qualifiers and __launch_bounds__.
 */
#include "muda.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

#define OK(cond)                                                             \
    do {                                                                     \
        if (cond) g_pass++;                                                  \
        else { g_fail++; printf("  [FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define MUDA(expr)                                                           \
    do {                                                                     \
        mudaError_t _e = (expr);                                            \
        if (_e != mudaSuccess) {                                            \
            g_fail++;                                                        \
            printf("  [FAIL] %s:%d: %s -> %s\n", __FILE__, __LINE__,        \
                   #expr, mudaGetErrorString(_e));                          \
        }                                                                    \
    } while (0)

static const char *kCompat =
    "__device__ __forceinline__ float twice(float x) { return 2.0f * x; }\n"
    "__launch_bounds__(256, 2)\n"
    "__global__ void compat(float *out, int *icount, float *fsum,\n"
    "                       const float *in, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    float x = in[i];\n"
    "    out[i] = fmaf(x, 2.0f, __expf(x)) + __fdividef(x, 2.0f);\n"
    "    atomicAdd(&icount[0], 1);\n"
    "    atomicAdd(&fsum[0], x);\n"
    "    atomicAdd(&fsum[1], __saturatef(x));\n"
    "    atomicAdd(&fsum[2], twice(x));\n"
    "}\n";

int main(void)
{
    printf("CUDA source compatibility test\n\n");

    if (mudaInit(0) != mudaSuccess) {
        printf("mudaInit failed: %s\n", mudaGetErrorString(mudaGetLastError()));
        return 1;
    }

    const int N = 256;
    float *in = (float *)malloc(N * 4), *out = (float *)malloc(N * 4);
    float *fsum = (float *)malloc(3 * 4);
    for (int i = 0; i < N; i++)
        in[i] = (float)i / (float)(N - 1);

    void *din = NULL, *dout = NULL, *dicount = NULL, *dfsum = NULL;
    MUDA(mudaMalloc(&din, N * 4));
    MUDA(mudaMalloc(&dout, N * 4));
    MUDA(mudaMalloc(&dicount, 4));
    MUDA(mudaMalloc(&dfsum, 3 * 4));
    MUDA(mudaMemset(dicount, 0, 4));
    MUDA(mudaMemset(dfsum, 0, 3 * 4));
    MUDA(mudaMemcpy(din, in, N * 4, mudaMemcpyHostToDevice));

    mudaModule_t mod;
    mudaFunction_t fn;
    MUDA(mudaModuleLoadData(&mod, kCompat));
    MUDA(mudaModuleGetFunction(&fn, mod, "compat"));

    int n = N;
    void *args[] = { &dout, &dicount, &dfsum, &din, &n };
    MUDA(mudaLaunchKernel(fn, 1, 1, 1, N, 1, 1, 0, NULL, args, NULL));
    MUDA(mudaDeviceSynchronize());

    int count = 0;
    MUDA(mudaMemcpy(&count, dicount, 4, mudaMemcpyDeviceToHost));
    MUDA(mudaMemcpy(fsum, dfsum, 3 * 4, mudaMemcpyDeviceToHost));
    MUDA(mudaMemcpy(out, dout, N * 4, mudaMemcpyDeviceToHost));

    OK(count == N);

    float sum_x = 0.0f;
    for (int i = 0; i < N; i++)
        sum_x += in[i];
    OK(fabsf(fsum[0] - sum_x) < 1e-3f);          /* atomicAdd(float)   */
    OK(fabsf(fsum[1] - sum_x) < 1e-3f);          /* __saturatef        */
    OK(fabsf(fsum[2] - 2.0f * sum_x) < 1e-3f);   /* __forceinline__    */

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float x = in[i];
        float expected = 2.0f * x + expf(x) + x / 2.0f; /* fmaf + __expf + __fdividef */
        if (fabsf(out[i] - expected) > 1e-3f)
            ok = 0;
    }
    OK(ok);

    printf("  count=%d, sum_x=%.4f, sat=%.4f, twice=%.4f\n",
           count, fsum[0], fsum[1], fsum[2]);

    MUDA(mudaModuleUnload(mod));
    MUDA(mudaFree(din));
    MUDA(mudaFree(dout));
    MUDA(mudaFree(dicount));
    MUDA(mudaFree(dfsum));
    free(in); free(out); free(fsum);
    mudaShutdown();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
