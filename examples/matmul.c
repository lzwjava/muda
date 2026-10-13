/*
 * matmul.c - tiled single-precision matrix multiply.
 *
 * Demonstrates 2D grids/blocks, __shared__ memory and __syncthreads(),
 * all of which the muda translator maps onto OpenCL local memory and
 * barriers.
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

#define TILE 16

static const char *kMatmul =
    "#define TILE 16\n"
    "__global__ void matmul_tiled(const float *A, const float *B, float *C, int N)\n"
    "{\n"
    "    __shared__ float As[TILE][TILE];\n"
    "    __shared__ float Bs[TILE][TILE];\n"
    "    int tx = threadIdx.x;\n"
    "    int ty = threadIdx.y;\n"
    "    int row = blockIdx.y * TILE + ty;\n"
    "    int col = blockIdx.x * TILE + tx;\n"
    "    float sum = 0.0f;\n"
    "    for (int t = 0; t < N / TILE; t++) {\n"
    "        As[ty][tx] = A[row * N + t * TILE + tx];\n"
    "        Bs[ty][tx] = B[(t * TILE + ty) * N + col];\n"
    "        __syncthreads();\n"
    "        for (int k = 0; k < TILE; k++)\n"
    "            sum += As[ty][k] * Bs[k][tx];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    C[row * N + col] = sum;\n"
    "}\n";

static void cpu_matmul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main(void)
{
    const int N = 256;
    const size_t bytes = (size_t)N * N * sizeof(float);

    CHECK(mudaInit(0));

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);
    float *ref = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)((i % 17) - 8) * 0.1f;
        B[i] = (float)((i % 13) - 6) * 0.1f;
    }
    cpu_matmul(A, B, ref, N);

    void *dA = NULL, *dB = NULL, *dC = NULL;
    CHECK(mudaMalloc(&dA, bytes));
    CHECK(mudaMalloc(&dB, bytes));
    CHECK(mudaMalloc(&dC, bytes));
    CHECK(mudaMemcpy(dA, A, bytes, mudaMemcpyHostToDevice));
    CHECK(mudaMemcpy(dB, B, bytes, mudaMemcpyHostToDevice));

    mudaModule_t mod;
    mudaFunction_t fn;
    CHECK(mudaModuleLoadData(&mod, kMatmul));
    CHECK(mudaModuleGetFunction(&fn, mod, "matmul_tiled"));

    int n = N;
    void *args[] = { &dA, &dB, &dC, &n };
    const int grid = N / TILE;

    mudaEvent_t t0, t1;
    CHECK(mudaEventCreate(&t0));
    CHECK(mudaEventCreate(&t1));
    CHECK(mudaEventRecord(t0, NULL));
    CHECK(mudaLaunchKernel(fn, grid, grid, 1, TILE, TILE, 1, 0, NULL, args, NULL));
    CHECK(mudaEventRecord(t1, NULL));
    CHECK(mudaDeviceSynchronize());

    float ms = 0;
    CHECK(mudaEventElapsedTime(&ms, t0, t1));
    CHECK(mudaMemcpy(C, dC, bytes, mudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N * N; i++)
        if (fabsf(C[i] - ref[i]) > 1e-2f)
            errors++;

    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
    printf("matmul %dx%d: %s (%.3f ms, %.1f GFLOP/s)\n",
           N, N, errors ? "FAILED" : "OK", ms, errors ? 0.0 : gflops);

    CHECK(mudaEventDestroy(t0));
    CHECK(mudaEventDestroy(t1));
    CHECK(mudaModuleUnload(mod));
    CHECK(mudaFree(dA));
    CHECK(mudaFree(dB));
    CHECK(mudaFree(dC));
    free(A);
    free(B);
    free(C);
    free(ref);
    mudaShutdown();
    return errors ? 1 : 0;
}
