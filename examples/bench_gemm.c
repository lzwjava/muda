/*
 * bench_gemm.c - mudaBLAS SGEMM benchmark.
 *
 * Compares the simple one-thread-per-element kernel with the shared-memory,
 * register-blocked one, and reports GFLOP/s.  Results are checked against a
 * CPU reference for the smaller sizes.
 *
 * Usage: bench_gemm [max_size]
 */
#include "muda.h"
#include "mudablas.h"

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

static float frand(unsigned *seed)
{
    *seed = (*seed * 1664525u + 1013904223u);
    return (float)((*seed >> 8) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
}

static void fill(float *p, size_t n, unsigned *seed)
{
    for (size_t i = 0; i < n; i++)
        p[i] = frand(seed);
}

static void cpu_sgemm_nn(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

/* Returns milliseconds for `reps` launches. */
static double time_sgemm(int size, int naive, int reps)
{
    const int M = size, N = size, K = size;
    const size_t asz = (size_t)M * K, bsz = (size_t)K * N, csz = (size_t)M * N;

    unsigned seed = 12345u;
    float *A = malloc(asz * sizeof(float));
    float *B = malloc(bsz * sizeof(float));
    float *C = malloc(csz * sizeof(float));
    fill(A, asz, &seed);
    fill(B, bsz, &seed);

    void *dA = NULL, *dB = NULL, *dC = NULL;
    CHECK(mudaMalloc(&dA, asz * sizeof(float)));
    CHECK(mudaMalloc(&dB, bsz * sizeof(float)));
    CHECK(mudaMalloc(&dC, csz * sizeof(float)));
    CHECK(mudaMemcpy(dA, A, asz * sizeof(float), mudaMemcpyHostToDevice));
    CHECK(mudaMemcpy(dB, B, bsz * sizeof(float), mudaMemcpyHostToDevice));

    /* warmup */
    if (naive)
        CHECK(mudaSgemmNaive(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC, N));
    else
        CHECK(mudaSgemm(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC, N));
    CHECK(mudaDeviceSynchronize());

    mudaEvent_t t0, t1;
    CHECK(mudaEventCreate(&t0));
    CHECK(mudaEventCreate(&t1));
    CHECK(mudaEventRecord(t0, NULL));
    for (int r = 0; r < reps; r++) {
        if (naive)
            CHECK(mudaSgemmNaive(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC, N));
        else
            CHECK(mudaSgemm(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC, N));
    }
    CHECK(mudaEventRecord(t1, NULL));
    CHECK(mudaDeviceSynchronize());
    float ms = 0;
    CHECK(mudaEventElapsedTime(&ms, t0, t1));

    /* verify against CPU reference for small sizes */
    if (size <= 256) {
        CHECK(mudaMemcpy(C, dC, csz * sizeof(float), mudaMemcpyDeviceToHost));
        float *ref = malloc(csz * sizeof(float));
        cpu_sgemm_nn(A, B, ref, M, N, K);
        float maxdiff = 0.0f;
        for (size_t i = 0; i < csz; i++) {
            float d = fabsf(C[i] - ref[i]);
            if (d > maxdiff)
                maxdiff = d;
        }
        if (maxdiff > 1e-3f)
            printf("  [warn] size %d max diff %.5f\n", size, maxdiff);
        free(ref);
    }

    CHECK(mudaEventDestroy(t0));
    CHECK(mudaEventDestroy(t1));
    CHECK(mudaFree(dA));
    CHECK(mudaFree(dB));
    CHECK(mudaFree(dC));
    free(A);
    free(B);
    free(C);
    return (double)ms / reps;
}

int main(int argc, char **argv)
{
    int max_size = argc > 1 ? atoi(argv[1]) : 1024;

    CHECK(mudaInit(0));

    mudaDeviceProp p;
    CHECK(mudaGetDeviceProperties(&p, 0));
    printf("mudaBLAS SGEMM benchmark on %s (sm_%d%d, %d SMs)\n\n",
           p.name, p.computeCapabilityMajor, p.computeCapabilityMinor,
           p.multiProcessorCount);

    printf("%8s %14s %14s %10s\n", "size", "naive GFLOP/s", "tuned GFLOP/s", "speedup");
    printf("--------------------------------------------------------------\n");

    for (int size = 128; size <= max_size; size *= 2) {
        int reps = size <= 256 ? 50 : (size <= 512 ? 20 : 5);
        double mn = time_sgemm(size, 1, reps);
        double mt = time_sgemm(size, 0, reps);
        double fn = 2.0 * size * size * size / (mn * 1e6);
        double ft = 2.0 * size * size * size / (mt * 1e6);
        printf("%8d %14.1f %14.1f %9.2fx\n", size, fn, ft, fn > 0 ? ft / fn : 0.0);
    }

    mudaShutdown();
    return 0;
}
