/*
 * test_memory.c - driver-independent memory / stream / event tests.
 *
 * No kernels are launched, so this runs under both the OpenCL driver and the
 * CPU reference driver (MUDA_DRIVER=cpu).
 */
#include "muda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void)
{
    printf("muda memory/stream/event test\n");
    if (mudaInit(0) != mudaSuccess) {
        printf("mudaInit failed: %s\n", mudaGetErrorString(mudaGetLastError()));
        return 1;
    }
    printf("driver: %s\n", mudaGetDriverName());

    int count = 0;
    MUDA(mudaGetDeviceCount(&count));
    OK(count >= 1);

    mudaDeviceProp p;
    MUDA(mudaGetDeviceProperties(&p, 0));
    OK(p.totalGlobalMem > 0);
    printf("  device: %s (%zu MiB)\n", p.name, p.totalGlobalMem >> 20);

    const size_t N = 4096;
    unsigned char *h = (unsigned char *)malloc(N);
    unsigned char *back = (unsigned char *)malloc(N);
    for (size_t i = 0; i < N; i++)
        h[i] = (unsigned char)(i * 7 + 1);

    void *d = NULL;
    MUDA(mudaMalloc(&d, N));
    OK(d != NULL);

    MUDA(mudaMemcpy(d, h, N, mudaMemcpyHostToDevice));
    MUDA(mudaMemset(d, 0x5A, N));
    MUDA(mudaMemcpy(back, d, N, mudaMemcpyDeviceToHost));
    int ok = 1;
    for (size_t i = 0; i < N; i++)
        if (back[i] != 0x5A)
            ok = 0;
    OK(ok);

    MUDA(mudaMemcpy(d, h, N, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(back, d, N, mudaMemcpyDeviceToHost));
    OK(memcmp(back, h, N) == 0);

    mudaPointerAttributes pa;
    MUDA(mudaPointerGetAttributes(&pa, d));
    OK(pa.type == mudaMemoryTypeDevice);

    size_t freeBytes = 0, totalBytes = 0;
    MUDA(mudaMemGetInfo(&freeBytes, &totalBytes));
    OK(totalBytes > 0);

    mudaStream_t s;
    MUDA(mudaStreamCreate(&s));
    MUDA(mudaStreamSynchronize(s));
    MUDA(mudaStreamDestroy(s));

    mudaEvent_t e0, e1;
    MUDA(mudaEventCreate(&e0));
    MUDA(mudaEventCreate(&e1));
    MUDA(mudaEventRecord(e0, NULL));
    MUDA(mudaEventRecord(e1, NULL));
    MUDA(mudaEventSynchronize(e1));
    float ms = 0.0f;
    MUDA(mudaEventElapsedTime(&ms, e0, e1));
    OK(ms >= 0.0f);
    MUDA(mudaEventDestroy(e0));
    MUDA(mudaEventDestroy(e1));

    MUDA(mudaFree(d));
    free(h);
    free(back);
    mudaShutdown();

    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
