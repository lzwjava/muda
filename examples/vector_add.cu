/*
 * vector_add.cu - a normal CUDA program, run through the mudac preprocessor.
 *
 * Build:
 *   make build/vector_add_mudac
 *   ./build/vector_add_mudac
 *
 * mudac turns this into C that links against libmuda; the kernel is compiled
 * by the muda driver at run time.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main(void)
{
    const int N = 1 << 20;
    const size_t bytes = (size_t)N * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(2 * i);
    }

    float *da = NULL, *db = NULL, *dc = NULL;
    cudaMalloc((void **)&da, bytes);
    cudaMalloc((void **)&db, bytes);
    cudaMalloc((void **)&dc, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = (N + block - 1) / block;
    vector_add<<<grid, block>>>(da, db, dc, N);

    cudaDeviceSynchronize();
    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-3f)
            errors++;

    printf("mudac vector_add: %s\n", errors ? "FAILED" : "OK");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
    return errors ? 1 : 0;
}
