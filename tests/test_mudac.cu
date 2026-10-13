/*
 * test_mudac.cu - exercises the mudac launch rewriting.
 *
 * Covers:
 *   - a 1D launch with integer grid/block
 *   - a 2D launch with dim3 grid/block
 *   - multiple kernels in one translation unit
 *   - scalar and pointer kernel arguments
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

__global__ void fill_2d(int *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
        out[y * w + x] = y * w + x;
}

__global__ void saxpy(float a, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void)
{
    int fails = 0;

    /* 1D launch with integer dimensions */
    {
        const int N = 1 << 16;
        const size_t bytes = (size_t)N * sizeof(float);
        float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes), *c = (float *)malloc(bytes);
        for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = 1.0f; }

        float *da = NULL, *db = NULL, *dc = NULL;
        cudaMalloc((void **)&da, bytes);
        cudaMalloc((void **)&db, bytes);
        cudaMalloc((void **)&dc, bytes);
        cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

        const int block = 256;
        const int grid = N / block;
        vector_add<<<grid, block>>>(da, db, dc, N);

        /* scalar + pointer arguments */
        saxpy<<<grid, block>>>(2.0f, da, dc, N);

        cudaDeviceSynchronize();
        cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int i = 0; i < N; i++)
            if (fabsf(c[i] - (2.0f * a[i] + (a[i] + b[i]))) > 1e-3f)
                ok = 0;
        printf("mudac 1D (ints, scalar arg): %s\n", ok ? "OK" : "FAILED");
        fails += !ok;

        cudaFree(da); cudaFree(db); cudaFree(dc);
        free(a); free(b); free(c);
    }

    /* 2D launch with dim3 */
    {
        const int W = 64, H = 32;
        const size_t bytes = (size_t)W * H * sizeof(int);
        int *back = (int *)malloc(bytes);

        int *dout = NULL;
        cudaMalloc((void **)&dout, bytes);

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        fill_2d<<<grid, block>>>(dout, W, H);

        cudaDeviceSynchronize();
        cudaMemcpy(back, dout, bytes, cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int i = 0; i < W * H; i++)
            if (back[i] != i)
                ok = 0;
        printf("mudac 2D (dim3): %s\n", ok ? "OK" : "FAILED");
        fails += !ok;

        cudaFree(dout);
        free(back);
    }

    return fails ? 1 : 0;
}
