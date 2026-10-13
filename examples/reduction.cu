/*
 * reduction.cu - a normal CUDA block reduction, built through mudac.
 *
 * Demonstrates __shared__ memory, __syncthreads() and atomicAdd(float) on a
 * CUDA-style program that never mentions muda.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void reduce_sum(const float *in, float *out, int n)
{
    __shared__ float s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }

    if (t == 0)
        atomicAdd(out, s[0]);
}

int main(void)
{
    const int N = 1 << 20;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    float *in = (float *)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++)
        in[i] = 1.0f / (float)(i + 1);

    float *din = NULL, *dout = NULL;
    cudaMalloc((void **)&din, (size_t)N * sizeof(float));
    cudaMalloc((void **)&dout, sizeof(float));
    cudaMemcpy(din, in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, sizeof(float));

    reduce_sum<<<grid, block>>>(din, dout, N);
    cudaDeviceSynchronize();

    float sum = 0.0f;
    cudaMemcpy(&sum, dout, sizeof(float), cudaMemcpyDeviceToHost);

    double ref = 0.0;
    for (int i = 0; i < N; i++)
        ref += in[i];

    int ok = fabs((double)sum - ref) < 1e-2;
    printf("reduction: %s (got %.4f, ref %.4f)\n", ok ? "OK" : "FAILED", sum, ref);

    cudaFree(din);
    cudaFree(dout);
    free(in);
    return ok ? 0 : 1;
}
