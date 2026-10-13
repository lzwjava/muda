/*
 * mudablas.h - a small, row-major linear algebra library for muda.
 *
 * All functions operate on device memory allocated with mudaMalloc().  The
 * storage convention is ROW-MAJOR (C style), which differs from cuBLAS, but
 * matches how most CUDA kernels and Python/numpy arrays are laid out.
 *
 *   C = alpha * op(A) * op(B) + beta * C
 *
 * where op(X) is X (trans=0) or X^T (trans=1).
 *
 * The GEMM entry point uses a shared-memory, register-blocked kernel
 * (64x64 block, 4x4 per thread) and is the performance centrepiece of muda.
 */
#ifndef MUDABLAS_H
#define MUDABLAS_H

#include "muda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum mudaOp {
    mudaOpN = 0,   /* no transpose */
    mudaOpT = 1    /* transpose    */
} mudaOp;

/* SGEMM: C(MxN) = alpha * op(A) * op(B) + beta * C, row-major.
 *
 *   transa=0: A is MxK, lda = K
 *   transa=1: A is KxM, lda = M
 *   transb=0: B is KxN, ldb = N
 *   transb=1: B is NxK, ldb = K
 *   C is MxN, ldc = N
 */
mudaError_t mudaSgemm(int transa, int transb, int m, int n, int k,
                      float alpha, const float *A, int lda,
                      const float *B, int ldb,
                      float beta, float *C, int ldc);

/* Same as mudaSgemm but with a simple one-thread-per-element kernel.
 * Useful as a correctness/performance reference. */
mudaError_t mudaSgemmNaive(int transa, int transb, int m, int n, int k,
                           float alpha, const float *A, int lda,
                           const float *B, int ldb,
                           float beta, float *C, int ldc);

/* SGEMV: y = alpha * op(A) * x + beta * y, row-major.  A is always MxN, lda=N.
 *   trans=0: x has N elements, y has M elements
 *   trans=1: x has M elements, y has N elements
 */
mudaError_t mudaSgemv(int trans, int m, int n,
                      float alpha, const float *A, int lda,
                      const float *x, int incx,
                      float beta, float *y, int incy);

/* y = alpha * x + y, with strides.  Uses float4 when possible. */
mudaError_t mudaSaxpy(int n, float alpha, const float *x, int incx,
                      float *y, int incy);

/* x = alpha * x, with stride. */
mudaError_t mudaSscal(int n, float alpha, float *x, int incx);

/* In-place ReLU. */
mudaError_t mudaSrelu(int n, float *x);

/* In-place ReLU backward: dx[i] = y[i] > 0 ? dx[i] : 0, where y is the
 * forward activation that went through ReLU. */
mudaError_t mudaSreluBackward(int n, const float *y, float *dx);

/* y = sigmoid(x), elementwise. */
mudaError_t mudaSsigmoid(int n, const float *x, float *y);

/* out = a - b, elementwise. */
mudaError_t mudaSsub(int n, const float *a, const float *b, float *out);

/* y[r*cols + c] += bias[c], for r in [0, rows). */
mudaError_t mudaSaddBias(int rows, int cols, float *y, const float *bias);

/* out[c] = sum_r x[r*cols + c] (column sums, used for bias gradients). */
mudaError_t mudaScolSum(int rows, int cols, const float *x, float *out);

#ifdef __cplusplus
}
#endif

#endif /* MUDABLAS_H */
