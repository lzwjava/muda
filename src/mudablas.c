/*
 * mudablas.c - linear algebra kernels for muda.
 *
 * The kernels are written in CUDA C and compiled once, lazily, by the
 * active muda driver.  The GEMM kernel is shared-memory tiled and register
 * blocked (64x64 output tile per block, 4x4 per thread).
 */

#include "mudablas.h"

#include <stdio.h>
#include <string.h>
#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Kernel source                                                      */
/* ------------------------------------------------------------------ */
static const char *kBlasSource =
"#define BM 64\n"
"#define BN 64\n"
"#define BK 16\n"
"#define TM 4\n"
"#define TN 4\n"
"\n"
"__global__ void sgemm_naive(int M, int N, int K, float alpha,\n"
"                            const float *A, int lda, int ta,\n"
"                            const float *B, int ldb, int tb,\n"
"                            float beta, float *C, int ldc)\n"
"{\n"
"    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (row >= M || col >= N) return;\n"
"    float sum = 0.0f;\n"
"    for (int t = 0; t < K; t++) {\n"
"        float av = ta ? A[t * lda + row] : A[row * lda + t];\n"
"        float bv = tb ? B[col * ldb + t] : B[t * ldb + col];\n"
"        sum += av * bv;\n"
"    }\n"
"    float v = alpha * sum;\n"
"    if (beta != 0.0f) v += beta * C[row * ldc + col];\n"
"    C[row * ldc + col] = v;\n"
"}\n"
"\n"
"__global__ void sgemm_rb(int M, int N, int K, float alpha,\n"
"                         const float *A, int lda, int ta,\n"
"                         const float *B, int ldb, int tb,\n"
"                         float beta, float *C, int ldc)\n"
"{\n"
"    __shared__ float As[BM][BK];\n"
"    __shared__ float Bs[BK][BN];\n"
"    int tx = threadIdx.x;\n"
"    int ty = threadIdx.y;\n"
"    int tid = ty * 16 + tx;\n"
"    int row0 = blockIdx.y * BM + ty * TM;\n"
"    int col0 = blockIdx.x * BN + tx * TN;\n"
"    float acc[TM][TN];\n"
"    for (int i = 0; i < TM; i++)\n"
"        for (int j = 0; j < TN; j++)\n"
"            acc[i][j] = 0.0f;\n"
"    for (int t = 0; t < K; t += BK) {\n"
"        for (int i = 0; i < (BM * BK) / 256; i++) {\n"
"            int idx = i * 256 + tid;\n"
"            int r = idx / BK;\n"
"            int c = idx % BK;\n"
"            int gr = blockIdx.y * BM + r;\n"
"            int gc = t + c;\n"
"            float v = 0.0f;\n"
"            if (gr < M && gc < K) v = ta ? A[gc * lda + gr] : A[gr * lda + gc];\n"
"            As[r][c] = v;\n"
"        }\n"
"        for (int i = 0; i < (BK * BN) / 256; i++) {\n"
"            int idx = i * 256 + tid;\n"
"            int r = idx / BN;\n"
"            int c = idx % BN;\n"
"            int gr = t + r;\n"
"            int gc = blockIdx.x * BN + c;\n"
"            float v = 0.0f;\n"
"            if (gr < K && gc < N) v = tb ? B[gc * ldb + gr] : B[gr * ldb + gc];\n"
"            Bs[r][c] = v;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < BK; kk++) {\n"
"            float aReg[TM];\n"
"            float bReg[TN];\n"
"            for (int i = 0; i < TM; i++) aReg[i] = As[ty * TM + i][kk];\n"
"            for (int j = 0; j < TN; j++) bReg[j] = Bs[kk][tx * TN + j];\n"
"            for (int i = 0; i < TM; i++)\n"
"                for (int j = 0; j < TN; j++)\n"
"                    acc[i][j] += aReg[i] * bReg[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i = 0; i < TM; i++) {\n"
"        for (int j = 0; j < TN; j++) {\n"
"            int r = row0 + i;\n"
"            int c = col0 + j;\n"
"            if (r < M && c < N) {\n"
"                float v = alpha * acc[i][j];\n"
"                if (beta != 0.0f) v += beta * C[r * ldc + c];\n"
"                C[r * ldc + c] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"#define R2_BM 128\n"
"#define R2_BN 64\n"
"#define R2_BK 16\n"
"#define R2_TM 8\n"
"#define R2_TN 4\n"
"__global__ void sgemm_rb128(int M, int N, int K, float alpha,\n"
"                            const float *A, int lda, int ta,\n"
"                            const float *B, int ldb, int tb,\n"
"                            float beta, float *C, int ldc)\n"
"{\n"
"    __shared__ float As[R2_BM][R2_BK];\n"
"    __shared__ float Bs[R2_BK][R2_BN];\n"
"    int tx = threadIdx.x;\n"
"    int ty = threadIdx.y;\n"
"    int tid = ty * 16 + tx;\n"
"    int row0 = blockIdx.y * R2_BM + ty * R2_TM;\n"
"    int col0 = blockIdx.x * R2_BN + tx * R2_TN;\n"
"    float acc[R2_TM][R2_TN];\n"
"    for (int i = 0; i < R2_TM; i++)\n"
"        for (int j = 0; j < R2_TN; j++)\n"
"            acc[i][j] = 0.0f;\n"
"    for (int t = 0; t < K; t += R2_BK) {\n"
"        for (int i = 0; i < (R2_BM * R2_BK) / 256; i++) {\n"
"            int idx = i * 256 + tid;\n"
"            int r = idx / R2_BK;\n"
"            int c = idx % R2_BK;\n"
"            int gr = blockIdx.y * R2_BM + r;\n"
"            int gc = t + c;\n"
"            float v = 0.0f;\n"
"            if (gr < M && gc < K) v = ta ? A[gc * lda + gr] : A[gr * lda + gc];\n"
"            As[r][c] = v;\n"
"        }\n"
"        for (int i = 0; i < (R2_BK * R2_BN) / 256; i++) {\n"
"            int idx = i * 256 + tid;\n"
"            int r = idx / R2_BN;\n"
"            int c = idx % R2_BN;\n"
"            int gr = t + r;\n"
"            int gc = blockIdx.x * R2_BN + c;\n"
"            float v = 0.0f;\n"
"            if (gr < K && gc < N) v = tb ? B[gc * ldb + gr] : B[gr * ldb + gc];\n"
"            Bs[r][c] = v;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < R2_BK; kk++) {\n"
"            float aReg[R2_TM];\n"
"            float bReg[R2_TN];\n"
"            for (int i = 0; i < R2_TM; i++) aReg[i] = As[ty * R2_TM + i][kk];\n"
"            for (int j = 0; j < R2_TN; j++) bReg[j] = Bs[kk][tx * R2_TN + j];\n"
"            for (int i = 0; i < R2_TM; i++)\n"
"                for (int j = 0; j < R2_TN; j++)\n"
"                    acc[i][j] += aReg[i] * bReg[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i = 0; i < R2_TM; i++) {\n"
"        for (int j = 0; j < R2_TN; j++) {\n"
"            int r = row0 + i;\n"
"            int c = col0 + j;\n"
"            if (r < M && c < N) {\n"
"                float v = alpha * acc[i][j];\n"
"                if (beta != 0.0f) v += beta * C[r * ldc + c];\n"
"                C[r * ldc + c] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void sgemv(int M, int N, float alpha, const float *A, int lda, int ta,\n"
"                      const float *x, int incx, float beta, float *y, int incy)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int len = ta ? N : M;\n"
"    if (i >= len) return;\n"
"    float sum = 0.0f;\n"
"    if (ta == 0) {\n"
"        for (int j = 0; j < N; j++) sum += A[i * lda + j] * x[j * incx];\n"
"    } else {\n"
"        for (int j = 0; j < M; j++) sum += A[j * lda + i] * x[j * incx];\n"
"    }\n"
"    y[i * incy] = alpha * sum + beta * y[i * incy];\n"
"}\n"
"\n"
"__global__ void saxpy(int n, float alpha, const float *x, float *y)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) y[i] += alpha * x[i];\n"
"}\n"
"\n"
"__global__ void saxpy_vec4(int n4, float alpha, const float *x, float *y)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n4) return;\n"
"    const __global float4 *x4 = (const __global float4 *)x;\n"
"    __global float4 *y4 = (__global float4 *)y;\n"
"    float4 a = x4[i];\n"
"    float4 b = y4[i];\n"
"    b.x += alpha * a.x;\n"
"    b.y += alpha * a.y;\n"
"    b.z += alpha * a.z;\n"
"    b.w += alpha * a.w;\n"
"    y4[i] = b;\n"
"}\n"
"\n"
"__global__ void sscal(int n, float alpha, float *x)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) x[i] *= alpha;\n"
"}\n"
"\n"
"__global__ void relu(int n, float *x)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n && x[i] < 0.0f) x[i] = 0.0f;\n"
"}\n"
"\n"
"__global__ void relu_backward(int n, const float *y, float *dx)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n && y[i] <= 0.0f) dx[i] = 0.0f;\n"
"}\n"
"\n"
"__global__ void sigmoid(int n, const float *x, float *y)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) y[i] = 1.0f / (1.0f + exp(-x[i]));\n"
"}\n"
"\n"
"__global__ void ssub(int n, const float *a, const float *b, float *out)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) out[i] = a[i] - b[i];\n"
"}\n"
"\n"
"__global__ void bias_add(int total, int cols, float *y, const float *bias)\n"
"{\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < total) y[i] += bias[i % cols];\n"
"}\n"
"\n"
"__global__ void col_sum(int rows, int cols, const float *x, float *out)\n"
"{\n"
"    int c = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (c >= cols) return;\n"
"    float s = 0.0f;\n"
"    for (int r = 0; r < rows; r++) s += x[r * cols + c];\n"
"    out[c] = s;\n"
"}\n";

/* ------------------------------------------------------------------ */
/* Lazy module state                                                  */
/* ------------------------------------------------------------------ */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static int             g_ready = 0;
static mudaModule_t    g_mod = NULL;

static mudaFunction_t g_f_sgemm_rb;
static mudaFunction_t g_f_sgemm_rb128;
static mudaFunction_t g_f_sgemm_naive;
static mudaFunction_t g_f_sgemv;
static mudaFunction_t g_f_saxpy;
static mudaFunction_t g_f_saxpy_vec4;
static mudaFunction_t g_f_sscal;
static mudaFunction_t g_f_relu;
static mudaFunction_t g_f_relu_bwd;
static mudaFunction_t g_f_sigmoid;
static mudaFunction_t g_f_ssub;
static mudaFunction_t g_f_bias_add;
static mudaFunction_t g_f_col_sum;

static mudaError_t ensure_ready(void)
{
    pthread_mutex_lock(&g_mtx);
    if (g_ready) {
        pthread_mutex_unlock(&g_mtx);
        return mudaSuccess;
    }
    mudaError_t e = mudaModuleLoadData(&g_mod, kBlasSource);
    if (e != mudaSuccess) {
        pthread_mutex_unlock(&g_mtx);
        return e;
    }
#define GET(f, name)                                                        \
    do {                                                                    \
        e = mudaModuleGetFunction(&f, g_mod, name);                         \
        if (e != mudaSuccess) {                                             \
            pthread_mutex_unlock(&g_mtx);                                   \
            return e;                                                       \
        }                                                                   \
    } while (0)
    GET(g_f_sgemm_rb,    "sgemm_rb");
    GET(g_f_sgemm_rb128, "sgemm_rb128");
    GET(g_f_sgemm_naive, "sgemm_naive");
    GET(g_f_sgemv,       "sgemv");
    GET(g_f_saxpy,       "saxpy");
    GET(g_f_saxpy_vec4,  "saxpy_vec4");
    GET(g_f_sscal,       "sscal");
    GET(g_f_relu,        "relu");
    GET(g_f_relu_bwd,    "relu_backward");
    GET(g_f_sigmoid,     "sigmoid");
    GET(g_f_ssub,        "ssub");
    GET(g_f_bias_add,    "bias_add");
    GET(g_f_col_sum,     "col_sum");
#undef GET
    g_ready = 1;
    pthread_mutex_unlock(&g_mtx);
    return mudaSuccess;
}

/* ------------------------------------------------------------------ */
/* GEMM                                                               */
/* ------------------------------------------------------------------ */
static mudaError_t sgemm_impl(mudaFunction_t fn, int tile_x, int tile_y,
                              int block_x, int block_y,
                              int transa, int transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta, float *C, int ldc)
{
    if (m <= 0 || n <= 0)
        return mudaSuccess;
    if (k < 0 || !A || !B || !C)
        return mudaErrorInvalidValue;

    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;

    int ta = transa ? 1 : 0;
    int tb = transb ? 1 : 0;
    float a = alpha, b = beta;
    void *args[] = { &m, &n, &k, &a, &A, &lda, &ta, &B, &ldb, &tb, &b, &C, &ldc };

    unsigned grid_x = (unsigned)((n + tile_x - 1) / tile_x);
    unsigned grid_y = (unsigned)((m + tile_y - 1) / tile_y);
    return mudaLaunchKernel(fn, grid_x, grid_y, 1,
                            (unsigned)block_x, (unsigned)block_y, 1,
                            0, NULL, args, NULL);
}

mudaError_t mudaSgemm(int transa, int transb, int m, int n, int k,
                      float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc)
{
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    /* The 128x128 tile wins for larger matrices; 64x64 is better for
     * small ones (less wasted work and lower register pressure). */
    if (m >= 1024 && n >= 1024)
        return sgemm_impl(g_f_sgemm_rb128, 64, 128, 16, 16, transa, transb, m, n, k,
                          alpha, A, lda, B, ldb, beta, C, ldc);
    return sgemm_impl(g_f_sgemm_rb, 64, 64, 16, 16, transa, transb, m, n, k,
                      alpha, A, lda, B, ldb, beta, C, ldc);
}

mudaError_t mudaSgemmNaive(int transa, int transb, int m, int n, int k,
                           float alpha, const float *A, int lda,
                           const float *B, int ldb, float beta, float *C, int ldc)
{
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    return sgemm_impl(g_f_sgemm_naive, 16, 16, 16, 16, transa, transb, m, n, k,
                      alpha, A, lda, B, ldb, beta, C, ldc);
}

/* ------------------------------------------------------------------ */
/* GEMV                                                               */
/* ------------------------------------------------------------------ */
mudaError_t mudaSgemv(int trans, int m, int n,
                      float alpha, const float *A, int lda,
                      const float *x, int incx,
                      float beta, float *y, int incy)
{
    if (m <= 0 || n <= 0)
        return mudaSuccess;
    if (!A || !x || !y)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;

    int ta = trans ? 1 : 0;
    float a = alpha, b = beta;
    int len = ta ? n : m;
    void *args[] = { &m, &n, &a, &A, &lda, &ta, &x, &incx, &b, &y, &incy };

    unsigned grid = (unsigned)((len + 255) / 256);
    return mudaLaunchKernel(g_f_sgemv, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

/* ------------------------------------------------------------------ */
/* Elementwise                                                        */
/* ------------------------------------------------------------------ */
mudaError_t mudaSaxpy(int n, float alpha, const float *x, int incx,
                      float *y, int incy)
{
    if (n <= 0)
        return mudaSuccess;
    if (!x || !y)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;

    float a = alpha;
    if (incx == 1 && incy == 1 && (n % 4) == 0) {
        int n4 = n / 4;
        void *args[] = { &n4, &a, &x, &y };
        unsigned grid = (unsigned)((n4 + 255) / 256);
        return mudaLaunchKernel(g_f_saxpy_vec4, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    }
    /* Only unit strides are supported by the scalar fallback. */
    if (incx != 1 || incy != 1)
        return mudaErrorNotSupported;
    void *args[] = { &n, &a, &x, &y };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_saxpy, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSscal(int n, float alpha, float *x, int incx)
{
    if (n <= 0)
        return mudaSuccess;
    if (!x)
        return mudaErrorInvalidValue;
    if (incx != 1)
        return mudaErrorNotSupported;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    float a = alpha;
    void *args[] = { &n, &a, &x };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_sscal, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSrelu(int n, float *x)
{
    if (n <= 0)
        return mudaSuccess;
    if (!x)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    void *args[] = { &n, &x };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_relu, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSreluBackward(int n, const float *y, float *dx)
{
    if (n <= 0)
        return mudaSuccess;
    if (!y || !dx)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    void *args[] = { &n, &y, &dx };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_relu_bwd, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSsigmoid(int n, const float *x, float *y)
{
    if (n <= 0)
        return mudaSuccess;
    if (!x || !y)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    void *args[] = { &n, &x, &y };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_sigmoid, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSsub(int n, const float *a, const float *b, float *out)
{
    if (n <= 0)
        return mudaSuccess;
    if (!a || !b || !out)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    void *args[] = { &n, &a, &b, &out };
    unsigned grid = (unsigned)((n + 255) / 256);
    return mudaLaunchKernel(g_f_ssub, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaSaddBias(int rows, int cols, float *y, const float *bias)
{
    if (rows <= 0 || cols <= 0)
        return mudaSuccess;
    if (!y || !bias)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    int total = rows * cols;
    void *args[] = { &total, &cols, &y, &bias };
    unsigned grid = (unsigned)((total + 255) / 256);
    return mudaLaunchKernel(g_f_bias_add, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}

mudaError_t mudaScolSum(int rows, int cols, const float *x, float *out)
{
    if (rows <= 0 || cols <= 0)
        return mudaSuccess;
    if (!x || !out)
        return mudaErrorInvalidValue;
    mudaError_t e = ensure_ready();
    if (e != mudaSuccess)
        return e;
    void *args[] = { &rows, &cols, &x, &out };
    unsigned grid = (unsigned)((cols + 255) / 256);
    return mudaLaunchKernel(g_f_col_sum, grid, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
}
