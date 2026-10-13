/*
 * test_mudablas.c - correctness tests for mudaBLAS against CPU references.
 */
#include "muda.h"
#include "mudablas.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;
static const char *g_test = "";

#define OK(cond)                                                             \
    do {                                                                     \
        if (cond) {                                                          \
            g_pass++;                                                        \
        } else {                                                             \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s\n", g_test, __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

#define MUDA(expr)                                                           \
    do {                                                                     \
        mudaError_t _e = (expr);                                            \
        if (_e != mudaSuccess) {                                            \
            g_fail++;                                                        \
            printf("  [FAIL] %s: %s:%d: %s -> %s\n", g_test, __FILE__,      \
                   __LINE__, #expr, mudaGetErrorString(_e));                \
        }                                                                    \
    } while (0)

#define TEST(name) do { g_test = name; printf("== %s ==\n", name); } while (0)

static float rnd(unsigned *s)
{
    *s = (*s * 1664525u + 1013904223u);
    return (float)((*s >> 8) & 0xFFFFFF) / 16777216.0f - 0.5f;
}

static void fill(float *p, size_t n, unsigned *s)
{
    for (size_t i = 0; i < n; i++)
        p[i] = rnd(s);
}

static int close_enough(const float *a, const float *b, size_t n, float tol)
{
    for (size_t i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > tol)
            return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
static void gemm_case(const char *name, int ta, int tb,
                      int M, int N, int K, float alpha, float beta)
{
    int lda = ta ? M : K;
    int ldb = tb ? K : N;
    int ldc = N;
    size_t asz = (size_t)M * K, bsz = (size_t)K * N, csz = (size_t)M * N;

    float *A = malloc(asz * 4), *B = malloc(bsz * 4);
    float *C = malloc(csz * 4), *ref = malloc(csz * 4), *got = malloc(csz * 4);
    unsigned seed = 7u;
    fill(A, asz, &seed);
    fill(B, bsz, &seed);
    fill(C, csz, &seed);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                float av = ta ? A[k * lda + i] : A[i * lda + k];
                float bv = tb ? B[j * ldb + k] : B[k * ldb + j];
                s += av * bv;
            }
            ref[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
        }

    void *dA = NULL, *dB = NULL, *dC = NULL;
    MUDA(mudaMalloc(&dA, asz * 4));
    MUDA(mudaMalloc(&dB, bsz * 4));
    MUDA(mudaMalloc(&dC, csz * 4));
    MUDA(mudaMemcpy(dA, A, asz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dB, B, bsz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dC, C, csz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaSgemm(ta, tb, M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dC, csz * 4, mudaMemcpyDeviceToHost));

    OK(close_enough(got, ref, csz, 1e-3f));
    if (!close_enough(got, ref, csz, 1e-3f)) {
        for (size_t i = 0; i < csz; i++)
            if (fabsf(got[i] - ref[i]) > 1e-3f) {
                printf("    %s: first mismatch at %zu: got %f ref %f\n",
                       name, i, got[i], ref[i]);
                break;
            }
    }
    MUDA(mudaFree(dA));
    MUDA(mudaFree(dB));
    MUDA(mudaFree(dC));
    free(A); free(B); free(C); free(ref); free(got);
}

static void test_sgemm(void)
{
    TEST("sgemm");
    gemm_case("NN", 0, 0, 70, 50, 33, 1.0f, 0.0f);
    gemm_case("TN", 1, 0, 70, 50, 33, 1.0f, 0.0f);
    gemm_case("NT", 0, 1, 70, 50, 33, 1.0f, 0.0f);
    gemm_case("TT", 1, 1, 70, 50, 33, 1.0f, 0.0f);
    gemm_case("NN-exact", 0, 0, 64, 64, 16, 1.0f, 0.0f);
    gemm_case("TT-beta", 1, 1, 130, 65, 17, 0.5f, 0.5f);
}

static void test_sgemm_naive_vs_tuned(void)
{
    TEST("sgemm naive vs tuned");
    const int M = 96, N = 80, K = 48;
    size_t asz = (size_t)M * K, bsz = (size_t)K * N, csz = (size_t)M * N;
    float *A = malloc(asz * 4), *B = malloc(bsz * 4);
    float *C1 = malloc(csz * 4), *C2 = malloc(csz * 4);
    unsigned seed = 99u;
    fill(A, asz, &seed);
    fill(B, bsz, &seed);

    void *dA = NULL, *dB = NULL, *dC1 = NULL, *dC2 = NULL;
    MUDA(mudaMalloc(&dA, asz * 4));
    MUDA(mudaMalloc(&dB, bsz * 4));
    MUDA(mudaMalloc(&dC1, csz * 4));
    MUDA(mudaMalloc(&dC2, csz * 4));
    MUDA(mudaMemcpy(dA, A, asz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dB, B, bsz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaSgemm(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC1, N));
    MUDA(mudaSgemmNaive(0, 0, M, N, K, 1.0f, dA, K, dB, N, 0.0f, dC2, N));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(C1, dC1, csz * 4, mudaMemcpyDeviceToHost));
    MUDA(mudaMemcpy(C2, dC2, csz * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(C1, C2, csz, 1e-3f));
    MUDA(mudaFree(dA)); MUDA(mudaFree(dB)); MUDA(mudaFree(dC1)); MUDA(mudaFree(dC2));
    free(A); free(B); free(C1); free(C2);
}

static void test_gemv(void)
{
    TEST("sgemv");
    const int M = 37, N = 23;
    size_t asz = (size_t)M * N;
    float *A = malloc(asz * 4), *x = malloc((size_t)(M > N ? M : N) * 4);
    float *y = malloc((size_t)(M > N ? M : N) * 4);
    float *ref = malloc((size_t)(M > N ? M : N) * 4);
    float *got = malloc((size_t)(M > N ? M : N) * 4);
    unsigned seed = 3u;
    fill(A, asz, &seed);
    fill(x, (size_t)(M > N ? M : N), &seed);
    fill(y, (size_t)(M > N ? M : N), &seed);
    memcpy(ref, y, (size_t)(M > N ? M : N) * 4);

    void *dA = NULL, *dx = NULL, *dy = NULL;
    MUDA(mudaMalloc(&dA, asz * 4));
    MUDA(mudaMalloc(&dx, (size_t)(M > N ? M : N) * 4));
    MUDA(mudaMalloc(&dy, (size_t)(M > N ? M : N) * 4));
    MUDA(mudaMemcpy(dA, A, asz * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dx, x, (size_t)(M > N ? M : N) * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(dy, y, (size_t)(M > N ? M : N) * 4, mudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.5f;

    /* trans = 0: y(M) = alpha*A*x(N) + beta*y */
    for (int i = 0; i < M; i++) {
        float s = 0.0f;
        for (int j = 0; j < N; j++)
            s += A[i * N + j] * x[j];
        ref[i] = alpha * s + beta * y[i];
    }
    MUDA(mudaSgemv(0, M, N, alpha, dA, N, dx, 1, beta, dy, 1));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dy, (size_t)M * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, M, 1e-3f));

    /* trans = 1: y(N) = alpha*A^T*x(M) + beta*y */
    MUDA(mudaMemcpy(dy, y, (size_t)(M > N ? M : N) * 4, mudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j < M; j++)
            s += A[j * N + i] * x[j];
        ref[i] = alpha * s + beta * y[i];
    }
    MUDA(mudaSgemv(1, M, N, alpha, dA, N, dx, 1, beta, dy, 1));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dy, (size_t)N * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, N, 1e-3f));

    MUDA(mudaFree(dA)); MUDA(mudaFree(dx)); MUDA(mudaFree(dy));
    free(A); free(x); free(y); free(ref); free(got);
}

static void test_axpy_scal(void)
{
    TEST("saxpy / sscal");
    for (int n = 1000; n <= 1001; n++) {
        float *x = malloc(n * 4), *y = malloc(n * 4), *ref = malloc(n * 4), *got = malloc(n * 4);
        unsigned seed = 11u;
        fill(x, n, &seed);
        fill(y, n, &seed);
        memcpy(ref, y, n * 4);
        const float alpha = 0.75f;
        for (int i = 0; i < n; i++)
            ref[i] += alpha * x[i];

        void *dx = NULL, *dy = NULL;
        MUDA(mudaMalloc(&dx, n * 4));
        MUDA(mudaMalloc(&dy, n * 4));
        MUDA(mudaMemcpy(dx, x, n * 4, mudaMemcpyHostToDevice));
        MUDA(mudaMemcpy(dy, y, n * 4, mudaMemcpyHostToDevice));
        MUDA(mudaSaxpy(n, alpha, dx, 1, dy, 1));
        MUDA(mudaDeviceSynchronize());
        MUDA(mudaMemcpy(got, dy, n * 4, mudaMemcpyDeviceToHost));
        OK(close_enough(got, ref, n, 1e-3f));

        /* sscal on top */
        const float beta = -0.5f;
        for (int i = 0; i < n; i++)
            ref[i] *= beta;
        MUDA(mudaSscal(n, beta, dy, 1));
        MUDA(mudaDeviceSynchronize());
        MUDA(mudaMemcpy(got, dy, n * 4, mudaMemcpyDeviceToHost));
        OK(close_enough(got, ref, n, 1e-3f));

        MUDA(mudaFree(dx)); MUDA(mudaFree(dy));
        free(x); free(y); free(ref); free(got);
    }
}

static void test_activations(void)
{
    TEST("activations");
    const int n = 257;
    float *x = malloc(n * 4), *ref = malloc(n * 4), *got = malloc(n * 4);
    unsigned seed = 5u;
    fill(x, n, &seed);

    void *dx = NULL, *dy = NULL;
    MUDA(mudaMalloc(&dx, n * 4));
    MUDA(mudaMalloc(&dy, n * 4));
    MUDA(mudaMemcpy(dx, x, n * 4, mudaMemcpyHostToDevice));

    /* relu */
    for (int i = 0; i < n; i++)
        ref[i] = x[i] > 0 ? x[i] : 0.0f;
    MUDA(mudaMemcpy(dy, dx, n * 4, mudaMemcpyDeviceToDevice));
    MUDA(mudaSrelu(n, dy));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dy, n * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, n, 1e-6f));

    /* relu backward: dx *= (y > 0) */
    for (int i = 0; i < n; i++)
        ref[i] = x[i] > 0 ? x[i] : 0.0f;
    MUDA(mudaMemcpy(dy, x, n * 4, mudaMemcpyHostToDevice));
    MUDA(mudaSreluBackward(n, dx, dy)); /* y = raw x, so y<=0 => zero */
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dy, n * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, n, 1e-6f));

    /* sigmoid */
    for (int i = 0; i < n; i++)
        ref[i] = 1.0f / (1.0f + expf(-x[i]));
    MUDA(mudaSsigmoid(n, dx, dy));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dy, n * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, n, 1e-5f));

    MUDA(mudaFree(dx)); MUDA(mudaFree(dy));
    free(x); free(ref); free(got);
}

static void test_bias_colsum(void)
{
    TEST("add_bias / col_sum");
    const int rows = 13, cols = 7;
    size_t n = (size_t)rows * cols;
    float *x = malloc(n * 4), *bias = malloc(cols * 4);
    float *ref = malloc(n * 4), *got = malloc(n * 4), *col = malloc(cols * 4);
    unsigned seed = 21u;
    fill(x, n, &seed);
    fill(bias, cols, &seed);

    void *dx = NULL, *db = NULL, *dout = NULL;
    MUDA(mudaMalloc(&dx, n * 4));
    MUDA(mudaMalloc(&db, cols * 4));
    MUDA(mudaMalloc(&dout, cols * 4));
    MUDA(mudaMemcpy(dx, x, n * 4, mudaMemcpyHostToDevice));
    MUDA(mudaMemcpy(db, bias, cols * 4, mudaMemcpyHostToDevice));

    for (size_t i = 0; i < n; i++)
        ref[i] = x[i] + bias[i % cols];
    MUDA(mudaSaddBias(rows, cols, dx, db));
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(got, dx, n * 4, mudaMemcpyDeviceToHost));
    OK(close_enough(got, ref, n, 1e-5f));

    for (int c = 0; c < cols; c++) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++)
            s += x[r * cols + c];
        ref[c] = s;
    }
    MUDA(mudaScolSum(rows, cols, dx, dout)); /* sum of (x+bias) */
    MUDA(mudaDeviceSynchronize());
    MUDA(mudaMemcpy(col, dout, cols * 4, mudaMemcpyDeviceToHost));
    /* adjust reference: dx now contains x+bias, so expected = ref + rows*bias */
    int ok = 1;
    for (int c = 0; c < cols; c++)
        if (fabsf(col[c] - (ref[c] + rows * bias[c])) > 1e-3f)
            ok = 0;
    OK(ok);

    MUDA(mudaFree(dx)); MUDA(mudaFree(db)); MUDA(mudaFree(dout));
    free(x); free(bias); free(ref); free(got); free(col);
}

int main(void)
{
    printf("mudaBLAS test suite\n\n");

    if (mudaInit(0) != mudaSuccess) {
        printf("mudaInit failed: %s\n", mudaGetErrorString(mudaGetLastError()));
        return 1;
    }

    test_sgemm();
    test_sgemm_naive_vs_tuned();
    test_gemv();
    test_axpy_scal();
    test_activations();
    test_bias_colsum();

    mudaShutdown();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
