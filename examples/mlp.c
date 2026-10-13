/*
 * mlp.c - train a small multi-layer perceptron on the Tesla M40 with muda.
 *
 * Task: classify points inside vs outside a circle (a nonlinear boundary).
 * The network is 2 -> 32 -> 32 -> 1 with ReLU hidden layers and a sigmoid
 * output, trained with full-batch gradient descent.
 *
 * Every forward and backward pass is expressed with mudaBLAS SGEMM calls,
 * so this exercises memory management, kernel launches, transposed GEMMs,
 * ReLU/sigmoid elementwise kernels and reductions.
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

#define BATCH   1024
#define IN_DIM  2
#define H1_DIM  32
#define H2_DIM  32
#define OUT_DIM 1
#define EPOCHS  4000
#define LR      0.5f

static float urand(unsigned *seed)
{
    *seed = (*seed * 1664525u + 1013904223u);
    return (float)((*seed >> 8) & 0xFFFFFF) / (float)0x800000 - 1.0f;
}

int main(void)
{
    CHECK(mudaInit(0));

    mudaDeviceProp prop;
    CHECK(mudaGetDeviceProperties(&prop, 0));
    printf("training a %d-%d-%d-%d MLP on %s (sm_%d%d)\n",
           IN_DIM, H1_DIM, H2_DIM, OUT_DIM, prop.name,
           prop.computeCapabilityMajor, prop.computeCapabilityMinor);

    /* ---- data: circle classification with 5% label noise ---- */
    float *hx = (float *)malloc(BATCH * IN_DIM * sizeof(float));
    float *hy = (float *)malloc(BATCH * OUT_DIM * sizeof(float));
    unsigned seed = 42u;
    for (int i = 0; i < BATCH; i++) {
        float x = urand(&seed);
        float y = urand(&seed);
        float label = (x * x + y * y < 0.45f) ? 1.0f : 0.0f;
        if (urand(&seed) < -0.9f)
            label = 1.0f - label;
        hx[i * IN_DIM + 0] = x;
        hx[i * IN_DIM + 1] = y;
        hy[i] = label;
    }

    /* ---- weights ---- */
    float *hW1 = (float *)malloc(IN_DIM * H1_DIM * sizeof(float));
    float *hb1 = (float *)calloc(H1_DIM, sizeof(float));
    float *hW2 = (float *)malloc(H1_DIM * H2_DIM * sizeof(float));
    float *hb2 = (float *)calloc(H2_DIM, sizeof(float));
    float *hW3 = (float *)malloc(H2_DIM * OUT_DIM * sizeof(float));
    float *hb3 = (float *)calloc(OUT_DIM, sizeof(float));

    float s1 = sqrtf(1.0f / IN_DIM);
    float s2 = sqrtf(1.0f / H1_DIM);
    float s3 = sqrtf(1.0f / H2_DIM);
    for (int i = 0; i < IN_DIM * H1_DIM; i++) hW1[i] = urand(&seed) * s1;
    for (int i = 0; i < H1_DIM * H2_DIM; i++) hW2[i] = urand(&seed) * s2;
    for (int i = 0; i < H2_DIM * OUT_DIM; i++) hW3[i] = urand(&seed) * s3;

    /* ---- device buffers ---- */
    void *dX = NULL, *dY = NULL;
    void *dW1 = NULL, *db1 = NULL, *dW2 = NULL, *db2 = NULL, *dW3 = NULL, *db3 = NULL;
    void *dA1 = NULL, *dA2 = NULL, *dZ3 = NULL, *dP = NULL;
    void *dG3 = NULL, *dGA2 = NULL, *dGA1 = NULL;
    void *dDW1 = NULL, *dDB1 = NULL, *dDW2 = NULL, *dDB2 = NULL, *dDW3 = NULL, *dDB3 = NULL;
    float *hp = (float *)malloc(BATCH * OUT_DIM * sizeof(float));

#define ALLOC(ptr, n) CHECK(mudaMalloc(&ptr, (size_t)(n) * sizeof(float)))
#define PUSH(ptr, h, n) CHECK(mudaMemcpy(ptr, h, (size_t)(n) * sizeof(float), mudaMemcpyHostToDevice))
#define PULL(h, ptr, n) CHECK(mudaMemcpy(h, ptr, (size_t)(n) * sizeof(float), mudaMemcpyDeviceToHost))

    ALLOC(dX, BATCH * IN_DIM);   ALLOC(dY, BATCH * OUT_DIM);
    ALLOC(dW1, IN_DIM * H1_DIM); ALLOC(db1, H1_DIM);
    ALLOC(dW2, H1_DIM * H2_DIM); ALLOC(db2, H2_DIM);
    ALLOC(dW3, H2_DIM * OUT_DIM); ALLOC(db3, OUT_DIM);
    ALLOC(dA1, BATCH * H1_DIM);  ALLOC(dA2, BATCH * H2_DIM);
    ALLOC(dZ3, BATCH * OUT_DIM); ALLOC(dP, BATCH * OUT_DIM);
    ALLOC(dG3, BATCH * OUT_DIM); ALLOC(dGA2, BATCH * H2_DIM); ALLOC(dGA1, BATCH * H1_DIM);
    ALLOC(dDW1, IN_DIM * H1_DIM); ALLOC(dDB1, H1_DIM);
    ALLOC(dDW2, H1_DIM * H2_DIM); ALLOC(dDB2, H2_DIM);
    ALLOC(dDW3, H2_DIM * OUT_DIM); ALLOC(dDB3, OUT_DIM);

    PUSH(dX, hx, BATCH * IN_DIM);
    PUSH(dY, hy, BATCH * OUT_DIM);
    PUSH(dW1, hW1, IN_DIM * H1_DIM);  PUSH(db1, hb1, H1_DIM);
    PUSH(dW2, hW2, H1_DIM * H2_DIM);  PUSH(db2, hb2, H2_DIM);
    PUSH(dW3, hW3, H2_DIM * OUT_DIM); PUSH(db3, hb3, OUT_DIM);

    /* ---- training ---- */
    float lr = -LR / (float)BATCH; /* fold the 1/batch mean into the step */
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        /* forward */
        CHECK(mudaSgemm(0, 0, BATCH, H1_DIM, IN_DIM, 1.0f, dX, IN_DIM, dW1, H1_DIM, 0.0f, dA1, H1_DIM));
        CHECK(mudaSaddBias(BATCH, H1_DIM, dA1, db1));
        CHECK(mudaSrelu(BATCH * H1_DIM, dA1));

        CHECK(mudaSgemm(0, 0, BATCH, H2_DIM, H1_DIM, 1.0f, dA1, H1_DIM, dW2, H2_DIM, 0.0f, dA2, H2_DIM));
        CHECK(mudaSaddBias(BATCH, H2_DIM, dA2, db2));
        CHECK(mudaSrelu(BATCH * H2_DIM, dA2));

        CHECK(mudaSgemm(0, 0, BATCH, OUT_DIM, H2_DIM, 1.0f, dA2, H2_DIM, dW3, OUT_DIM, 0.0f, dZ3, OUT_DIM));
        CHECK(mudaSaddBias(BATCH, OUT_DIM, dZ3, db3));
        CHECK(mudaSsigmoid(BATCH * OUT_DIM, dZ3, dP));

        /* backward: g3 = dL/dlogit = P - Y (mean folded into lr) */
        CHECK(mudaSsub(BATCH * OUT_DIM, dP, dY, dG3));

        CHECK(mudaSgemm(1, 0, H2_DIM, OUT_DIM, BATCH, 1.0f, dA2, H2_DIM, dG3, OUT_DIM, 0.0f, dDW3, OUT_DIM));
        CHECK(mudaScolSum(BATCH, OUT_DIM, dG3, dDB3));

        CHECK(mudaSgemm(0, 1, BATCH, H2_DIM, OUT_DIM, 1.0f, dG3, OUT_DIM, dW3, OUT_DIM, 0.0f, dGA2, H2_DIM));
        CHECK(mudaSreluBackward(BATCH * H2_DIM, dA2, dGA2));

        CHECK(mudaSgemm(1, 0, H1_DIM, H2_DIM, BATCH, 1.0f, dA1, H1_DIM, dGA2, H2_DIM, 0.0f, dDW2, H2_DIM));
        CHECK(mudaScolSum(BATCH, H2_DIM, dGA2, dDB2));

        CHECK(mudaSgemm(0, 1, BATCH, H1_DIM, H2_DIM, 1.0f, dGA2, H2_DIM, dW2, H2_DIM, 0.0f, dGA1, H1_DIM));
        CHECK(mudaSreluBackward(BATCH * H1_DIM, dA1, dGA1));

        CHECK(mudaSgemm(1, 0, IN_DIM, H1_DIM, BATCH, 1.0f, dX, IN_DIM, dGA1, H1_DIM, 0.0f, dDW1, H1_DIM));
        CHECK(mudaScolSum(BATCH, H1_DIM, dGA1, dDB1));

        /* SGD update: param += lr * grad (lr is already negative) */
        CHECK(mudaSaxpy(IN_DIM * H1_DIM, lr, dDW1, 1, dW1, 1));
        CHECK(mudaSaxpy(H1_DIM, lr, dDB1, 1, db1, 1));
        CHECK(mudaSaxpy(H1_DIM * H2_DIM, lr, dDW2, 1, dW2, 1));
        CHECK(mudaSaxpy(H2_DIM, lr, dDB2, 1, db2, 1));
        CHECK(mudaSaxpy(H2_DIM * OUT_DIM, lr, dDW3, 1, dW3, 1));
        CHECK(mudaSaxpy(OUT_DIM, lr, dDB3, 1, db3, 1));

        if (epoch % 400 == 0 || epoch == EPOCHS - 1) {
            CHECK(mudaDeviceSynchronize());
            PULL(hp, dP, BATCH * OUT_DIM);
            int correct = 0;
            for (int i = 0; i < BATCH; i++)
                if ((hp[i] > 0.5f) == (hy[i] > 0.5f))
                    correct++;
            printf("epoch %5d  train accuracy %.3f\n", epoch, (float)correct / BATCH);
        }
    }

    printf("\nfinal training accuracy: see above\n");

    /* cleanup */
    mudaFree(dX);  mudaFree(dY);
    mudaFree(dW1); mudaFree(db1); mudaFree(dW2); mudaFree(db2); mudaFree(dW3); mudaFree(db3);
    mudaFree(dA1); mudaFree(dA2); mudaFree(dZ3); mudaFree(dP);
    mudaFree(dG3); mudaFree(dGA2); mudaFree(dGA1);
    mudaFree(dDW1); mudaFree(dDB1); mudaFree(dDW2); mudaFree(dDB2); mudaFree(dDW3); mudaFree(dDB3);
    free(hx); free(hy); free(hp);
    free(hW1); free(hb1); free(hW2); free(hb2); free(hW3); free(hb3);
    mudaShutdown();
    return 0;
}
