# muda

**A minimal, self-contained CUDA-compatible runtime that runs kernels on the
NVIDIA Tesla M40 — without linking or loading the CUDA driver (`libcuda`).**

`muda` implements a small subset of the CUDA driver API, a CUDA **runtime**
shim (`mudart`), a row-major linear algebra library (`mudaBLAS`) and a Python
binding.  It ships its own device-driver layer: the default driver translates
CUDA-style kernel source into OpenCL C at run time and dispatches it to the
GPU.  This makes it possible to keep using Maxwell (`sm_52`) hardware even
with CUDA toolkits that no longer ship `sm_52` binaries.

```
   your program / python/muda.py
        |  muda.h | mudart.h | mudablas.h
        v
   libmuda.so  ---- driver layer ---- OpenCL ICD ---- Tesla M40
        |
        +-- CUDA C  ->  OpenCL C  translator
```

## Why this exists

The Tesla M40 is a Maxwell GM200 (`sm_52`) compute card.  Recent CUDA releases
dropped `sm_52` from their prebuilt runtime, so `torch` / `llama.cpp` builds can
silently fall back to the CPU.  The card is still perfectly capable, and the
NVIDIA OpenCL runtime still exposes it.

`muda` sidesteps the CUDA toolchain entirely:

* no `libcuda`, no CUDA toolkit, no `nvcc` at run time;
* kernels are written in CUDA C and compiled on the fly;
* the only external dependency is `libOpenCL.so.1`, loaded with `dlopen()`.

## Status

Tested on **Tesla M40 (sm_52)**, driver 580.178.04.  `make test` runs three
suites — the core runtime (35 checks), mudaBLAS (18) and the CUDA runtime shim
(11) — all passing.

Benchmarks (single run, not tuned):

| example | result |
| --- | --- |
| `vector_add` (1M floats) | ~0.18 ms, ~70 GB/s |
| `saxpy` (4M floats) | ~0.24 ms, ~210 GB/s |
| `matmul` 256x256 tiled | ~0.09 ms, ~390 GFLOP/s |
| `bench_gemm` 1024³ (tuned) | ~1.17 TFLOP/s |
| `bench_gemm` 2048³ (tuned) | ~1.49 TFLOP/s |
| `mlp` (2-32-32-1, 4k epochs) | 94% accuracy (95% is Bayes-optimal with 5% label noise) |
| `python/demo_matmul.py` 2048³ | ~1.49 TFLOP/s (same kernel, no host traffic) |

## Build

No CUDA and no OpenCL development headers are required.

```sh
make            # builds build/libmuda.{so,a}, build/mudainfo,
                # build/vector_add, build/saxpy, build/matmul,
                # build/bench_gemm, build/mlp and the test binaries
make test       # runs all three test suites
```

Requirements: a C11 compiler, `make`, and `libdl`.  At run time `libOpenCL.so.1`
must be present (it is part of the NVIDIA driver on this machine).  The Python
binding additionally needs `numpy`:

```sh
uv venv .venv && uv pip install --python .venv/bin/python numpy
make python PYTHON=.venv/bin/python
```

## Quick start

```c
#include <muda.h>

static const char *kernel =
    "__global__ void vector_add(const float *a, const float *b, float *c, int n)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) c[i] = a[i] + b[i];\n"
    "}\n";

int main(void)
{
    mudaInit(0);

    float *a = ..., *b = ..., *c = ...;   /* host buffers */
    void *da, *db, *dc;
    mudaMalloc(&da, bytes);
    mudaMalloc(&db, bytes);
    mudaMalloc(&dc, bytes);
    mudaMemcpy(da, a, bytes, mudaMemcpyHostToDevice);
    mudaMemcpy(db, b, bytes, mudaMemcpyHostToDevice);

    mudaModule_t mod;
    mudaFunction_t fn;
    mudaModuleLoadData(&mod, kernel);
    mudaModuleGetFunction(&fn, mod, "vector_add");

    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    mudaLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL);
    mudaDeviceSynchronize();

    mudaMemcpy(c, dc, bytes, mudaMemcpyDeviceToHost);
    ...
}
```

See `examples/vector_add.c`, `examples/saxpy.c` (streams + events) and
`examples/matmul.c` (tiled `__shared__` memory + `__syncthreads`), and
`tools/mudainfo.c` for a device-information tool.

## What is implemented

| Area | API |
| --- | --- |
| Runtime | `mudaInit`, `mudaShutdown`, `mudaGetDriverName`, `mudaGetLastError`, `mudaPeekAtLastError`, `mudaRuntimeGetVersion`, `mudaDriverGetVersion` |
| Devices | `mudaGetDeviceCount`, `mudaSetDevice`, `mudaGetDevice`, `mudaGetDeviceProperties`, `mudaDeviceGetName`, `mudaDeviceGetAttribute`, `mudaDeviceGetPCIBusId`, `mudaDeviceReset`, `mudaDeviceSynchronize` |
| Contexts | `mudaCtxCreate`, `mudaCtxDestroy`, push/pop/get current, `mudaCtxGetDevice` |
| Memory | `mudaMalloc`, `mudaFree`, `mudaMallocHost`, `mudaFreeHost`, `mudaHostGetDevicePointer`, `mudaMemGetInfo`, `mudaPointerGetAttributes` |
| Copies | `mudaMemcpy`, `mudaMemcpyAsync`, `mudaMemcpy2D`, `mudaMemcpy2DAsync`, `mudaMemset`, `mudaMemsetD8`, `mudaMemsetD32` |
| Modules | `mudaModuleLoadData`, `mudaModuleLoad`, `mudaModuleUnload`, `mudaModuleGetFunction`, `mudaFuncGetAttributes` |
| Launch | `mudaLaunchKernel`, `mudaLaunchKernelEx`, packed `extra[]` arguments, `mudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| Streams | `mudaStreamCreate`, `mudaStreamCreateWithFlags`, `mudaStreamDestroy`, `mudaStreamSynchronize`, `mudaStreamQuery`, `mudaStreamWaitEvent` |
| Events | `mudaEventCreate`, `mudaEventCreateWithFlags`, `mudaEventRecord`, `mudaEventSynchronize`, `mudaEventQuery`, `mudaEventElapsedTime` |
| Advisories | `mudaMemAdvise`, `mudaMemPrefetchAsync` (no-ops on this driver) |

### Kernel language

Kernels are written in CUDA C.  The translator rewrites the syntactic
difference to OpenCL C:

| CUDA | OpenCL |
| --- | --- |
| `__global__` | `__kernel` |
| `__device__`, `__host__` | removed |
| `__shared__` | `__local` |
| `__syncthreads()`, `__syncwarp()` | `barrier(...)` |
| `threadIdx.x/y/z` | `get_local_id(0/1/2)` |
| `blockIdx.x/y/z` | `get_group_id(0/1/2)` |
| `blockDim.x/y/z` | `get_local_size(0/1/2)` |
| `gridDim.x/y/z` | `get_num_groups(0/1/2)` |

Bare pointer parameters of a kernel are automatically given the `__global`
address-space qualifier, so ordinary CUDA signatures such as
`void k(const float *a, float *b)` work unchanged.  Kernel arguments are passed
exactly like `cuLaunchKernel`: an array of pointers to the argument values.

## CUDA runtime shim (`mudart`)

`include/mudart.h` + `src/mudart.c` provide the CUDA **runtime** API names on
top of muda: `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaMemGetInfo`,
`cudaStream*`, `cudaEvent*`, `cudaGetDevice*`, `cudaDeviceSynchronize`,
`cudaGetLastError`, and a launch bridge.  Kernels are still loaded by name
(there is no `nvcc` in the loop):

```c
#include "mudart.h"

mudaFunction_t fn;
mudaModule_t mod;
mudaModuleLoadData(&mod, kernel_src);
mudaModuleGetFunction(&fn, mod, "vector_add");

int n = 1 << 20;
void *args[] = { &da, &db, &dc, &n };
cudaLaunchKernel((const void *)fn, make_dim3(grid, 1, 1),
                 make_dim3(256, 1, 1), args, 0, NULL);
cudaDeviceSynchronize();
```

The legacy `cudaConfigureCall` / `cudaSetupArgument` / `cudaLaunch` sequence is
supported too.  `tests/test_mudart.c` exercises both.

## mudaBLAS

`include/mudablas.h` + `src/mudablas.c` add a small **row-major** linear
algebra library built entirely on muda kernels:

| function | operation |
| --- | --- |
| `mudaSgemm` | `C = alpha*op(A)*op(B) + beta*C`, shared-memory + 4x4 register-blocked |
| `mudaSgemmNaive` | same, one thread per output element (reference) |
| `mudaSgemv` | `y = alpha*op(A)*x + beta*y` |
| `mudaSaxpy` | `y = alpha*x + y` (float4 path) |
| `mudaSscal` | `x = alpha*x` |
| `mudaSrelu` / `mudaSreluBackward` | ReLU forward / backward |
| `mudaSsigmoid` | elementwise sigmoid |
| `mudaSsub` | `out = a - b` |
| `mudaSaddBias` | broadcast bias add |
| `mudaScolSum` | column sums (bias gradients) |

Storage is C row-major (unlike cuBLAS).  `examples/bench_gemm.c` compares the
naive and tuned kernels:

```
    size  naive GFLOP/s  tuned GFLOP/s    speedup
     256          233.9          519.6      2.22x
     512          273.3         1104.1      4.04x
    1024          243.8         1171.4      4.80x
```

`examples/mlp.c` trains a 2-32-32-1 ReLU/sigmoid MLP on a nonlinear circle
classification task using only `mudaSgemm` and the elementwise kernels,
reaching ~94% accuracy (the Bayes limit is 95% given the injected 5% label
noise).

## Python binding

`python/muda.py` is a pure-`ctypes` binding (nothing to compile on the Python
side).  It locates `build/libmuda.so`, exposes the runtime, device queries,
modules, kernels, streams and events, and offers numpy interop plus
`muda.matmul` / `muda.gemm` / `muda.saxpy` backed by mudaBLAS.

```python
import numpy as np, muda
muda.init()
print(muda.device_info())

A = np.random.rand(1024, 1024).astype(np.float32)
B = np.random.rand(1024, 1024).astype(np.float32)
C = muda.matmul(A, B)          # ~1.37 TFLOP/s on the M40 (kernel only)

mod = muda.Module(open("kernel.cu").read())
vadd = mod.function("vector_add")
vadd(grid=4096, block=256, args=[da, db, dc, n], sync=True)
```

See `python/demo_kernel.py` (load + launch a kernel from source) and
`python/demo_matmul.py` (mudaBLAS benchmark).

## Architecture

```
include/muda.h          public CUDA-like API
include/mudablas.h      row-major linear algebra API
include/mudart.h        CUDA runtime API shim
src/muda.c              runtime state + dispatch to the active driver
src/muda_internal.h     the muda_driver vtable (the "driver ABI")
src/muda_translate.c    CUDA C -> OpenCL C rewriter
src/driver_opencl.c     default driver: dlopen(libOpenCL) + dispatch
src/mudablas.c          SGEMM/SGEMV and elementwise kernels
src/mudart.c            CUDA runtime shim implementation
src/cl_min.h            hand-written minimal OpenCL declarations
tools/mudainfo.c        device/PCI information tool
examples/*.c            vector_add, saxpy, matmul, bench_gemm, mlp
python/muda.py          ctypes binding + numpy interop
tests/test_muda.c       core test suite
tests/test_mudablas.c   linear algebra test suite
tests/test_mudart.c     CUDA runtime shim test suite
```

A *muda driver* is a vtable of `init`, `malloc`, `memcpy`, `module_load`,
`launch`, `stream_*`, `event_*` entry points.  Adding a new backend (for
example a native `nouveau`/MMIO driver) only requires implementing
`muda_driver` and registering it in `mudaInit`.

The active backend can be selected with the `MUDA_DRIVER` environment variable
(currently only `opencl`).  Set `MUDA_VERBOSE=1` to dump the generated OpenCL
build log.

## Limitations and honest notes

* This is a **runtime**, not a kernel-mode driver.  The lowest layer is still
  NVIDIA's OpenCL ICD (their user-space driver); `muda` does not
  reimplement GPU bring-up.  What it does provide is a CUDA-compatible surface
  with no dependency on `libcuda` and no need for an `sm_52`-capable CUDA
  toolkit.
* `mudaLaunchKernel`/`mudaLaunchKernelEx` support packed `extra[]` arguments,
  but the packer assumes natural alignment (`align = min(size, 16)`).
* Dynamic `__local` memory is supported for a single local parameter per kernel.
* Struct arguments passed by value are rejected (`mudaErrorNotSupported`)
  because argument sizes are inferred from the kernel signature.
* `mudaMemAdvise` and `mudaMemPrefetchAsync` are no-ops (there is no managed
  memory path yet).
* `mudaCtxPopCurrent` is simplified.

## Roadmap

Done: core CUDA-compatible API (0.2.0), linear algebra + real-workload layer
(0.3.0: mudaBLAS, GEMM benchmark, MLP training), and the CUDA runtime shim +
Python binding (0.4.0).

Next:

* A tiny source preprocessor that rewrites `kernel<<<grid, block>>>(...)` into
  `cudaLaunchKernel(...)`, so `.cu` files using `<<<>>>` compile unchanged
  (there is still no `nvcc`; the muda driver compiles the kernels).
* `driver_opencl`: multiple local-memory parameters, double-buffered GEMM,
  stream priority and overlap hints.
* `driver_cpu`: a reference driver so the test suite runs without a GPU.
* `driver_nouveau`: talk to the open `nouveau` kernel driver directly
  (`drm_nouveau` GEM/pushbuf), bypassing NVIDIA user space altogether.
* Managed memory (`mudaMallocManaged`) with real prefetch/advice semantics.

## License

MIT.  See `LICENSE`.
