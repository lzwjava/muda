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
   your .cu program --(mudac)--> C/C++ --+
                                        |  muda.h | mudart.h | mudablas.h
   python/muda.py ----------------------+
                                        v
                    libmuda.so ---- driver layer ---- OpenCL ICD ---- Tesla M40
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

Tested on **Tesla M40 (sm_52)**, driver 580.178.04.  `make test` runs the core
runtime (35 checks), mudaBLAS (18), the CUDA runtime shim (11), the CUDA
source-compatibility suite (5) and the driver-independent memory suite (8);
builds ordinary `<<<>>>` CUDA programs through `mudac` (vector add and a block
reduction); and runs the `mudac` launch-rewrite tests.  `make test-cpu` runs
the memory suite against the CPU reference driver.

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
make            # builds build/libmuda.{so,a}, build/mudainfo, the
                # examples (vector_add, saxpy, matmul, bench_gemm, mlp),
                # build/mudac and the test binaries
make test       # runs all suites, including a mudac-built program
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

Also handled:

* qualifiers `__launch_bounds__(...)`, `__forceinline__`, `__inline__`,
  `__noinline__`, `__device_builtin__`, `__align__(...)`;
* math intrinsics `fmaf`, `__expf`, `__exp10f`, `__logf`, `__log2f`,
  `__log10f`, `__powf`, `__sinf`, `__cosf`, `__tanf`, `__sqrtf`, `rsqrtf`,
  `__fdividef`, `__saturatef`, `__ldg`;
* atomics `atomicAdd`, `atomicExch`, `atomicMax`, `atomicMin`, `atomicCAS` for
  `int`, `unsigned int`, `unsigned long` and `float`.  Float atomics use a
  compare-and-swap loop; the pointer type is selected with
  `__builtin_types_compatible_p`/`__typeof__` (the device compiler's `_Generic`
  turned out to be unreliable).

`tests/test_translate.c` exercises all of the above.

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

## `mudac` preprocessor

`tools/mudac.c` is a tiny source-to-source preprocessor that lets ordinary
`.cu` files — including the `<<<...>>>` launch syntax — compile against muda,
still with no `nvcc`: the muda driver compiles the kernels at run time.

```sh
make build/mudac
build/mudac prog.cu -o prog.cc
c++ -std=gnu++17 -Iinclude prog.cc build/libmuda.a -ldl -lpthread -lm -o prog
./prog
```

It:

* extracts every `__global__`/`__device__` function into a string constant and
  removes the definition from the host code;
* rewrites `kernel<<<grid, block, shared, stream>>>(a, b, n)` into a call to
  `mudartLaunchByName("kernel", ...)`, taking the address of each argument in a
  GNU statement expression so literals and rvalues work;
* replaces `#include <cuda_runtime.h>` with `#include "mudart.h"`.

Launch dimensions may be integers or `dim3` values; the generated code picks
whichever you wrote.  `examples/vector_add.cu` is a normal CUDA program built
this way, and `tests/test_mudac.cu` covers 1D integer dims, 2D `dim3` dims, a
scalar argument and several kernels in one file.

It is a preprocessor, not a compiler: it handles the launch syntax and
device-function extraction, not general C++ metaprogramming that a real `nvcc`
would perform.

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

## Native backend (`driver_nouveau`) - experimental

`src/driver_nouveau.c` talks straight to the open **nouveau** kernel driver
through `/dev/dri/renderD*` using the DRM ioctl ABI, with no dependency on
NVIDIA's user-space stack.  It currently provides:

* device enumeration and properties using `DRM_IOCTL_VERSION`, `GETPARAM`
  (chipset, PCI id, framebuffer size) and the PCI resource sizes;
* GPU memory management: `DRM_IOCTL_NOUVEAU_GEM_NEW` + `mmap`, so `mudaMalloc`
  returns real GEM objects;
* the full stream/event surface.

Kernel execution is **not implemented yet** (`module_load`/`launch` return
`mudaErrorNotSupported`); growing a compute channel on top of this is the next
step.

It requires the nouveau kernel driver to be bound to the M40.  `nouveau` is
blacklisted by the NVIDIA package, so this needs root.  The helper script does
it safely and reversibly:

```sh
tools/muda-gpu-mode.sh status          # show current state (no root)
sudo tools/muda-gpu-mode.sh nouveau    # unbind nvidia, load nouveau
MUDA_DRIVER=nouveau build/mudainfo
sudo tools/muda-gpu-mode.sh nvidia     # switch back
```

Unbinding only affects the running kernel: a reboot always restores the nvidia
driver, because nouveau stays blacklisted.  Stop anything using the M40
(ComfyUI, llama.cpp) first, or `rmmod` will fail.

## Architecture

```
include/muda.h          public CUDA-like API
include/mudablas.h      row-major linear algebra API
include/mudart.h        CUDA runtime API shim
src/muda.c              runtime state + dispatch to the active driver
src/muda_internal.h     the muda_driver vtable (the "driver ABI")
src/muda_translate.c    CUDA C -> OpenCL C rewriter
src/driver_opencl.c     default driver: dlopen(libOpenCL) + dispatch
src/driver_cpu.c        CPU reference driver (memory only; no kernels)
src/driver_nouveau.c    native DRM/nouveau backend (experimental)
src/mudablas.c          SGEMM/SGEMV and elementwise kernels
src/mudart.c            CUDA runtime shim implementation
src/cl_min.h            hand-written minimal OpenCL declarations
tools/mudainfo.c        device/PCI information tool
tools/mudac.c           CUDA `<<<>>>` -> mudart preprocessor
tools/muda-gpu-mode.sh  switch the M40 between nvidia and nouveau
examples/*.c            vector_add, saxpy, matmul, bench_gemm, mlp
examples/*.cu           vector_add, reduction (built through mudac)
python/muda.py          ctypes binding + numpy interop
tests/test_muda.c       core test suite
tests/test_mudablas.c   linear algebra test suite
tests/test_mudart.c     CUDA runtime shim test suite
tests/test_translate.c  CUDA source compatibility test suite
tests/test_memory.c     driver-independent memory/stream/event test
tests/test_mudac.cu     mudac launch-rewrite tests
```

A *muda driver* is a vtable of `init`, `malloc`, `memcpy`, `module_load`,
`launch`, `stream_*`, `event_*` entry points.  Adding a new backend (for
example a native `nouveau`/MMIO driver) only requires implementing
`muda_driver` and registering it in `mudaInit`.

The active backend can be selected with the `MUDA_DRIVER` environment variable:
`opencl` (default), `cpu` or `nouveau`.  If OpenCL is unavailable, `muda`
automatically falls back to the CPU reference driver, which supports the full
memory, stream and event surface but cannot execute kernels.  Set
`MUDA_VERBOSE=1` to dump the generated OpenCL build log, and
`MUDA_CL_OPTIONS="..."` to pass build options (for example `-cl-std=CL3.0`).

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

Done: core CUDA-compatible API (0.2.0); linear algebra + real-workload layer
(0.3.0: mudaBLAS, GEMM benchmark, MLP training); CUDA runtime shim + Python
binding (0.4.0); `<<<>>>` preprocessor, `mudac` (0.5.0); expanded CUDA source
compatibility - intrinsics, atomics, qualifiers (0.6.0); CPU reference driver +
reduction example (0.7.0); native DRM/nouveau memory backend + driver switch
tool (0.8.0, experimental).

Next: give `driver_nouveau` a compute channel (pushbuffer + channel classes)
so kernels can run with no NVIDIA user space at all; until then it is a memory
and information backend.

Next:

* `mudac`: handle `cudaMemcpyToSymbol`, `__constant__` and `__managed__`, and
  source-level `-I` includes, to cover more real `.cu` files.
* `driver_opencl`: multiple local-memory parameters, double-buffered GEMM,
  stream priority and overlap hints.
* `driver_cpu`: a reference driver so the test suite runs without a GPU.
* `driver_nouveau`: talk to the open `nouveau` kernel driver directly
  (`drm_nouveau` GEM/pushbuf), bypassing NVIDIA user space altogether.
* Managed memory (`mudaMallocManaged`) with real prefetch/advice semantics.

## License

MIT.  See `LICENSE`.
