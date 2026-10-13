# muda

**A minimal, self-contained CUDA-compatible runtime that runs kernels on the
NVIDIA Tesla M40 — without linking or loading the CUDA driver (`libcuda`).**

`muda` implements a small subset of the CUDA driver API and ships its own
device-driver layer.  The default driver translates CUDA-style kernel source
into OpenCL C at run time and dispatches it to the GPU.  This makes it possible
to keep using Maxwell (`sm_52`) hardware even with CUDA toolkits that no longer
ship `sm_52` binaries.

```
   your program
        |  muda.h  (CUDA-like API)
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

Tested on **Tesla M40 (sm_52)**, driver 580.178.04:

```
$ make test
muda 0.2.0 test suite
driver: opencl
== device ==               device 0: Tesla M40 sm_52, 11436 MiB, 24 SMs
== device_queries ==       name=Tesla M40 pci=0000:01:00.0
== memory ==
== pointer_attributes ==
== memcpy_2d ==
== memset_d32 ==
== vector_add ==
== saxpy ==
== shared_memory_+_syncthreads ==
== 2d_grid ==
== packed_launch ==
== launch_ex ==
== func_attributes ==      occupancy@256=1 blocks/SM
== streams_and_events ==
== stream_event_query ==

35 passed, 0 failed
```

Benchmarks (single run, for a feel, not tuned):

| example | result |
| --- | --- |
| `vector_add` (1M floats) | ~0.18 ms, ~70 GB/s |
| `saxpy` (4M floats) | ~0.24 ms, ~210 GB/s |
| `matmul` 256x256 tiled | ~0.09 ms, ~390 GFLOP/s |

## Build

No CUDA and no OpenCL development headers are required.

```sh
make            # builds build/libmuda.{so,a}, build/mudainfo,
                # build/vector_add, build/saxpy, build/matmul and
                # build/test_muda
make test       # runs the test suite
```

Requirements: a C11 compiler, `make`, and `libdl`.  At run time `libOpenCL.so.1`
must be present (it is part of the NVIDIA driver on this machine).

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

## Architecture

```
include/muda.h          public CUDA-like API
src/muda.c              runtime state + dispatch to the active driver
src/muda_internal.h     the muda_driver vtable (the "driver ABI")
src/muda_translate.c    CUDA C -> OpenCL C rewriter
src/driver_opencl.c     default driver: dlopen(libOpenCL) + dispatch
src/cl_min.h            hand-written minimal OpenCL declarations
tools/mudainfo.c        device/PCI information tool
examples/*.c            vector_add, saxpy, matmul
tests/test_muda.c       test suite
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

* `driver_opencl`: multiple local-memory parameters, real free-memory queries,
  stream priority and overlap hints.
* `driver_cpu`: a reference driver so the test suite runs without a GPU.
* `driver_nouveau`: talk to the open `nouveau` kernel driver directly
  (`drm_nouveau` GEM/pushbuf), bypassing NVIDIA user space altogether.
* Managed memory (`mudaMallocManaged`) with real prefetch/advice semantics.
* Load PTX/`cubin` through a native channel instead of OpenCL source.

## License

MIT.  See `LICENSE`.
