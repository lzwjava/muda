"""
muda.py - Python binding for the muda runtime and mudaBLAS.

The binding is pure ctypes, so there is nothing to compile on the Python
side.  It locates build/libmuda.so relative to this file (or falls back to
the system loader path).

    import muda, numpy as np

    muda.init()
    print(muda.device_info())

    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    C = muda.matmul(A, B)          # runs on the GPU through mudaBLAS

    # load and launch your own CUDA-style kernel
    mod = muda.Module('''
        __global__ void vector_add(const float *a, const float *b, float *c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
    ''')
    vadd = mod.function("vector_add")
    da, db, dc = muda.DeviceBuffer.from_numpy(a), muda.DeviceBuffer.from_numpy(b), muda.DeviceBuffer.empty_like(c)
    vadd(grid=(n + 255) // 256, block=256, args=[da, db, dc, n])
"""

import ctypes
import os

import numpy as np

__all__ = [
    "MudaError", "DeviceProp", "DeviceBuffer", "Module", "Function",
    "init", "shutdown", "driver_name", "device_count", "device_info",
    "malloc", "free", "memcpy", "memset", "synchronize", "matmul",
    "gemm", "gemm_buffers", "gemv", "saxpy", "stream", "event", "lib",
]

# --------------------------------------------------------------------- #
# Library loading
# --------------------------------------------------------------------- #
def _find_library():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "build", "libmuda.so"),
        os.path.join(here, "build", "libmuda.so"),
        "libmuda.so",
    ]
    for cand in candidates:
        if os.path.sep in cand and not os.path.exists(cand):
            continue
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    raise OSError(
        "could not find libmuda.so; run `make` in the repository root first"
    )


lib = _find_library()

# --------------------------------------------------------------------- #
# Constants / structs
# --------------------------------------------------------------------- #
MEMCPY_HOST_TO_HOST = 0
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2
MEMCPY_DEVICE_TO_DEVICE = 3

STREAM_NON_BLOCKING = 0x01
EVENT_DISABLE_TIMING = 0x01


class MudaError(RuntimeError):
    pass


class DeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("driverVersion", ctypes.c_char * 64),
        ("deviceVersion", ctypes.c_char * 64),
        ("computeCapabilityMajor", ctypes.c_int),
        ("computeCapabilityMinor", ctypes.c_int),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("sharedMemPerMultiprocessor", ctypes.c_size_t),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("maxRegistersPerBlock", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("clockRateKHz", ctypes.c_int),
        ("memoryClockRateKHz", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("concurrentKernels", ctypes.c_int),
    ]


# --------------------------------------------------------------------- #
# C signatures
# --------------------------------------------------------------------- #
def _sig(name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_c_init = _sig("mudaInit", ctypes.c_int, ctypes.c_uint)
_c_shutdown = _sig("mudaShutdown", ctypes.c_int)
_c_driver_name = _sig("mudaGetDriverName", ctypes.c_char_p)
_c_device_count = _sig("mudaGetDeviceCount", ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_c_device_props = _sig("mudaGetDeviceProperties", ctypes.c_int,
                       ctypes.POINTER(DeviceProp), ctypes.c_int)
_c_malloc = _sig("mudaMalloc", ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                 ctypes.c_size_t)
_c_malloc_host = _sig("mudaMallocHost", ctypes.c_int,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
_c_free = _sig("mudaFree", ctypes.c_int, ctypes.c_void_p)
_c_free_host = _sig("mudaFreeHost", ctypes.c_int, ctypes.c_void_p)
_c_memcpy = _sig("mudaMemcpy", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_size_t, ctypes.c_int)
_c_memset = _sig("mudaMemset", ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                 ctypes.c_size_t)
_c_sync = _sig("mudaDeviceSynchronize", ctypes.c_int)
_c_module_load = _sig("mudaModuleLoadData", ctypes.c_int,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p)
_c_module_unload = _sig("mudaModuleUnload", ctypes.c_int, ctypes.c_void_p)
_c_func_get = _sig("mudaModuleGetFunction", ctypes.c_int,
                   ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                   ctypes.c_char_p)
_c_launch = _sig("mudaLaunchKernel", ctypes.c_int, ctypes.c_void_p,
                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                 ctypes.c_uint, ctypes.c_void_p,
                 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
_c_stream_create = _sig("mudaStreamCreate", ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p))
_c_stream_create_flags = _sig("mudaStreamCreateWithFlags", ctypes.c_int,
                              ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint)
_c_stream_destroy = _sig("mudaStreamDestroy", ctypes.c_int, ctypes.c_void_p)
_c_stream_sync = _sig("mudaStreamSynchronize", ctypes.c_int, ctypes.c_void_p)
_c_event_create = _sig("mudaEventCreate", ctypes.c_int,
                       ctypes.POINTER(ctypes.c_void_p))
_c_event_destroy = _sig("mudaEventDestroy", ctypes.c_int, ctypes.c_void_p)
_c_event_record = _sig("mudaEventRecord", ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_void_p)
_c_event_sync = _sig("mudaEventSynchronize", ctypes.c_int, ctypes.c_void_p)
_c_event_elapsed = _sig("mudaEventElapsedTime", ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                        ctypes.c_void_p)
_c_error_string = _sig("mudaGetErrorString", ctypes.c_char_p, ctypes.c_int)

_c_sgemm = _sig("mudaSgemm", ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_float, ctypes.c_void_p, ctypes.c_int)
_c_sgemv = _sig("mudaSgemv", ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_float, ctypes.c_void_p, ctypes.c_int)
_c_saxpy = _sig("mudaSaxpy", ctypes.c_int, ctypes.c_int, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)


def _check(err):
    if err != 0:
        raise MudaError(_c_error_string(err).decode())
    return err


# --------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------- #
_initialized = [False]


def init(flags=0):
    _check(_c_init(flags))
    _initialized[0] = True


def shutdown():
    _check(_c_shutdown())
    _initialized[0] = False


def driver_name():
    return _c_driver_name().decode()


def device_count():
    n = ctypes.c_int(0)
    _check(_c_device_count(ctypes.byref(n)))
    return n.value


def device_info(index=0):
    p = DeviceProp()
    _check(_c_device_props(ctypes.byref(p), index))
    return {
        "name": p.name.decode(),
        "compute_capability": f"{p.computeCapabilityMajor}.{p.computeCapabilityMinor}",
        "total_global_mem": p.totalGlobalMem,
        "shared_mem_per_block": p.sharedMemPerBlock,
        "multiprocessors": p.multiProcessorCount,
        "max_threads_per_block": p.maxThreadsPerBlock,
        "warp_size": p.warpSize,
        "clock_khz": p.clockRateKHz,
        "pci_bus_id": p.pciBusID,
        "pci_device_id": p.pciDeviceID,
        "driver_version": p.driverVersion.decode(),
        "device_version": p.deviceVersion.decode(),
    }


def malloc(nbytes):
    p = ctypes.c_void_p()
    _check(_c_malloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes))))
    return p


def free(ptr):
    _check(_c_free(ptr if isinstance(ptr, ctypes.c_void_p) else ctypes.c_void_p(ptr)))


def memcpy(dst, src, count, kind):
    _check(_c_memcpy(dst, src, ctypes.c_size_t(int(count)), int(kind)))


def memset(ptr, value, count):
    _check(_c_memset(ptr, int(value), ctypes.c_size_t(int(count))))


def synchronize():
    _check(_c_sync())


# --------------------------------------------------------------------- #
# Device buffers
# --------------------------------------------------------------------- #
class DeviceBuffer:
    """A chunk of device memory.  ``ptr`` is a ctypes.c_void_p."""

    def __init__(self, nbytes):
        self.nbytes = int(nbytes)
        self.ptr = malloc(self.nbytes)
        self._freed = False

    @classmethod
    def from_numpy(cls, arr):
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes)
        memcpy(buf.ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes,
               MEMCPY_HOST_TO_DEVICE)
        return buf

    @classmethod
    def empty_like(cls, arr):
        return cls(np.ascontiguousarray(arr).nbytes)

    def to_numpy(self, shape=None, dtype=np.float32):
        dtype = np.dtype(dtype)
        if shape is None:
            shape = (self.nbytes // dtype.itemsize,)
        out = np.empty(shape, dtype=dtype)
        nbytes = out.nbytes
        if nbytes > self.nbytes:
            raise ValueError("destination larger than device buffer")
        memcpy(out.ctypes.data_as(ctypes.c_void_p), self.ptr, nbytes,
               MEMCPY_DEVICE_TO_HOST)
        return out

    def copy_from(self, arr):
        arr = np.ascontiguousarray(arr)
        memcpy(self.ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes,
               MEMCPY_HOST_TO_DEVICE)

    def copy_to(self, arr):
        memcpy(arr.ctypes.data_as(ctypes.c_void_p), self.ptr, arr.nbytes,
               MEMCPY_DEVICE_TO_HOST)

    def free(self):
        if not self._freed:
            free(self.ptr)
            self._freed = True

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


# --------------------------------------------------------------------- #
# Modules and kernels
# --------------------------------------------------------------------- #
def _dim3(d):
    if isinstance(d, int):
        return (d, 1, 1)
    d = tuple(d)
    if len(d) == 1:
        return (d[0], 1, 1)
    if len(d) == 2:
        return (d[0], d[1], 1)
    return (d[0], d[1], d[2])


def _pack_args(args):
    holders = []
    ptrs = []
    for a in args:
        if isinstance(a, DeviceBuffer):
            obj = ctypes.c_void_p(a.ptr.value)
        elif isinstance(a, ctypes.c_void_p):
            obj = a
        elif isinstance(a, np.generic):
            obj = ctypes.c_float(float(a)) if isinstance(a, np.floating) \
                else ctypes.c_int(int(a))
        elif isinstance(a, bool):
            obj = ctypes.c_int(int(a))
        elif isinstance(a, int):
            obj = ctypes.c_int(a)
        elif isinstance(a, float):
            obj = ctypes.c_float(a)
        else:
            raise TypeError(f"unsupported kernel argument: {type(a)!r}")
        holders.append(obj)
        ptrs.append(ctypes.cast(ctypes.pointer(obj), ctypes.c_void_p))
    arr = (ctypes.c_void_p * len(ptrs))(*ptrs) if ptrs else None
    return holders, arr


class Function:
    def __init__(self, handle, name=""):
        self._f = handle
        self.name = name

    def __call__(self, grid, block, args, shared=0, stream=None, sync=False):
        gx, gy, gz = _dim3(grid)
        bx, by, bz = _dim3(block)
        holders, arr = _pack_args(args)
        st = stream._s if isinstance(stream, Stream) else (stream or None)
        _check(_c_launch(self._f, gx, gy, gz, bx, by, bz, int(shared), st, arr, None))
        if sync:
            synchronize()
        return holders


class Module:
    def __init__(self, source):
        m = ctypes.c_void_p()
        _check(_c_module_load(ctypes.byref(m), source.encode()))
        self._m = m
        self._funcs = {}

    def function(self, name):
        if name not in self._funcs:
            f = ctypes.c_void_p()
            _check(_c_func_get(ctypes.byref(f), self._m, name.encode()))
            self._funcs[name] = Function(f, name)
        return self._funcs[name]

    def unload(self):
        if self._m is not None:
            _check(_c_module_unload(self._m))
            self._m = None
            self._funcs.clear()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


# --------------------------------------------------------------------- #
# Streams and events
# --------------------------------------------------------------------- #
class Stream:
    def __init__(self, flags=0):
        s = ctypes.c_void_p()
        _check(_c_stream_create_flags(ctypes.byref(s), flags))
        self._s = s

    def synchronize(self):
        _check(_c_stream_sync(self._s))

    def free(self):
        if self._s is not None:
            _check(_c_stream_destroy(self._s))
            self._s = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class Event:
    def __init__(self):
        e = ctypes.c_void_p()
        _check(_c_event_create(ctypes.byref(e)))
        self._e = e

    def record(self, stream=None):
        s = stream._s if isinstance(stream, Stream) else (stream or None)
        _check(_c_event_record(self._e, s))

    def synchronize(self):
        _check(_c_event_sync(self._e))

    @staticmethod
    def elapsed_ms(start, end):
        ms = ctypes.c_float(0.0)
        _check(_c_event_elapsed(ctypes.byref(ms), start._e, end._e))
        return ms.value

    def free(self):
        if self._e is not None:
            _check(_c_event_destroy(self._e))
            self._e = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def stream(flags=0):
    return Stream(flags)


def event():
    return Event()


# --------------------------------------------------------------------- #
# mudaBLAS helpers
# --------------------------------------------------------------------- #
def gemm(A, B, alpha=1.0, beta=0.0, C=None):
    """Row-major C = alpha*A@B + beta*C using mudaBLAS on the GPU."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix shapes")
    M, K = A.shape
    N = B.shape[1]
    dA = DeviceBuffer.from_numpy(A)
    dB = DeviceBuffer.from_numpy(B)
    if C is None:
        dC = DeviceBuffer(M * N * 4)
        beta = 0.0
    else:
        C = np.ascontiguousarray(C, dtype=np.float32)
        dC = DeviceBuffer.from_numpy(C)
    _check(_c_sgemm(0, 0, M, N, K, ctypes.c_float(alpha), dA.ptr, K,
                    dB.ptr, N, ctypes.c_float(beta), dC.ptr, N))
    synchronize()
    out = dC.to_numpy((M, N), np.float32)
    if C is not None:
        out = out + 0  # out already holds alpha*AB + beta*C
    dA.free(); dB.free(); dC.free()
    return out


matmul = gemm


def gemm_buffers(dA, dB, dC, M, N, K, alpha=1.0, beta=0.0,
                 lda=None, ldb=None, ldc=None):
    """Launch C = alpha*A@B + beta*C on pre-allocated DeviceBuffers.

    No host/device copies and no synchronisation, so this is the right entry
    point for benchmarking the kernel itself.
    """
    lda = K if lda is None else lda
    ldb = N if ldb is None else ldb
    ldc = N if ldc is None else ldc
    _check(_c_sgemm(0, 0, int(M), int(N), int(K), ctypes.c_float(alpha),
                    dA.ptr, int(lda), dB.ptr, int(ldb),
                    ctypes.c_float(beta), dC.ptr, int(ldc)))


def gemv(A, x, alpha=1.0, beta=0.0, y=None, trans=0):
    A = np.ascontiguousarray(A, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    M, N = A.shape
    dA = DeviceBuffer.from_numpy(A)
    dx = DeviceBuffer.from_numpy(x)
    if trans == 0:
        out_shape = (M,)
    else:
        out_shape = (N,)
    if y is None:
        y = np.zeros(out_shape, dtype=np.float32)
    dy = DeviceBuffer.from_numpy(y)
    _check(_c_sgemv(int(trans), M, N, ctypes.c_float(alpha), dA.ptr, N,
                    dx.ptr, 1, ctypes.c_float(beta), dy.ptr, 1))
    synchronize()
    out = dy.to_numpy(out_shape, np.float32)
    dA.free(); dx.free(); dy.free()
    return out


def saxpy(x, y, alpha=1.0):
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    dx = DeviceBuffer.from_numpy(x)
    dy = DeviceBuffer.from_numpy(y)
    _check(_c_saxpy(x.size, ctypes.c_float(alpha), dx.ptr, 1, dy.ptr, 1))
    synchronize()
    out = dy.to_numpy(y.shape, np.float32)
    dx.free(); dy.free()
    return out
