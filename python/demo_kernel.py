#!/usr/bin/env python3
"""Load and launch a CUDA-style kernel from Python with muda."""
import numpy as np

import muda

KERNEL = r"""
__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""


def main():
    muda.init()
    print("device:", muda.device_info()["name"])

    mod = muda.Module(KERNEL)
    vadd = mod.function("vector_add")

    n = 1 << 20
    a = np.arange(n, dtype=np.float32)
    b = np.full(n, 7.0, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)

    da = muda.DeviceBuffer.from_numpy(a)
    db = muda.DeviceBuffer.from_numpy(b)
    dc = muda.DeviceBuffer.empty_like(c)

    vadd(grid=(n + 255) // 256, block=256, args=[da, db, dc, n], sync=True)
    dc.copy_to(c)

    print("vector_add:", "OK" if np.allclose(c, a + b) else "FAILED")

    mod.unload()
    da.free(); db.free(); dc.free()
    muda.shutdown()


if __name__ == "__main__":
    main()
