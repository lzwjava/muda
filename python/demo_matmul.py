#!/usr/bin/env python3
"""Benchmark mudaBLAS SGEMM from Python via the ctypes binding."""
import numpy as np

import muda


def main():
    muda.init()
    print("driver:", muda.driver_name())
    info = muda.device_info()
    print("device:", info["name"], "sm_" + info["compute_capability"],
          f"{info['total_global_mem'] // 1048576} MiB, "
          f"{info['multiprocessors']} SMs")
    print()

    for size in (256, 512, 1024, 2048):
        rng = np.random.default_rng(0)
        A = rng.random((size, size), dtype=np.float32)
        B = rng.random((size, size), dtype=np.float32)

        # Pre-allocate device buffers once so we time the kernel, not the
        # host/device traffic.
        dA = muda.DeviceBuffer.from_numpy(A)
        dB = muda.DeviceBuffer.from_numpy(B)
        dC = muda.DeviceBuffer(size * size * 4)

        muda.gemm_buffers(dA, dB, dC, size, size, size)
        muda.synchronize()
        err = float(np.abs(dC.to_numpy((size, size)) - A @ B).max())

        reps = 20 if size <= 1024 else 8
        t0, t1 = muda.event(), muda.event()
        t0.record()
        for _ in range(reps):
            muda.gemm_buffers(dA, dB, dC, size, size, size)
        t1.record()
        muda.synchronize()
        ms = muda.Event.elapsed_ms(t0, t1) / reps
        gflops = 2.0 * size ** 3 / (ms * 1e-3) / 1e9
        print(f"  {size:5d}^3  {ms:8.3f} ms  {gflops:8.1f} GFLOP/s  max_err={err:.2e}")

        dA.free(); dB.free(); dC.free()

    muda.shutdown()


if __name__ == "__main__":
    main()
