import time

import numpy as np
import torch

from examples.kernels import vadd, vadd4
from pycubin import DeviceArray

N = 1 << 26
ITERS = 100


def gbps(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return 3 * 4 * N / ((time.perf_counter() - start) / ITERS) / 1e9


x, y, out = (DeviceArray(np.random.randn(N).astype(np.float32)) for _ in range(3))
a, b = torch.randn(N, device="cuda"), torch.randn(N, device="cuda")
c = torch.empty_like(a)
print(f"pycubin vadd (auto):   {gbps(lambda: vadd[N // 256](x, y, out, N)):.0f} GB/s")
print(f"pycubin vadd4 (manual): {gbps(lambda: vadd4[N // 1024](x, y, out, N)):.0f} GB/s")
print(f"torch.add:     {gbps(lambda: torch.add(a, b, out=c)):.0f} GB/s")
