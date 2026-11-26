import ctypes
from typing import Any

import numpy as np

_LAUNCH_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_cu: ctypes.CDLL | None = None
_launch_kernel: Any = None
_ctx: ctypes.c_void_p | None = None


def check(result: int) -> None:
    if result:
        msg = ctypes.c_char_p()
        assert _cu is not None
        _cu.cuGetErrorString(result, ctypes.byref(msg))
        detail = msg.value.decode() if msg.value else ""
        raise RuntimeError(f"CUDA error {result}: {detail}")


def init() -> None:
    global _cu, _launch_kernel, _ctx
    if _ctx is not None:
        return
    _cu = ctypes.CDLL("libcuda.so.1")
    _cu.cuLaunchKernel.argtypes = _LAUNCH_ARGTYPES
    _cu.cuLaunchKernel.restype = ctypes.c_int

    _launch_kernel = ctypes.PyDLL("libcuda.so.1").cuLaunchKernel
    _launch_kernel.argtypes = _LAUNCH_ARGTYPES
    _launch_kernel.restype = ctypes.c_int

    check(_cu.cuInit(0))
    dev = ctypes.c_int()
    check(_cu.cuDeviceGet(ctypes.byref(dev), 0))
    _ctx = ctypes.c_void_p()
    check(_cu.cuDevicePrimaryCtxRetain(ctypes.byref(_ctx), dev))
    check(_cu.cuCtxSetCurrent(_ctx))


def launch_raw(fn: ctypes.c_void_p, grid: int, block: int, ptrs: Any) -> None:
    check(_launch_kernel(fn, grid, 1, 1, block, 1, 1, 0, None, ptrs, None))


class DeviceArray:
    def __init__(self, host: np.ndarray) -> None:
        init()
        assert _cu is not None
        self.shape = host.shape
        self.dtype = host.dtype
        self.nbytes = host.nbytes
        self.ptr = ctypes.c_uint64()
        check(_cu.cuMemAlloc_v2(ctypes.byref(self.ptr), max(self.nbytes, 4)))
        self._ptr = self.ptr.value
        src = np.ascontiguousarray(host).ctypes.data_as(ctypes.c_void_p)
        check(_cu.cuMemcpyHtoD_v2(self.ptr, src, self.nbytes))

    def data_ptr(self) -> int:
        return self._ptr

    def numpy(self) -> np.ndarray:
        assert _cu is not None
        out = np.empty(self.shape, self.dtype)
        check(_cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), self.ptr, self.nbytes))
        return out


class Module:
    def __init__(self, cubin: bytes, name: str) -> None:
        init()
        assert _cu is not None
        self.blob = ctypes.create_string_buffer(cubin, len(cubin))
        module = ctypes.c_void_p()
        check(_cu.cuModuleLoadData(ctypes.byref(module), self.blob))
        self.fn = ctypes.c_void_p()
        check(_cu.cuModuleGetFunction(ctypes.byref(self.fn), module, name.encode()))

    def launch(self, grid: int, block: int, args: list[Any]) -> None:
        ptrs = [ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        arr = (ctypes.c_void_p * len(args))(*ptrs)
        launch_raw(self.fn, grid, block, arr)


def synchronize() -> None:
    init()
    assert _cu is not None
    check(_cu.cuCtxSynchronize())
