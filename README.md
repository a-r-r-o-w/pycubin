# pycubin

Minimal Python-to-SASS toy compiler for NVIDIA Blackwell (`sm_100a`). Compiles a small subset of Python directly to a `cubin` binary and runs it without `nvcc`, `ptxas`, NVVM, or NVIDIA Python packages at runtime, using only the CUDA driver.

**Blog post:** [Compiling Python to SASS](https://a-r-r-o-w.github.io/blog/00006_compiling_python_to_sass/)

```python
from pycubin import DeviceArray, f32, gid, i32, jit, ptr_f32, synchronize

@jit
def saxpy(x: ptr_f32, y: ptr_f32, out: ptr_f32, a: f32, n: i32):
    i = gid()
    if i < n:
        out[i] = a * x[i] + y[i]

grid = (n + 255) // 256
saxpy[grid](dx, dy, dout, 2.5, n)
synchronize()
```

Supported subset: `f32` / `i32` scalars and pointers, `+ - *`, `& | ^ << >>`, `if/elif/else`, `for` over `range`, `while`, `return`, `fma`, `float`, `gid/tid/bid`, `load4/store4`, shared memory, `sync`, `shfl_xor`, and automatic 128-bit vectorization for simple elementwise kernels.

## Setup

```bash
git clone https://github.com/a-r-r-o-w/pycubin.git
cd pycubin
uv venv pycubin-venv
source pycubin-venv/bin/activate
uv pip install -e .
```

See [Examples](./examples) for usage.
