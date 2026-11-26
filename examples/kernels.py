from pycubin import (
    bid,
    f32,
    gid,
    i32,
    jit,
    load4,
    ptr_f32,
    ptr_i32,
    shared_f32,
    shared_i32,
    shfl_xor,
    store4,
    sync,
    tid,
)


@jit
def vadd(x: ptr_f32, y: ptr_f32, out: ptr_f32, n: i32):
    i = gid()
    if i < n:
        out[i] = x[i] + y[i]


@jit
def vadd4(x: ptr_f32, y: ptr_f32, out: ptr_f32, n: i32):
    i = gid()
    if i * 4 < n:
        a, b, c, d = load4(x, i)
        e, f, g, h = load4(y, i)
        store4(out, i, a + e, b + f, c + g, d + h)


@jit
def saxpy(x: ptr_f32, y: ptr_f32, out: ptr_f32, a: f32, n: i32):
    i = gid()
    if i < n:
        out[i] = a * x[i] + y[i]


@jit
def clamp_relu(x: ptr_f32, out: ptr_f32, lo: f32, hi: f32, n: i32):
    i = gid()
    if i >= n:
        return
    v = x[i]
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    out[i] = v


@jit
def horner(x: ptr_f32, c: ptr_f32, out: ptr_f32, deg: i32, n: i32):
    i = gid()
    if i < n:
        xv = x[i]
        acc = c[0]
        for k in range(1, deg):
            acc = acc * xv + c[k]
        out[i] = acc


@jit
def gather_scale(src: ptr_f32, idx: ptr_i32, out: ptr_f32, n: i32):
    i = gid()
    if i < n and idx[i] >= 0:
        out[i] = src[idx[i]] * 2.0


@jit
def tri_sum(x: ptr_i32, out: ptr_i32, n: i32):
    i = gid()
    if i < n:
        v = x[i]
        s = 0
        while v > 0 and s < 100000:
            s += v
            v -= 1
        out[i] = s


@jit
def bit_mix(x: ptr_i32, out: ptr_i32, n: i32):
    i = gid()
    if i < n:
        v = x[i]
        out[i] = ((v << 3) ^ (v >> 2)) & 0xFFFF | (v & 7)


@jit
def reverse_blocks(x: ptr_i32, out: ptr_i32, n: i32):
    buf = shared_i32(256)
    i = gid()
    buf[tid()] = x[i]
    sync()
    out[i] = buf[255 - tid()]


@jit
def block_sum(x: ptr_f32, out: ptr_f32, n: i32):
    buf = shared_f32(8)
    i = gid()
    v = 0.0
    if i < n:
        v = x[i]
    v += shfl_xor(v, 16)
    v += shfl_xor(v, 8)
    v += shfl_xor(v, 4)
    v += shfl_xor(v, 2)
    v += shfl_xor(v, 1)
    lane = tid() & 31
    if lane == 0:
        buf[tid() >> 5] = v
    sync()
    v = 0.0
    if lane < 8:
        v = buf[lane]
    v += shfl_xor(v, 4)
    v += shfl_xor(v, 2)
    v += shfl_xor(v, 1)
    if tid() == 0:
        out[bid()] = v
