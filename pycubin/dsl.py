import ast
import ctypes
import inspect
import struct
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from . import elf, runtime
from . import sass as S
from .vectorize import Unsupported, vectorize

f32 = "f32"
i32 = "i32"
ptr_f32 = ("ptr", f32)
ptr_i32 = ("ptr", i32)

DType = str | tuple[str, str]

TYPES: dict[str, DType] = {
    "f32": f32,
    "i32": i32,
    "ptr_f32": ptr_f32,
    "ptr_i32": ptr_i32,
}
SHARED_TYPES: dict[str, str] = {"shared_f32": f32, "shared_i32": i32}
SPECIAL_CALLS = ("gid", "tid", "bid")
CMP_OPS: dict[type[ast.cmpop], str] = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}
ARITH_OPS = (ast.Add, ast.Sub, ast.Mult)
BIT_LUTS: dict[type[ast.operator], int] = {
    ast.BitAnd: 0xC0,
    ast.BitOr: 0xFC,
    ast.BitXor: 0x3C,
}


def device_only(*args: Any) -> Any:
    raise RuntimeError("device-only")


gid = device_only
tid = device_only
bid = device_only
fma = device_only
sync = device_only
shfl_xor = device_only
load4 = device_only
store4 = device_only
shared_f32 = device_only
shared_i32 = device_only


class CompileError(Exception):
    pass


@dataclass
class Param:
    dtype: DType
    offset: int
    size: int
    reg: int | None = None


@dataclass
class Var:
    reg: int
    dtype: DType


@dataclass
class SharedArray:
    offset: int
    dtype: str


def _is_call(node: ast.AST, *names: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names


class RegAlloc:
    def __init__(self) -> None:
        self.next_reg = 0
        self.temps: list[tuple[int, int]] = []
        self.free: dict[int, list[int]] = defaultdict(list)
        self.gaps: list[int] = []

    def _take_free(self, width: int) -> int | None:
        for size in (1, 2, 4):
            if size >= width and self.free[size]:
                reg = self.free[size].pop()
                self.free[width].extend(range(reg + width, reg + size, width))
                return reg
        return None

    def alloc(self, width: int = 1, temp: bool = True) -> int:
        reg = self._take_free(width) if temp else None
        if reg is None and width == 1 and self.gaps:
            reg = self.gaps.pop()
        if reg is None:
            while self.next_reg % width:
                self.gaps.append(self.next_reg)
                self.next_reg += 1
            reg = self.next_reg
            self.next_reg += width
            if self.next_reg > 250:
                raise CompileError("out of registers")
        if temp:
            self.temps.append((reg, width))
        return reg

    def mark(self) -> int:
        return len(self.temps)

    def release_temps(self, mark: int) -> None:
        for reg, width in self.temps[mark:]:
            self.free[width].append(reg)
        del self.temps[mark:]

    def release_reg(self, reg: int, width: int = 1) -> None:
        self.free[width].append(reg)


class Compiler:
    def __init__(self, fn: ast.FunctionDef, block: int) -> None:
        self.block = block
        self.regs = RegAlloc()
        self.prologue: list[S.Item] = []
        self.body: list[S.Item] = []

        self.vars: dict[str, Var] = {}
        self.consts: dict[tuple[int, str], int] = {}
        self.specials: dict[str, int] = {}
        self.shared: dict[str, SharedArray] = {}
        self.params: dict[str, Param] = self._parse_params(fn)
        self.param_sizes = [p.size for p in self.params.values()]

        self.pred_cursor = 0
        self.label_count = 0
        self.barrier_depth = 0
        self.smem = 0
        self.uses_global_desc = False

        self._compile_stmts(fn.body)
        self.body.append(S.EXIT())
        if self.uses_global_desc:
            self.prologue.insert(0, S.LDCU64(4, 0x358))

        self.items = self.prologue + self.body
        used_regs = [
            reg
            for item in self.items
            if isinstance(item, S.Insn)
            for reg in item.reads + item.writes
            if reg < 1000
        ]
        self.maxreg = max(used_regs, default=0)

    def _parse_params(self, fn: ast.FunctionDef) -> dict[str, Param]:
        params: dict[str, Param] = {}
        offset = 0
        for arg in fn.args.args:
            ann = arg.annotation
            if not (isinstance(ann, ast.Name) and ann.id in TYPES):
                raise CompileError(f"parameter {arg.arg} needs an f32/i32/ptr_f32/ptr_i32 annotation")
            dtype = TYPES[ann.id]
            size = 8 if isinstance(dtype, tuple) else 4
            offset = (offset + size - 1) // size * size
            params[arg.arg] = Param(dtype=dtype, offset=offset, size=size)
            offset += size
        return params

    def _next_pred(self) -> int:
        pred = self.pred_cursor
        self.pred_cursor = (self.pred_cursor + 1) % 6
        return pred

    def _next_label(self) -> str:
        self.label_count += 1
        return f"L{self.label_count}"

    def _push_barrier(self) -> int:
        if self.barrier_depth == 16:
            raise CompileError("control flow nested too deeply")
        barrier = self.barrier_depth
        self.barrier_depth += 1
        return barrier

    def _emit(self, *items: S.Item) -> None:
        self.body.extend(items)

    def _dest(self, dest: int | None) -> int:
        return self.regs.alloc() if dest is None else dest

    def _move(self, reg: int, dest: int | None) -> int:
        if dest is None:
            return reg
        self._emit(S.IADD3(dest, reg, S.RZ))
        return dest

    def _const(self, value: float | int, dtype: str) -> int:
        bits = struct.unpack("<I", struct.pack("<f", value))[0] if dtype == f32 else int(value) & 0xFFFFFFFF
        if bits == 0:
            return S.RZ
        key = (bits, dtype)
        if key not in self.consts:
            reg = self.regs.alloc(1, temp=False)
            self.prologue.append(S.MOV_I(reg, bits))
            self.consts[key] = reg
        return self.consts[key]

    def _special_reg(self, name: str) -> int:
        if name not in self.specials:
            reg = self.regs.alloc(1, temp=False)
            if name == "gid":
                self.prologue.append(
                    S.IMAD(reg, self._special_reg("bid"), self._const(self.block, i32), self._special_reg("tid"))
                )
            else:
                self.prologue.append(S.S2R(reg, name))
            self.specials[name] = reg
        return self.specials[name]

    def _param_reg(self, name: str) -> int:
        param = self.params[name]
        if param.reg is None:
            wide = isinstance(param.dtype, tuple)
            param.reg = self.regs.alloc(2 if wide else 1, temp=False)
            self.prologue.append(S.LDC(param.reg, elf.PARAM_BASE + param.offset, wide))
        return param.reg

    def _var_reg(self, name: str, dtype: DType) -> int:
        if name in self.params or name in self.shared:
            raise CompileError(f"cannot assign to {name}")
        if name not in self.vars:
            self.vars[name] = Var(reg=self.regs.alloc(1, temp=False), dtype=dtype)
        elif self.vars[name].dtype != dtype:
            raise CompileError(f"{name} changes type {self.vars[name].dtype} -> {dtype}")
        return self.vars[name].reg

    def _is_shared_subscript(self, node: ast.Subscript) -> bool:
        return isinstance(node.value, ast.Name) and node.value.id in self.shared

    def _infer_type(self, node: ast.expr) -> DType:
        if isinstance(node, ast.Constant):
            return f32 if isinstance(node.value, float) else i32
        if isinstance(node, ast.Name):
            if node.id in self.vars:
                return self.vars[node.id].dtype
            if node.id in self.params:
                return self.params[node.id].dtype
            raise CompileError(f"undefined name {node.id}")
        if isinstance(node, ast.UnaryOp):
            return self._infer_type(node.operand)
        if isinstance(node, ast.BinOp):
            left = self._infer_type(node.left)
            right = self._infer_type(node.right)
            if isinstance(node.left, ast.Constant) and right == f32:
                return f32
            if isinstance(node.right, ast.Constant) and left == f32:
                return f32
            if left != right:
                raise CompileError(f"mixed types {left}/{right}; use float()")
            return left
        if _is_call(node, *SPECIAL_CALLS):
            return i32
        if _is_call(node, "float", "fma"):
            return f32
        if _is_call(node, "shfl_xor"):
            assert isinstance(node, ast.Call)
            return self._infer_type(node.args[0])
        if isinstance(node, ast.Subscript):
            if self._is_shared_subscript(node):
                assert isinstance(node.value, ast.Name)
                return self.shared[node.value.id].dtype
            dtype = self._infer_type(node.value)
            if isinstance(dtype, tuple):
                return dtype[1]
        raise CompileError(f"unsupported expression {ast.dump(node)[:60]}")

    def _compile_expr(self, node: ast.expr, dest: int | None = None, want: DType | None = None) -> int:
        if isinstance(node, ast.Constant):
            dtype = f32 if isinstance(node.value, float) or want == f32 else i32
            val = float(node.value) if dtype == f32 else int(node.value)
            return self._move(self._const(val, dtype), dest)
        if isinstance(node, ast.Name):
            reg = self.vars[node.id].reg if node.id in self.vars else self._param_reg(node.id)
            return self._move(reg, dest)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._compile_expr(node.operand, dest)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            dtype = self._infer_type(node.operand)
            operand = self._compile_expr(node.operand)
            out = self._dest(dest)
            if dtype == f32:
                self._emit(S.FMUL(out, operand, self._const(-1.0, f32)))
            else:
                self._emit(S.IADD3(out, S.RZ, operand, neg_b=True))
            return out
        if isinstance(node, ast.BinOp):
            return self._compile_binop(node, dest)
        if isinstance(node, ast.Call):
            return self._compile_call(node, dest)
        if isinstance(node, ast.Subscript):
            out = self._dest(dest)
            if self._is_shared_subscript(node):
                addr, offset = self._shared_addr(node)
                self._emit(S.LDS(out, addr, offset))
            else:
                self._emit(S.LDG(out, self._global_addr(node)))
            return out
        raise CompileError(f"unsupported expression {ast.dump(node)[:60]}")

    def _compile_call(self, node: ast.Call, dest: int | None) -> int:
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        if name in SPECIAL_CALLS:
            return self._move(self._special_reg(name), dest)
        if name == "float":
            src = self._compile_expr(node.args[0])
            out = self._dest(dest)
            self._emit(S.I2F(out, src))
            return out
        if name == "fma":
            a = self._compile_expr(node.args[0], want=f32)
            b = self._compile_expr(node.args[1], want=f32)
            c = self._compile_expr(node.args[2], want=f32)
            out = self._dest(dest)
            self._emit(S.FFMA(out, a, b, c))
            return out
        if name == "shfl_xor":
            src = self._compile_expr(node.args[0])
            out = self._dest(dest)
            assert isinstance(node.args[1], ast.Constant)
            self._emit(S.SHFL(out, src, int(node.args[1].value)))
            return out
        raise CompileError(f"unsupported call {name}")

    def _compile_binop(self, node: ast.BinOp, dest: int | None) -> int:
        dtype = self._infer_type(node)
        op = type(node.op)
        if op not in ARITH_OPS and (dtype == f32 or op not in (*BIT_LUTS, ast.LShift, ast.RShift)):
            raise CompileError(f"unsupported operator {op.__name__}")

        if dtype == f32 and op is ast.Add:
            for mul_side, add_side in ((node.left, node.right), (node.right, node.left)):
                if isinstance(mul_side, ast.BinOp) and isinstance(mul_side.op, ast.Mult) and self._infer_type(mul_side) == f32:
                    a = self._compile_expr(mul_side.left, want=f32)
                    b = self._compile_expr(mul_side.right, want=f32)
                    c = self._compile_expr(add_side, want=f32)
                    out = self._dest(dest)
                    self._emit(S.FFMA(out, a, b, c))
                    return out

        left = self._compile_expr(node.left, want=dtype)
        right = self._compile_expr(node.right, want=dtype)
        out = self._dest(dest)

        if dtype == f32:
            if op is ast.Mult:
                self._emit(S.FMUL(out, left, right))
            else:
                self._emit(S.FADD(out, left, right, neg_b=op is ast.Sub))
        elif op in BIT_LUTS:
            self._emit(S.LOP3(out, left, right, BIT_LUTS[op]))
        elif op is ast.LShift:
            self._emit(S.SHL(out, left, right))
        elif op is ast.RShift:
            self._emit(S.SHR(out, left, right))
        elif op is ast.Mult:
            self._emit(S.IMAD(out, left, right))
        else:
            self._emit(S.IADD3(out, left, right, neg_b=op is ast.Sub))
        return out

    def _global_addr(self, node: ast.Subscript, scale: int = 4) -> int:
        name = node.value.id if isinstance(node.value, ast.Name) else None
        if name not in self.params or not isinstance(self.params[name].dtype, tuple):
            raise CompileError("only parameter pointers can be indexed")
        self.uses_global_desc = True
        base = self._param_reg(name)
        idx = self._compile_expr(node.slice)
        out = self.regs.alloc(2)
        self._emit(S.IMAD_WIDE(out, idx, scale, base))
        return out

    def _shared_addr(self, node: ast.Subscript) -> tuple[int, int]:
        assert isinstance(node.value, ast.Name)
        base = self.shared[node.value.id].offset
        out = self.regs.alloc()
        self._emit(S.IMAD(out, self._compile_expr(node.slice), self._const(4, i32)))
        return out, elf.SMEM_BASE + base

    def _compile_cond(self, node: ast.expr) -> int:
        pred = self._next_pred()
        if isinstance(node, ast.BoolOp):
            bop = "AND" if isinstance(node.op, ast.And) else "OR"
            for i, value in enumerate(node.values):
                self._compile_compare(value, pred, bop, (pred, 0) if i else None)
        else:
            self._compile_compare(node, pred, "AND", None)
        return pred

    def _compile_compare(self, node: ast.expr, pred: int, bop: str, src: S.Guard | None) -> None:
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            raise CompileError("conditions must be comparisons joined by and/or")
        left_node = node.left
        right_node = node.comparators[0]
        dtype = f32 if f32 in (self._infer_type(left_node), self._infer_type(right_node)) else i32
        left = self._compile_expr(left_node, want=dtype)
        right = self._compile_expr(right_node, want=dtype)
        cmp_op = CMP_OPS[type(node.ops[0])]
        insn = S.FSETP(cmp_op, pred, left, right, bop, src) if dtype == f32 else S.ISETP(cmp_op, pred, left, right, bop, src)
        self._emit(insn)

    def _compile_stmts(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            mark = self.regs.mark()
            self._compile_stmt(stmt)
            self.regs.release_temps(mark)

    def _compile_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise CompileError("single assignment target only")
            self._compile_assign(node.targets[0], node.value)
        elif isinstance(node, ast.AugAssign):
            target = node.target
            rhs_target: ast.expr = (
                ast.Subscript(target.value, target.slice, ast.Load()) if isinstance(target, ast.Subscript) else target
            )
            self._compile_assign(target, ast.BinOp(rhs_target, node.op, node.value))
        elif isinstance(node, ast.If):
            self._compile_if(node)
        elif isinstance(node, ast.For):
            self._compile_for(node)
        elif isinstance(node, ast.While):
            self._compile_while(node)
        elif isinstance(node, ast.Return):
            if node.value is not None:
                raise CompileError("kernels cannot return a value")
            self._emit(S.EXIT())
        elif isinstance(node, ast.Expr) and _is_call(node.value, "sync"):
            self._emit(S.BAR())
        elif isinstance(node, ast.Expr) and _is_call(node.value, "store4"):
            assert isinstance(node.value, ast.Call)
            self._compile_store4(node.value.args)
        elif isinstance(node, ast.Pass) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            pass
        else:
            raise CompileError(f"unsupported statement {type(node).__name__} at line {node.lineno}")

    def _compile_if(self, node: ast.If) -> None:
        pred = self._compile_cond(node.test)
        else_label = self._next_label()
        end_label = self._next_label()
        barrier = self._push_barrier()

        self._emit(S.BSSY(barrier, end_label), S.BRA(else_label, (pred, 1)))
        vars_before = set(self.vars)
        self._compile_stmts(node.body)

        if node.orelse:
            self._emit(S.BRA(end_label))
            else_names = {
                x.id
                for x in ast.walk(ast.Module(body=node.orelse, type_ignores=[]))
                if isinstance(x, ast.Name)
            }
            for name in list(self.vars):
                if name not in vars_before and name not in else_names:
                    var = self.vars.pop(name)
                    self.regs.release_reg(var.reg)

        self._emit(S.Label(else_label))
        self._compile_stmts(node.orelse)
        self._emit(S.Label(end_label), S.BSYNC(barrier))
        self.barrier_depth -= 1

    def _compile_for(self, node: ast.For) -> None:
        if not (_is_call(node.iter, "range") and isinstance(node.target, ast.Name)):
            raise CompileError("only `for i in range(...)`")
        assert isinstance(node.iter, ast.Call)
        args = node.iter.args
        start, stop = (args[0], args[1]) if len(args) > 1 else (ast.Constant(0), args[0])
        step_node = args[2] if len(args) > 2 else None
        step = step_node.value if isinstance(step_node, ast.Constant) else 1
        if not isinstance(step, int) or step == 0:
            raise CompileError("range step must be a nonzero int literal")

        idx_reg = self._var_reg(node.target.id, i32)
        self._compile_expr(start, dest=idx_reg)
        limit_reg = self.regs.alloc(1, temp=False)
        self._compile_expr(stop, dest=limit_reg)

        top_label = self._next_label()
        end_label = self._next_label()
        barrier = self._push_barrier()
        pred = self._next_pred()

        self._emit(S.BSSY(barrier, end_label), S.Label(top_label))
        self._emit(S.ISETP("lt" if step > 0 else "gt", pred, idx_reg, limit_reg), S.BRA(end_label, (pred, 1)))
        self._compile_stmts(node.body)
        self._emit(
            S.IADD3(idx_reg, idx_reg, self._const(step, i32)),
            S.BRA(top_label),
            S.Label(end_label),
            S.BSYNC(barrier),
        )
        self.barrier_depth -= 1

    def _compile_while(self, node: ast.While) -> None:
        top_label = self._next_label()
        end_label = self._next_label()
        barrier = self._push_barrier()

        self._emit(S.BSSY(barrier, end_label), S.Label(top_label))
        self._emit(S.BRA(end_label, (self._compile_cond(node.test), 1)))
        self._compile_stmts(node.body)
        self._emit(S.BRA(top_label), S.Label(end_label), S.BSYNC(barrier))
        self.barrier_depth -= 1

    def _compile_assign(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name) and _is_call(value, *SHARED_TYPES):
            assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            assert isinstance(value.args[0], ast.Constant)
            self.shared[target.id] = SharedArray(offset=self.smem, dtype=SHARED_TYPES[value.func.id])
            self.smem += 4 * int(value.args[0].value)
        elif isinstance(target, ast.Name):
            dtype = self._infer_type(value)
            self._compile_expr(value, dest=self._var_reg(target.id, dtype), want=dtype)
        elif isinstance(target, ast.Tuple):
            self._compile_load4(target, value)
        elif isinstance(target, ast.Subscript) and self._is_shared_subscript(target):
            val = self._compile_expr(value, want=self._infer_type(target))
            addr, offset = self._shared_addr(target)
            self._emit(S.STS(addr, val, offset))
        elif isinstance(target, ast.Subscript):
            val = self._compile_expr(value, want=self._infer_type(target))
            self._emit(S.STG(self._global_addr(target), val))
        else:
            raise CompileError("unsupported assignment target")

    def _compile_load4(self, target: ast.Tuple, value: ast.expr) -> None:
        if not (_is_call(value, "load4") and len(target.elts) == 4):
            raise CompileError("tuple assignment must be `a, b, c, d = load4(ptr, i)`")
        assert isinstance(value, ast.Call)
        ptr, idx = value.args
        elem_type = self._infer_type(ptr)
        assert isinstance(elem_type, tuple)
        base = self.regs.alloc(4, temp=False)
        for lane, elt in enumerate(target.elts):
            assert isinstance(elt, ast.Name)
            if elt.id in self.vars:
                raise CompileError(f"{elt.id} is already defined")
            self.vars[elt.id] = Var(reg=base + lane, dtype=elem_type[1])
        self._emit(S.LDG(base, self._global_addr(ast.Subscript(ptr, idx), scale=16), 4))

    def _compile_store4(self, args: list[ast.expr]) -> None:
        ptr, idx, *values = args
        if len(values) != 4:
            raise CompileError("store4(ptr, i, a, b, c, d)")
        elem_type = self._infer_type(ptr)
        assert isinstance(elem_type, tuple)
        dtype = elem_type[1]
        existing = [
            self.vars[v.id].reg if isinstance(v, ast.Name) and v.id in self.vars else None
            for v in values
        ]
        if existing[0] is not None and existing[0] % 4 == 0 and existing == list(range(existing[0], existing[0] + 4)):
            base = existing[0]
        else:
            base = self.regs.alloc(4)
            for lane, val in enumerate(values):
                self._compile_expr(val, dest=base + lane, want=dtype)
        self._emit(S.STG(self._global_addr(ast.Subscript(ptr, idx), scale=16), base, 4))


class Program:
    def __init__(self, node: ast.FunctionDef, block: int) -> None:
        compiled = Compiler(node, block)
        self.name = node.name
        self.block = block
        self.types = [p.dtype for p in compiled.params.values()]
        self.items = compiled.items
        self.nregs = compiled.maxreg + 3
        self.text = S.encode(compiled.items)
        self.cubin = elf.make_cubin(self.name, self.text, compiled.param_sizes, self.nregs, compiled.smem)
        self.module: runtime.Module | None = None

    def ensure_module(self) -> runtime.Module:
        if self.module is None:
            self.module = runtime.Module(self.cubin, self.name)
        return self.module

    def launch(self, grid: int, vals: list[Any]) -> None:
        self.ensure_module().launch(grid, self.block, vals)


def _ctype_for(dtype: DType) -> Any:
    if isinstance(dtype, tuple):
        return ctypes.c_uint64()
    return ctypes.c_float() if dtype == f32 else ctypes.c_int32()


class Kernel:
    def __init__(self, fn: Callable[..., Any], block: int = 256) -> None:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        assert isinstance(node, ast.FunctionDef)
        self.scalar = Program(node, block)
        self.block = block
        self.name = self.scalar.name
        self.types = self.scalar.types
        self.cubin = self.scalar.cubin
        self.text = self.scalar.text

        self._slots = tuple(_ctype_for(t) for t in self.types)
        self._ptrs = (ctypes.c_void_p * len(self._slots))(*(ctypes.addressof(s) for s in self._slots))
        self._is_ptr = tuple(isinstance(t, tuple) for t in self.types)
        self._grid = [0]
        self._launcher: Callable[..., None] | None = None

        self.vector: Program | None = None
        self.n_arg: int | None = None
        try:
            vec_node, limit_name = vectorize(node)
            self.vector = Program(vec_node, block)
            self.n_arg = [arg.arg for arg in node.args.args].index(limit_name)
        except Unsupported:
            pass

    def _build_launcher(self) -> Callable[..., None]:
        scalar_fn = self.scalar.ensure_module().fn
        vector_fn = self.vector.ensure_module().fn if self.vector is not None else None
        arg_names = ", ".join(f"a{i}" for i in range(len(self._slots)))
        ns: dict[str, Any] = {
            "_launch": runtime._launch_kernel,
            "_check": runtime.check,
            "_scalar": scalar_fn,
            "_vector": vector_fn,
            "_block": self.block,
            "_ptrs": self._ptrs,
            "_grid": self._grid,
        }

        lines = ["    grid = _grid[0]"]
        ptr_vars: list[str] = []
        for i, (slot, is_ptr) in enumerate(zip(self._slots, self._is_ptr)):
            ns[f"_slot{i}"] = slot
            if is_ptr:
                lines.append(f"    ptr{i} = a{i}.data_ptr(); _slot{i}.value = ptr{i}")
                ptr_vars.append(f"ptr{i}")
            else:
                lines.append(f"    _slot{i}.value = a{i}")

        if vector_fn is not None and ptr_vars:
            aligned = f"(({' | '.join(ptr_vars)}) & 15) == 0"
            lines.append(
                f"    status = _launch(_vector, (grid + 3) >> 2, 1, 1, _block, 1, 1, 0, None, _ptrs, None)"
                f" if {aligned} else _launch(_scalar, grid, 1, 1, _block, 1, 1, 0, None, _ptrs, None)"
            )
        else:
            lines.append("    status = _launch(_scalar, grid, 1, 1, _block, 1, 1, 0, None, _ptrs, None)")
        lines.append("    if status: _check(status)")

        exec(f"def _launch_bound({arg_names}):\n" + "\n".join(lines), ns)
        return ns["_launch_bound"]

    def __getitem__(self, grid: int) -> Callable[..., None]:
        if self._launcher is None:
            self._launcher = self._build_launcher()
        self._grid[0] = grid
        return self._launcher

    def launch(self, grid: int, *args: Any) -> None:
        self[grid](*args)


def jit(fn: Callable[..., Any] | None = None, *, block: int = 256) -> Any:
    return (lambda f: Kernel(f, block)) if fn is None else Kernel(fn, block)
