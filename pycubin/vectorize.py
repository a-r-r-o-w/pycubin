import ast
from collections import Counter
from dataclasses import dataclass

LANES = 4

BINOPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}


class Unsupported(Exception):
    pass


@dataclass(frozen=True)
class Const:
    literal: str


@dataclass(frozen=True)
class Arg:
    name: str


@dataclass(frozen=True)
class Idx:
    pass


@dataclass(frozen=True)
class Load:
    ptr: str


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Neg:
    operand: "Expr"


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Expr", ...]


Expr = Const | Arg | Idx | Load | BinOp | Neg | Call


def _children(node: Expr) -> tuple[Expr, ...]:
    if isinstance(node, BinOp):
        return (node.left, node.right)
    if isinstance(node, Neg):
        return (node.operand,)
    if isinstance(node, Call):
        return node.args
    return ()


class _ExprBuilder:
    def __init__(self, idx_name: str, params: dict[str, str]) -> None:
        self.idx_name = idx_name
        self.params = params
        self.env: dict[str, Expr] = {}
        self.stores: dict[str, Expr] = {}

    def _is_elem(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and self.params.get(node.value.id) == "ptr"
            and isinstance(node.slice, ast.Name)
            and node.slice.id == self.idx_name
        )

    def lower_expr(self, node: ast.expr) -> Expr:
        if isinstance(node, ast.Constant):
            return Const(repr(node.value))
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id == self.idx_name:
                return Idx()
            if self.params.get(node.id) == "scalar":
                return Arg(node.id)
        if self._is_elem(node):
            assert isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            return self.stores.get(node.value.id, Load(node.value.id))
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            return BinOp(BINOPS[type(node.op)], self.lower_expr(node.left), self.lower_expr(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return Neg(self.lower_expr(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("float", "fma"):
            return Call(node.func.id, tuple(self.lower_expr(arg) for arg in node.args))
        raise Unsupported

    def lower_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.AugAssign):
            node = ast.Assign(targets=[node.target], value=ast.BinOp(node.target, node.op, node.value))
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            raise Unsupported

        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id not in self.params and target.id != self.idx_name:
            self.env[target.id] = self.lower_expr(node.value)
        elif self._is_elem(target):
            assert isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            self.stores[target.value.id] = self.lower_expr(node.value)
        else:
            raise Unsupported


def _render(node: Expr, lane: int, temps: dict[Expr, int], use_temp: bool = True) -> str:
    if use_temp and node in temps:
        return f"t{temps[node]}_{lane}"
    if isinstance(node, Const):
        return node.literal
    if isinstance(node, Arg):
        return node.name
    if isinstance(node, Idx):
        return f"(_g * {LANES} + {lane})"
    if isinstance(node, Load):
        return f"v_{node.ptr}_{lane}"
    if isinstance(node, BinOp):
        left = _render(node.left, lane, temps)
        right = _render(node.right, lane, temps)
        return f"({left} {node.op} {right})"
    if isinstance(node, Neg):
        return f"(-{_render(node.operand, lane, temps)})"
    args = ", ".join(_render(arg, lane, temps) for arg in node.args)
    return f"{node.name}({args})"


def _collect_cse(roots: list[Expr]) -> tuple[list[Expr], dict[Expr, int]]:
    uses: Counter[Expr] = Counter()
    order: list[Expr] = []

    def walk(node: Expr) -> None:
        uses[node] += 1
        if uses[node] == 1:
            for child in _children(node):
                walk(child)
            order.append(node)

    for root in roots:
        walk(root)

    temps = {node: i for i, node in enumerate(order) if uses[node] > 1 and _children(node)}
    return order, temps


def _match_elementwise(fn: ast.FunctionDef, params: dict[str, str]) -> tuple[str, str, list[ast.stmt]]:
    if len(fn.body) != 2:
        raise Unsupported
    first, guard = fn.body
    if not (
        isinstance(first, ast.Assign)
        and len(first.targets) == 1
        and isinstance(first.targets[0], ast.Name)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "gid"
        and isinstance(guard, ast.If)
        and not guard.orelse
        and isinstance(guard.test, ast.Compare)
        and len(guard.test.ops) == 1
        and isinstance(guard.test.ops[0], ast.Lt)
    ):
        raise Unsupported

    idx_name = first.targets[0].id
    limit_node = guard.test.comparators[0]
    limit_name = limit_node.id if isinstance(limit_node, ast.Name) else None
    if not (
        isinstance(guard.test.left, ast.Name)
        and guard.test.left.id == idx_name
        and limit_name is not None
        and params.get(limit_name) == "scalar"
    ):
        raise Unsupported

    return idx_name, limit_name, guard.body


def vectorize(fn: ast.FunctionDef) -> tuple[ast.FunctionDef, str]:
    params = {
        arg.arg: "ptr" if isinstance(arg.annotation, ast.Name) and arg.annotation.id.startswith("ptr") else "scalar"
        for arg in fn.args.args
    }
    idx_name, limit_name, body = _match_elementwise(fn, params)

    builder = _ExprBuilder(idx_name, params)
    for stmt in body:
        builder.lower_stmt(stmt)
    if not builder.stores:
        raise Unsupported

    order, temps = _collect_cse(list(builder.stores.values()))
    lanes = range(LANES)

    vector_lines: list[str] = []
    for ptr in sorted({node.ptr for node in order if isinstance(node, Load)}):
        targets = ", ".join(f"v_{ptr}_{lane}" for lane in lanes)
        vector_lines.append(f"{targets} = load4({ptr}, _g)")

    for node, temp_id in temps.items():
        for lane in lanes:
            vector_lines.append(f"t{temp_id}_{lane} = {_render(node, lane, temps, use_temp=False)}")

    for ptr, expr in builder.stores.items():
        values = ", ".join(_render(expr, lane, temps) for lane in lanes)
        vector_lines.append(f"store4({ptr}, _g, {values})")

    scalar_lines = [ast.unparse(stmt) for stmt in body]
    lines = [
        f"def {fn.name}({ast.unparse(fn.args)}):",
        "    _g = gid()",
        f"    if _g < ({limit_name} >> 2):",
        *[f"        {line}" for line in vector_lines],
        "    else:",
        f"        {idx_name} = _g << 2",
        f"        while {idx_name} < {limit_name}:",
        *[f"            {line}" for line in scalar_lines],
        f"            {idx_name} = {idx_name} + 1",
    ]
    parsed = ast.parse("\n".join(lines)).body[0]
    assert isinstance(parsed, ast.FunctionDef)
    return parsed, limit_name
