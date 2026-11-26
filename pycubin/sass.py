from dataclasses import dataclass

RAW_TEMPLATES: dict[str, str] = {
    "LDC32": "000fe200000008000000df00ff017b82",
    "LDC64": "000e220000000a000000e000ff027b82",
    "LDCU64": "000e620008000a0000006b00ff0477ac",
    "S2R": "000e2200000021000000000000077919",
    "IMAD_WIDE": "001fca00078e02020000000407027825",
    "IMAD": "000fc600078e02ff0000000607007224",
    "IADD3": "000fc60007ffe0ff0000000607067210",
    "MOV_I": "000fe200078e00ff12345678ff0d7424",
    "FADD": "000fe400000001000000000908097221",
    "FMUL": "000fc800004000000000000900087220",
    "FFMA": "000fe200000000000000000809097223",
    "I2F": "000fe2000020140000000000000b7245",
    "LOP3": "000fc800078efcff0000000706057212",
    "SHL": "000fc800000006ff0000000706057219",
    "SHR": "002fe2000001160600000007ff047219",
    "ISETP": "0c8fe40003f06270000000000500720c",
    "FSETP": "000fc80003f0e000000000050000720b",
    "LDG": "002ea2000c1e19000000000402007981",
    "STG": "000fe2000c1019040000000d02007986",
    "LDS": "000fe80000000800000000000c087984",
    "STS": "000fe80000000800000000090c007388",
    "SHFL": "000e2200000e00000e001f0004097f89",
    "BAR": "000fec00000100000000000000007b1d",
    "BSSY": "000fe200038002000000008000007945",
    "BSYNC": "001fea00038002000000000000007941",
    "BRA": "000fc0000383fffffffffffc00fc7947",
    "EXIT": "000fea0003800000000000000000794d",
}

TEMPLATES: dict[str, int] = {name: int(word, 16) & ((1 << 105) - 1) for name, word in RAW_TEMPLATES.items()}
TEMPLATES["FADD"] &= ~(1 << 72)
TEMPLATES["EXIT"] |= 7 << 12
TEMPLATES["LOP3"] &= ~(0xFF << 72)
TEMPLATES["SHR"] ^= 1 << 73

SPECIAL_REGS = {"tid": 0x21, "bid": 0x25}
ICMP_CODES = {"lt": 1, "eq": 2, "le": 3, "gt": 4, "ne": 5, "ge": 6}
FCMP_CODES = {"lt": 1, "eq": 2, "le": 3, "gt": 4, "ne": 13, "ge": 6}
BOOL_OPS = {"AND": 0, "OR": 1}
ACCESS_WIDTHS = {1: 4, 2: 5, 4: 6}

RZ = 255
PT = 7
UR4 = 1004

ALU_LATENCY = 5
GUARD_LATENCY = 13
BRANCH_DELAY = 15
ALL_SLOTS_MASK = 0x3F
LDG_SLOT = 4
MEM_READ_SLOT = 5

OP_LDG = 0x981
OP_BSSY = 0x945
NEG_SRC_B = 1 << 63

Guard = tuple[int, int]
RegKey = tuple[str, int]


@dataclass
class Insn:
    word: int
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    pred_reads: tuple[int, ...] = ()
    pred_writes: tuple[int, ...] = ()
    kind: str = "alu"
    target: str | None = None
    delay: int = 1
    out_slot: int = 7
    in_slot: int = 7
    wait_mask: int = 0

    @property
    def opcode(self) -> int:
        return self.word & 0xFFF


@dataclass
class Label:
    name: str


Item = Insn | Label


def pack(word: int, *fields: tuple[int, int, int]) -> int:
    for lo, width, value in fields:
        mask = ((1 << width) - 1) << lo
        word = (word & ~mask) | ((value << lo) & mask)
    return word


def reg_pair(reg: int) -> tuple[int, ...]:
    return () if reg == RZ else (reg, reg + 1)


def reg_span(reg: int, count: int) -> tuple[int, ...]:
    return tuple(range(reg, reg + count))


def live_regs(*regs: int) -> tuple[int, ...]:
    return tuple(reg for reg in regs if reg != RZ)


def apply_guard(word: int, pred: Guard | None) -> tuple[int, tuple[int, ...]]:
    if pred is None:
        return word, ()
    reg, negated = pred
    return pack(word, (12, 4, reg | (negated << 3))), (reg,)


def LDC(rd: int, offset: int, wide: bool = False) -> Insn:
    template = TEMPLATES["LDC64" if wide else "LDC32"]
    word = pack(template, (16, 8, rd), (38, 16, offset))
    return Insn(word, writes=reg_pair(rd) if wide else (rd,), kind="var")


def LDCU64(ur: int, offset: int) -> Insn:
    word = pack(TEMPLATES["LDCU64"], (16, 8, ur), (37, 17, offset))
    return Insn(word, writes=(UR4,), kind="var")


def S2R(rd: int, sr: str) -> Insn:
    word = pack(TEMPLATES["S2R"], (16, 8, rd), (72, 8, SPECIAL_REGS[sr]))
    return Insn(word, writes=(rd,), kind="var")


def MOV_I(rd: int, imm: int) -> Insn:
    word = pack(TEMPLATES["MOV_I"], (16, 8, rd), (32, 32, imm & 0xFFFFFFFF))
    return Insn(word, writes=(rd,))


def _alu3(name: str, rd: int, ra: int, rb: int, rc: int = RZ, flags: int = 0) -> Insn:
    word = pack(TEMPLATES[name], (16, 8, rd), (24, 8, ra), (32, 8, rb), (64, 8, rc)) | flags
    return Insn(word, reads=live_regs(ra, rb, rc), writes=(rd,))


def IADD3(rd: int, ra: int, rb: int, rc: int = RZ, neg_b: bool = False) -> Insn:
    return _alu3("IADD3", rd, ra, rb, rc, NEG_SRC_B if neg_b else 0)


def IMAD(rd: int, ra: int, rb: int, rc: int = RZ) -> Insn:
    return _alu3("IMAD", rd, ra, rb, rc)


def FADD(rd: int, ra: int, rb: int, neg_b: bool = False) -> Insn:
    return _alu3("FADD", rd, ra, rb, flags=NEG_SRC_B if neg_b else 0)


def FMUL(rd: int, ra: int, rb: int) -> Insn:
    return _alu3("FMUL", rd, ra, rb)


def FFMA(rd: int, ra: int, rb: int, rc: int) -> Insn:
    return _alu3("FFMA", rd, ra, rb, rc)


def LOP3(rd: int, ra: int, rb: int, lut: int) -> Insn:
    return _alu3("LOP3", rd, ra, rb, flags=lut << 72)


def SHL(rd: int, ra: int, rb: int) -> Insn:
    return _alu3("SHL", rd, ra, rb)


def SHR(rd: int, ra: int, rb: int) -> Insn:
    return _alu3("SHR", rd, RZ, rb, ra)


def I2F(rd: int, ra: int) -> Insn:
    word = pack(TEMPLATES["I2F"], (16, 8, rd), (32, 8, ra))
    return Insn(word, reads=(ra,), writes=(rd,))


def IMAD_WIDE(rd: int, ra: int, imm: int, rc: int) -> Insn:
    word = pack(TEMPLATES["IMAD_WIDE"], (16, 8, rd), (24, 8, ra), (32, 32, imm), (64, 8, rc))
    return Insn(word, reads=live_regs(ra) + reg_pair(rc), writes=reg_pair(rd))


def _setp(
    name: str,
    cmp_code: int,
    pd: int,
    ra: int,
    rb: int,
    bop: str,
    src: Guard | None,
    cmp_width: int,
) -> Insn:
    src_pred, src_neg = src if src is not None else (PT, 0)
    word = pack(
        TEMPLATES[name],
        (24, 8, ra),
        (32, 8, rb),
        (74, 2, BOOL_OPS[bop]),
        (76, cmp_width, cmp_code),
        (81, 3, pd),
        (84, 3, PT),
        (87, 3, src_pred),
        (90, 1, src_neg),
    )
    return Insn(word, reads=live_regs(ra, rb), pred_reads=(src_pred,) if src is not None else (), pred_writes=(pd,))


def ISETP(op: str, pd: int, ra: int, rb: int, bop: str = "AND", src: Guard | None = None) -> Insn:
    return _setp("ISETP", ICMP_CODES[op], pd, ra, rb, bop, src, 3)


def FSETP(op: str, pd: int, ra: int, rb: int, bop: str = "AND", src: Guard | None = None) -> Insn:
    return _setp("FSETP", FCMP_CODES[op], pd, ra, rb, bop, src, 4)


def LDG(rd: int, ra: int, count: int = 1) -> Insn:
    word = pack(TEMPLATES["LDG"], (16, 8, rd), (24, 8, ra), (73, 3, ACCESS_WIDTHS[count]))
    return Insn(word, reads=reg_pair(ra) + (UR4,), writes=reg_span(rd, count), kind="var")


def STG(ra: int, rs: int, count: int = 1, pred: Guard | None = None) -> Insn:
    base = pack(TEMPLATES["STG"], (24, 8, ra), (32, 8, rs), (73, 3, ACCESS_WIDTHS[count]))
    word, pred_reads = apply_guard(base, pred)
    return Insn(word, reads=reg_pair(ra) + reg_span(rs, count) + (UR4,), pred_reads=pred_reads, kind="store")


def LDS(rd: int, ra: int, offset: int) -> Insn:
    word = pack(TEMPLATES["LDS"], (16, 8, rd), (24, 8, ra), (40, 24, offset))
    return Insn(word, reads=(ra,), writes=(rd,), kind="var")


def STS(ra: int, rs: int, offset: int) -> Insn:
    word = pack(TEMPLATES["STS"], (24, 8, ra), (32, 8, rs), (40, 24, offset))
    return Insn(word, reads=(ra, rs), kind="store")


def SHFL(rd: int, ra: int, lane: int) -> Insn:
    word = pack(TEMPLATES["SHFL"], (16, 8, rd), (24, 8, ra), (53, 5, lane))
    return Insn(word, reads=(ra,), writes=(rd,), kind="var")


def BAR() -> Insn:
    return Insn(TEMPLATES["BAR"])


def BSSY(barrier: int, label: str) -> Insn:
    return Insn(pack(TEMPLATES["BSSY"], (16, 4, barrier)), target=label)


def BSYNC(barrier: int) -> Insn:
    return Insn(pack(TEMPLATES["BSYNC"], (16, 4, barrier)))


def BRA(label: str, pred: Guard | None = None) -> Insn:
    word, pred_reads = apply_guard(TEMPLATES["BRA"], pred)
    return Insn(word, pred_reads=pred_reads, kind="branch", target=label)


def EXIT(pred: Guard | None = None) -> Insn:
    word, pred_reads = apply_guard(TEMPLATES["EXIT"], pred)
    return Insn(word, pred_reads=pred_reads, kind="branch")


def _insn_keys(insn: Insn) -> list[RegKey]:
    return [("r", reg) for reg in insn.reads + insn.writes] + [("p", p) for p in insn.pred_reads + insn.pred_writes]


def _write_keys(insn: Insn) -> list[RegKey]:
    return [("r", reg) for reg in insn.writes] + [("p", p) for p in insn.pred_writes]


def schedule(items: list[Item]) -> tuple[list[Insn], dict[str, int]]:
    insns: list[Insn] = []
    labels: dict[str, int] = {}
    ready: dict[RegKey, int] = {}
    pending_write: dict[RegKey, int] = {}
    pending_read: dict[RegKey, int] = {}

    cycle = 0
    prev: Insn | None = None
    slot_cursor = 0
    at_label = False

    for item in items:
        if isinstance(item, Label):
            labels[item.name] = len(insns)
            ready.clear()
            pending_write.clear()
            pending_read.clear()
            at_label = True
            continue

        wait_mask = ALL_SLOTS_MASK if at_label else 0
        fence = at_label or (prev is not None and prev.kind == "branch")
        at_label = False

        all_keys = _insn_keys(item)
        for key in all_keys:
            if key in pending_write:
                wait_mask |= 1 << pending_write[key]
        for key in _write_keys(item):
            if key in pending_read:
                wait_mask |= 1 << pending_read[key]

        for table in (pending_write, pending_read):
            for key in [k for k, slot in table.items() if (wait_mask >> slot) & 1]:
                del table[key]

        start = cycle + 1 if prev is not None else 0
        for key in all_keys:
            extra = GUARD_LATENCY - ALU_LATENCY if key[0] == "p" and item.kind in ("branch", "store") else 0
            start = max(start, ready.get(key, 0) + extra)

        if prev is not None:
            gap = start - cycle
            if gap > 15:
                raise RuntimeError("delay > 15")
            prev.delay = max(min(15, gap), BRANCH_DELAY if fence else 0)
            prev_slots = ((1 << prev.out_slot) | (1 << prev.in_slot)) & ALL_SLOTS_MASK
            if wait_mask & prev_slots:
                prev.delay = max(prev.delay, 2)

        cycle = start if prev is None else cycle + prev.delay
        item.wait_mask = wait_mask

        if item.kind == "var":
            if item.opcode == OP_LDG:
                item.out_slot = LDG_SLOT
            else:
                item.out_slot = slot_cursor
                slot_cursor = (slot_cursor + 1) % 4
            for reg in item.writes:
                pending_write[("r", reg)] = item.out_slot

        if item.kind in ("var", "store") and item.reads:
            item.in_slot = MEM_READ_SLOT
            for reg in item.reads:
                pending_read[("r", reg)] = MEM_READ_SLOT

        for reg in item.writes:
            ready[("r", reg)] = cycle + ALU_LATENCY
        for pred in item.pred_writes:
            ready[("p", pred)] = cycle + ALU_LATENCY

        if item.kind == "branch":
            item.delay = BRANCH_DELAY

        insns.append(item)
        prev = item

    if prev is not None:
        prev.delay = 1
    return insns, labels


def encode(items: list[Item]) -> bytes:
    insns, labels = schedule(items)
    chunks: list[bytes] = []

    for idx, insn in enumerate(insns):
        word = insn.word
        if insn.target is not None:
            delta = labels[insn.target] - (idx + 1)
            if insn.opcode == OP_BSSY:
                word = pack(word, (32, 32, delta * 16))
            else:
                offset = delta * 4
                word = pack(word, (16, 8, offset & 0xFF), (34, 48, (offset >> 8) & ((1 << 48) - 1)))

        word = pack(
            word,
            (105, 4, insn.delay),
            (109, 1, int(insn.delay < 12)),
            (110, 3, insn.out_slot),
            (113, 3, insn.in_slot),
            (116, 6, insn.wait_mask),
        )
        chunks.append(word.to_bytes(16, "little"))

    return b"".join(chunks)
