import struct
from dataclasses import dataclass

SM = 100
PARAM_BASE = 0x380
SMEM_BASE = 0x400

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_NV_INFO = 0x70000000
SHT_NV_COMPAT = 0x70000086

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXEC = 0x4
SHF_INFO_LINK = 0x40
SHF_NV_SHARED = 0x403
SHF_NV_CUINFO = 0x1000040
SHF_NV_TKINFO = 0x2000000

PT_LOAD = 1
PT_PHDR = 6
PF_R = 4
PF_RX = 5
PF_RW = 6

EIATTR_PARAM_CBANK = 0x0A
EIATTR_KPARAM_INFO = 0x17
EIATTR_REGCOUNT = 0x2F
EIFMT_SVAL = 4

NOTE_CUINFO = 1000
NOTE_TKINFO = 2000

SYM_SECTION = 3
SYM_GLOBAL_FUNC = 0x12
SYM_STO_CUDA_ENTRY = 0x10

EHDR_PHOFF = 0x20
EHDR_PHNUM = 0x38
EHDR_SHNUM = 0x3C

EHDR = (
    b"\x7fELF\x02\x01\x01\x41\x08"
    + bytes(7)
    + struct.pack("<HHIQQQIHHHHHH", 2, 0xBE, 1, 0, 0, 0, 0x06006402, 64, 56, 0, 64, 0, 0)
)
COMPAT = bytes.fromhex("0209010002020100020505000307010102030000040b08000900000000000000")


@dataclass
class Section:
    name: str
    kind: int
    data: bytes = b""
    flags: int = 0
    link: int = 0
    info: int = 0
    align: int = 1
    entsize: int = 0
    size: int | None = None

    def __post_init__(self) -> None:
        if self.size is None:
            self.size = len(self.data)


class StringTable:
    def __init__(self) -> None:
        self._buf = bytearray(1)

    def add(self, name: str) -> int:
        offset = len(self._buf)
        self._buf.extend(name.encode() + b"\0")
        return offset

    def to_bytes(self) -> bytes:
        return bytes(self._buf)


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


def _pack_note(kind: int, desc: bytes) -> bytes:
    return struct.pack("<III", 12, len(desc), kind) + b"NVIDIA Corp\0" + desc


def _pack_attr(attr_id: int, payload: bytes) -> bytes:
    return struct.pack("<BBH", EIFMT_SVAL, attr_id, len(payload)) + payload


def _pack_symbol(name_offset: int, info: int, other: int, shndx: int, size: int) -> bytes:
    return struct.pack("<IBBHQQ", name_offset, info, other, shndx, 0, size)


def _layout_params(param_sizes: list[int]) -> tuple[list[int], int]:
    offsets: list[int] = []
    end = 0
    for size in param_sizes:
        end = _align_up(end, size)
        offsets.append(end)
        end += size
    return offsets, end


def _encode_param_info(offsets: list[int], param_sizes: list[int], total_size: int) -> bytes:
    records = [
        _pack_attr(
            EIATTR_KPARAM_INFO,
            struct.pack("<IHHI", 0, idx, offsets[idx], (param_sizes[idx] << 18) | 0x1F000),
        )
        for idx in reversed(range(len(offsets)))
    ]
    records.append(_pack_attr(EIATTR_PARAM_CBANK, struct.pack("<IHH", 4, PARAM_BASE, total_size)))
    return b"".join(records)


def _program_header_flags(section_flags: int) -> int:
    if section_flags & SHF_EXEC:
        return PF_RX
    if section_flags & SHF_WRITE:
        return PF_RW
    return PF_R


def make_cubin(
    kernel: str,
    text: bytes,
    param_sizes: list[int],
    nregs: int,
    smem: int = 0,
    full: bool = False,
) -> bytes:
    offsets, param_total = _layout_params(param_sizes)
    strtab = StringTable()

    text_name = f".text.{kernel}"
    const_name = f".nv.constant0.{kernel}"
    info_name = f".nv.info.{kernel}"
    shared_name = f".nv.shared.{kernel}"

    order = [".strtab", ".symtab", ".note.nv.tkinfo", ".note.nv.cuinfo", ".nv.info"]
    if full:
        order.append(".nv.compat")
    order.extend([info_name, text_name])
    if smem:
        order.append(shared_name)
    order.append(const_name)

    section_index = {name: i for i, name in enumerate(order, 1)}

    symbols = [_pack_symbol(0, 0, 0, 0, 0)]
    if full:
        for name in (".note.nv.tkinfo", ".note.nv.cuinfo", text_name, const_name):
            symbols.append(_pack_symbol(strtab.add(name), SYM_SECTION, 0, section_index[name], 0))
    kernel_symbol = len(symbols)
    symbols.append(
        _pack_symbol(strtab.add(kernel), SYM_GLOBAL_FUNC, SYM_STO_CUDA_ENTRY, section_index[text_name], len(text))
    )

    section_map = {
        "": Section("", SHT_NULL),
        ".strtab": Section(".strtab", SHT_STRTAB),
        ".symtab": Section(
            ".symtab", SHT_SYMTAB, b"".join(symbols), link=1, info=len(symbols), align=8, entsize=24
        ),
        ".note.nv.tkinfo": Section(
            ".note.nv.tkinfo", SHT_NOTE, _pack_note(NOTE_TKINFO, bytes(32)), flags=SHF_NV_TKINFO, align=4
        ),
        ".note.nv.cuinfo": Section(
            ".note.nv.cuinfo",
            SHT_NOTE,
            _pack_note(NOTE_CUINFO, struct.pack("<HHI", 2, SM, 0)),
            flags=SHF_NV_CUINFO,
            link=2,
            info=kernel_symbol,
            align=4,
        ),
        ".nv.info": Section(
            ".nv.info",
            SHT_NV_INFO,
            _pack_attr(EIATTR_REGCOUNT, struct.pack("<II", kernel_symbol, nregs)),
            link=2,
            align=4,
        ),
        ".nv.compat": Section(".nv.compat", SHT_NV_COMPAT, COMPAT, align=4),
        info_name: Section(
            info_name,
            SHT_NV_INFO,
            _encode_param_info(offsets, param_sizes, param_total),
            flags=SHF_INFO_LINK,
            link=2,
            info=section_index[text_name],
            align=4,
        ),
        text_name: Section(
            text_name,
            SHT_PROGBITS,
            text,
            flags=SHF_ALLOC | SHF_EXEC,
            link=2,
            info=kernel_symbol,
            align=16,
        ),
        shared_name: Section(
            shared_name,
            SHT_NOBITS,
            flags=SHF_NV_SHARED,
            info=section_index[text_name],
            align=16,
            size=smem + SMEM_BASE,
        ),
        const_name: Section(
            const_name,
            SHT_NOBITS,
            flags=SHF_ALLOC | SHF_INFO_LINK,
            info=section_index[text_name],
            align=4,
            size=PARAM_BASE + param_total,
        ),
    }

    sections = [section_map[name] for name in ("", *order)]
    name_offsets = {sec.name: strtab.add(sec.name) for sec in sections[1:]}
    sections[1].data = strtab.to_bytes()
    sections[1].size = len(sections[1].data)

    return _pack_elf(sections, name_offsets, full)


def _pack_elf(sections: list[Section], name_offsets: dict[str, int], full: bool) -> bytes:
    out = bytearray(EHDR)
    file_offsets: list[int] = []

    for sec in sections:
        out += bytes(-len(out) % sec.align)
        file_offsets.append(len(out))
        out += sec.data

    out += bytes(-len(out) % 8)
    shoff = len(out)

    for i, sec in enumerate(sections):
        if i == 0:
            out += bytes(64)
            continue
        out += struct.pack(
            "<IIQQQQIIQQ",
            name_offsets[sec.name],
            sec.kind,
            sec.flags,
            0,
            file_offsets[i],
            sec.size,
            sec.link,
            sec.info,
            sec.align,
            sec.entsize,
        )

    headers: list[tuple[int, int, int, int, int]] = []
    phoff = len(out)
    if full:
        loads = [
            (_program_header_flags(sec.flags), file_offsets[i], len(sec.data), sec.size or 0)
            for i, sec in enumerate(sections)
            if sec.flags & SHF_ALLOC
        ]
        ph_size = 56 * (len(loads) + 2)
        headers = [(PT_PHDR, PF_R, phoff, ph_size, ph_size), (PT_LOAD, PF_R, phoff, ph_size, ph_size)]
        headers.extend((PT_LOAD, *load) for load in loads)

    for kind, flags, offset, filesz, memsz in headers:
        out += struct.pack("<IIQQQQQQ", kind, flags, offset, 0, 0, filesz, memsz, 8)

    struct.pack_into("<QQ", out, EHDR_PHOFF, phoff if headers else 0, shoff)
    struct.pack_into("<H", out, EHDR_PHNUM, len(headers))
    struct.pack_into("<HH", out, EHDR_SHNUM, len(sections), 1)
    return bytes(out)
