"""Minimal little-endian (Bedrock) NBT writer + reader (for round-trip verification)."""
import struct


class Byte(int): tag = 1
class Short(int): tag = 2
class Int(int): tag = 3
class Long(int): tag = 4
class Str(str): tag = 8


class List(list):
    tag = 9
    def __init__(self, elem_tag, items=()):
        super().__init__(items); self.elem_tag = elem_tag


class Compound(dict): tag = 10


def _tag_of(v):
    if isinstance(v, (Byte, Short, Int, Long, Str, List, Compound)):
        return v.tag
    if isinstance(v, bool): return 1
    if isinstance(v, int): return 3
    if isinstance(v, str): return 8
    if isinstance(v, dict): return 10
    raise TypeError(type(v))


def _payload(tag, v, out):
    if tag == 1: out += struct.pack("<b", int(v))
    elif tag == 2: out += struct.pack("<h", int(v))
    elif tag == 3: out += struct.pack("<i", int(v))
    elif tag == 4: out += struct.pack("<q", int(v))
    elif tag == 8:
        b = str(v).encode("utf-8"); out += struct.pack("<H", len(b)) + b
    elif tag == 9:
        et = v.elem_tag if isinstance(v, List) else (_tag_of(v[0]) if v else 1)
        out += struct.pack("<bi", et, len(v))
        for x in v: _payload(et, x, out)
    elif tag == 10:
        for k, x in v.items():
            t = _tag_of(x)
            out += struct.pack("<b", t)
            kb = k.encode("utf-8"); out += struct.pack("<H", len(kb)) + kb
            _payload(t, x, out)
        out += b"\x00"
    else:
        raise ValueError(tag)


def dumps(root, name=""):
    out = bytearray(b"\x0a")
    nb = name.encode(); out += struct.pack("<H", len(nb)) + nb
    _payload(10, root, out)
    return bytes(out)


def loads(b):
    pos = [0]
    def rd(fmt):
        v = struct.unpack_from(fmt, b, pos[0]); pos[0] += struct.calcsize(fmt); return v[0]
    def rs():
        n = rd("<H"); s = b[pos[0]:pos[0] + n].decode(); pos[0] += n; return s
    def pl(t):
        if t == 1: return rd("<b")
        if t == 2: return rd("<h")
        if t == 3: return rd("<i")
        if t == 4: return rd("<q")
        if t == 8: return rs()
        if t == 9:
            et = rd("<b"); n = rd("<i"); return [pl(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = rd("<b")
                if tt == 0: return d
                k = rs(); d[k] = pl(tt)
        raise ValueError(t)
    assert rd("<b") == 10; rs()
    return pl(10)
