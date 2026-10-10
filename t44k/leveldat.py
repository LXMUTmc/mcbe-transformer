"""Typed little-endian NBT for Bedrock level.dat (keeps every tag type, so a template can be edited and
written back byte-compatible).

level.dat = int32 storage version + int32 payload length + NBT (root compound, empty name).
Values are (tag, value) pairs; compounds are dicts (insertion order kept), lists are (elem_tag, [values]).

  python -m t44k.leveldat LEVEL_DAT          print the fields
"""
import struct, sys

END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BARR, STR, LIST, COMP, IARR, LARR = range(13)


def _rd(b, pos, t):
    if t == BYTE: return struct.unpack_from("<b", b, pos)[0], pos + 1
    if t == SHORT: return struct.unpack_from("<h", b, pos)[0], pos + 2
    if t == INT: return struct.unpack_from("<i", b, pos)[0], pos + 4
    if t == LONG: return struct.unpack_from("<q", b, pos)[0], pos + 8
    if t == FLOAT: return struct.unpack_from("<f", b, pos)[0], pos + 4
    if t == DOUBLE: return struct.unpack_from("<d", b, pos)[0], pos + 8
    if t == STR:
        n = struct.unpack_from("<H", b, pos)[0]
        return b[pos + 2:pos + 2 + n].decode("utf-8"), pos + 2 + n
    if t in (BARR, IARR, LARR):
        n = struct.unpack_from("<i", b, pos)[0]; pos += 4
        fmt, size = {BARR: ("b", 1), IARR: ("i", 4), LARR: ("q", 8)}[t]
        return list(struct.unpack_from("<%d%s" % (n, fmt), b, pos)), pos + n * size
    if t == LIST:
        et, n = struct.unpack_from("<bi", b, pos); pos += 5
        out = []
        for _ in range(n):
            v, pos = _rd(b, pos, et); out.append(v)
        return (et, out), pos
    if t == COMP:
        d = {}
        while True:
            tt = b[pos]; pos += 1
            if tt == END:
                return d, pos
            n = struct.unpack_from("<H", b, pos)[0]
            k = b[pos + 2:pos + 2 + n].decode("utf-8"); pos += 2 + n
            v, pos = _rd(b, pos, tt)
            d[k] = (tt, v)
    raise ValueError(t)


def _wr(t, v):
    if t == BYTE: return struct.pack("<b", v)
    if t == SHORT: return struct.pack("<h", v)
    if t == INT: return struct.pack("<i", v)
    if t == LONG: return struct.pack("<q", v)
    if t == FLOAT: return struct.pack("<f", v)
    if t == DOUBLE: return struct.pack("<d", v)
    if t == STR:
        e = v.encode("utf-8"); return struct.pack("<H", len(e)) + e
    if t in (BARR, IARR, LARR):
        fmt = {BARR: "b", IARR: "i", LARR: "q"}[t]
        return struct.pack("<i", len(v)) + struct.pack("<%d%s" % (len(v), fmt), *v)
    if t == LIST:
        et, items = v
        return struct.pack("<bi", et, len(items)) + b"".join(_wr(et, x) for x in items)
    if t == COMP:
        out = b""
        for k, (tt, x) in v.items():
            e = k.encode("utf-8")
            out += bytes([tt]) + struct.pack("<H", len(e)) + e + _wr(tt, x)
        return out + b"\x00"
    raise ValueError(t)


def read(path):
    b = open(path, "rb").read()
    version, length = struct.unpack_from("<ii", b, 0)
    assert b[8] == COMP and len(b) - 8 == length, "not a Bedrock level.dat"
    n = struct.unpack_from("<H", b, 9)[0]
    root, _ = _rd(b, 11 + n, COMP)
    return version, root


def write(path, version, root):
    body = bytes([COMP]) + struct.pack("<H", 0) + _wr(COMP, root)
    open(path, "wb").write(struct.pack("<ii", version, len(body)) + body)


def show(root, ind=""):
    for k, (t, v) in root.items():
        if t == COMP:
            print(f"{ind}{k}:"); show(v, ind + "  ")
        else:
            print(f"{ind}{k} [{t}] = {v!r}"[:160])


if __name__ == "__main__":
    ver, root = read(sys.argv[1])
    print("storage version", ver)
    show(root)
