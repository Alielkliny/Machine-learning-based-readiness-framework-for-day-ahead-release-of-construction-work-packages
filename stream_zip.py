"""Sequential reader for streamed ZIP archives that lack a central directory.

Yields (name, bytes) for each local file entry. Uses the deflate stream end to
find the next entry, so it does not depend on data-descriptor sizes.
"""
import struct
import zlib

LOCAL_SIG = b"PK\x03\x04"


def iter_entries(path, chunk=1 << 16):
    import mmap
    f = open(path, "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    n = len(buf)
    while True:
        i = buf.find(LOCAL_SIG, pos)
        if i < 0 or i + 30 > n:
            return
        (_sig, _ver, flag, meth, _mt, _md, _crc, csize, _usize, fnl, exl) = struct.unpack(
            "<IHHHHHIIIHH", buf[i:i + 30])
        name = bytes(buf[i + 30:i + 30 + fnl]).decode("utf8", "replace")
        start = i + 30 + fnl + exl
        if meth == 0 and not (flag & 0x08):
            yield name, bytes(buf[start:start + csize])
            pos = start + csize
            continue
        d = zlib.decompressobj(-15)
        out = []
        p = start
        ok = True
        try:
            while not d.eof and p < n:
                out.append(d.decompress(buf[p:p + chunk]))
                p += chunk
        except zlib.error:
            ok = False
        if not ok or not d.eof:
            # truncated or corrupt final entry
            return
        consumed = p - len(d.unused_data)
        yield name, b"".join(out)
        pos = consumed
