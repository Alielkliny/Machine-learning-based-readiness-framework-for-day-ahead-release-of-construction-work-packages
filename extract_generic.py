"""Extract any Street Manager archived topic (activity, section_58) to Parquet,
keeping every top-level and object_data field as a string.
Usage: python -P -E extract_generic.py <month.zip> <out.parquet> <src_dir>"""
import hashlib, json, sys, time
sys.path.insert(0, sys.argv[3])
from stream_zip import iter_entries  # noqa: E402
import pyarrow as pa, pyarrow.parquet as pq  # noqa: E402,E401

def main():
    src, out = sys.argv[1], sys.argv[2]
    t0 = time.time(); rows = []; keys = set()
    for name, raw in iter_entries(src):
        j = json.loads(raw); od = j.pop("object_data", None) or {}
        r = {k: (None if v is None else str(v)) for k, v in j.items()}
        r.update({"od_" + k: (None if v is None else str(v)) for k, v in od.items()})
        r["entry_name"] = name; rows.append(r); keys |= set(r)
    keys = sorted(keys)
    table = pa.table({k: pa.array([r.get(k) for r in rows], type=pa.string()) for k in keys})
    pq.write_table(table, out, compression="zstd")
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    meta = {"archive": src, "archive_sha256": h, "rows": table.num_rows, "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out + ".json", "w"), indent=1); print(json.dumps(meta))

if __name__ == "__main__":
    main()
