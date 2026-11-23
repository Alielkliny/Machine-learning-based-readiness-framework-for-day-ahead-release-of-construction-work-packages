"""Extract Street Manager archived permit notifications into a flat Parquet table.

Usage: python -I extract_events.py <month.zip> <out.parquet> <src_dir>
Each output row is one notification event. Only fields needed by the evidence
ledger are kept. Raw JSON is not modified; the archive SHA-256 is recorded in a
sidecar JSON file.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, sys.argv[3])
from stream_zip import iter_entries  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

FIELDS = [
    "work_reference_number", "permit_reference_number", "promoter_swa_code",
    "highway_authority_swa_code", "usrn", "works_location_coordinates",
    "work_category_ref", "traffic_management_type_ref", "proposed_start_date",
    "proposed_start_time", "proposed_end_date", "proposed_end_time",
    "actual_start_date_time", "actual_end_date_time", "work_status_ref",
    "permit_status", "is_deemed", "is_ttro_required", "is_traffic_sensitive",
    "collaboration_type_ref", "close_footway_ref", "permit_conditions",
    "works_location_type", "activity_type", "road_category",
    "current_traffic_management_type_ref", "current_traffic_management_update_date",
    "is_covid_19_response",
]
TOP = ["event_reference", "event_type", "event_time", "object_type",
       "object_reference", "version"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    src, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    keys = TOP + FIELDS + ["entry_name"]
    cols = {k: [] for k in keys}
    schema = pa.schema([(k, pa.string()) for k in keys])
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    rows_total = 0
    n_bad = 0
    n = 0

    def flush():
        nonlocal cols, rows_total
        if cols["entry_name"]:
            writer.write_table(pa.table({k: pa.array(v, type=pa.string()) for k, v in cols.items()}, schema=schema))
            rows_total += len(cols["entry_name"])
            cols = {k: [] for k in keys}

    for name, raw in iter_entries(src):
        n += 1
        try:
            j = json.loads(raw)
        except ValueError:
            n_bad += 1
            continue
        od = j.get("object_data") or {}
        for k in TOP:
            v = j.get(k)
            cols[k].append(None if v is None else str(v))
        for k in FIELDS:
            v = od.get(k)
            cols[k].append(None if v is None else str(v))
        cols["entry_name"].append(name)
        if len(cols["entry_name"]) >= 200000:
            flush()
    flush()
    writer.close()
    meta = {"archive": src, "archive_sha256": sha256(src), "entries": n,
            "json_errors": n_bad, "rows": rows_total,
            "seconds": round(time.time() - t0, 1)}
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
