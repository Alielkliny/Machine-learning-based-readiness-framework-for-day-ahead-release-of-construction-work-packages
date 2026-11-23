"""Worked example: one hold-out package of test period R1 traced through the register at its cutoff.

Selection rule (fixed before reading the records): the first R1 hold-out package, in file order, that the practice
rule and the random-forest risk score released, the framework blocked, and that did not start on its planned day, and
whose register row records an alteration granted before the cutoff. Reads results/test/R1.parquet, the register
snapshot of the month and the permit and section 58 events; writes results/post/worked_example.json.
Usage: python src/worked_example.py
"""
import json
import os
import sys

import duckdb
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "src"))
from methods import gate_vector  # noqa: E402
from splits import is_holdout_authority  # noqa: E402


def main():
    rows = pd.read_parquet(os.path.join(R, "results", "test", "R1.parquet"),
                           columns=["pref", "D", "ha", "y", "dec_practice_granted", "dec_framework", "sc_framework",
                                    "sc_rf_latest", "dec_rf_latest"])
    rows = rows[rows["ha"].map(is_holdout_authority)]
    cand = rows[(rows.dec_practice_granted == 1) & (rows.sc_framework == -1.0) & (rows.y == 0) & (rows.dec_rf_latest == 1)]
    con = duckdb.connect()
    con.execute("SET threads=4")
    snap = os.path.join(R, "data", "processed", "ledger", "snap_2024_09.parquet").replace("\\", "/")
    reg = con.execute(f"SELECT * FROM '{snap}' WHERE L = 1 AND pref IN (SELECT UNNEST(?))", [list(cand.pref)]).df()
    reg = reg[reg["n_alt"] > 0]
    first = [p for p in cand.pref if p in set(reg.pref)][0]
    r = reg[reg.pref == first].iloc[0]
    c = cand[cand.pref == first].iloc[0]
    reg1 = reg[reg.pref == first].copy()
    reg1["load_ratio"] = reg1["load_ha"] / reg1["nd_cap"].where(reg1["nd_cap"] > 0)
    g = gate_vector(reg1).iloc[0].to_dict()
    ev = os.path.join(R, "data", "processed", "events").replace("\\", "/")
    permit = con.execute(f"""SELECT event_type, event_time, permit_status, proposed_start_date, actual_start_date_time
        FROM read_parquet('{ev}/permit_2024_0[789].parquet') WHERE permit_reference_number = ? ORDER BY event_time""",
                         [first]).df()
    s58 = con.execute(f"""SELECT event_type, event_time, od_section_58_status AS status, od_start_date AS start_date,
        od_end_date AS end_date, od_section_58_duration AS duration
        FROM read_parquet('{ev}/section_58_202[34]_*.parquet') WHERE CAST(od_usrn AS VARCHAR) = ? ORDER BY event_time""",
                      [str(r["usrn"])]).df()
    out = {
        "selection": "first R1 hold-out package released by practice and random forest, blocked by the framework, "
                     "not started on its planned day, with an alteration granted before the cutoff",
        "planned_day": str(pd.Timestamp(r["D"]).date()), "cutoff_day": str(pd.Timestamp(r["cday"]).date()),
        "category": r["cat"], "permit_status_at_cutoff": r["pstatus"], "latest_event_at_cutoff": r["last_etype"],
        "hours_since_grant": float(r["h_since_grant"]), "alterations_before_cutoff": int(r["n_alt"]),
        "date_changed_by_latest_record": int(r["date_changed"]), "section58_in_force": int(r["s58_n"]),
        "sibling_in_progress": int(r["sib_inprog"]), "sibling_newer_planned": int(r["sib_newer_planned"]),
        "gates": {k: str(v) for k, v in g.items()}, "rf_probability": float(c["sc_rf_latest"]),
        "actual_start": str(r["start_et"]) if pd.notna(r["start_et"]) else None,
        "permit_events": [{"type": e.event_type, "time": str(e.event_time), "status": e.permit_status,
                           "proposed_start": str(e.proposed_start_date)} for e in permit.itertuples()],
        "section58_events": [{"type": e.event_type, "time": str(e.event_time), "status": e.status,
                              "start": str(e.start_date), "end": str(e.end_date), "duration": str(e.duration)}
                             for e in s58.itertuples()],
    }
    json.dump(out, open(os.path.join(R, "results", "post", "worked_example.json"), "w", encoding="utf8"), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
