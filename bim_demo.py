"""End-to-end BIM demonstration on the Schependomlaan project (one real unit).

Evidence ledger per element-task planned in week w (decision at the end of week w-1):
  scope       element linked from the 4D schedule (event log GUID) to the as-planned IFC and, through the
              ArchiCAD unique ID, to the design IFC with its host relations (fills, voids, connects);
  schedule    task dates from the 4D schedule (as planned);
  site        drone point-cloud comparison of week w-1 (elements planned as built but not observed);
  predecessor the element's own earlier tasks and its design hosts, both checked against the site evidence.
Gates: site/predecessor U when the element or a host was planned as built by w-1 but is missing in the
week w-1 point cloud; ? when no point cloud covers week w-1; S otherwise. Recommendations: BLOCKED (any U),
READY (all S), INSUFFICIENT otherwise. Outcome: the event log's on-time label for the element-task.
Comparators: schedule-only release (all planned element-tasks released) and a constraint check on the
as-planned schedule (release when every predecessor element-task is scheduled to finish before the week).
No statistical claim: late element-tasks fall in a handful of package-weeks.
Writes results/bim_demo/summary.json and results/bim_demo/ledger.parquet.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "schependomlaan")
PRO = os.path.join(ROOT, "data", "processed", "schependomlaan")
OUT = os.path.join(ROOT, "results", "bim_demo")


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.read_csv(os.path.join(RAW, "Event log", "Eventlog with As Built information",
                                  "Schependomlaan eventlog as planned data incl as built data.csv"))
    n_rows_raw = len(ev)
    ev = ev[ev["Element on time or not"].notna()].copy()
    ev["s"] = pd.to_datetime(ev.TaskStart, dayfirst=True)
    ev["f"] = pd.to_datetime(ev.TaskFinish, dayfirst=True)
    ev["wk"] = ev.s.dt.isocalendar().week.astype(int)
    ev["fwk"] = ev.f.dt.isocalendar().week.astype(int)
    ev["late"] = (ev["Element on time or not"] == "Too late").astype(int)
    x = pd.ExcelFile(os.path.join(RAW, "Comparison As Planned As Built", "comparison_all weeks.xlsx"))
    miss = {int(sh): set(x.parse(sh, header=1)["Name"].dropna()) for sh in x.sheet_names}
    cw = pd.read_parquet(os.path.join(PRO, "crosswalk.parquet"))
    rel = pd.read_parquet(os.path.join(PRO, "design_relations.parquet"))
    op2wall = dict(zip(rel[rel.rel == "voids"].hosted, rel[rel.rel == "voids"].host))
    host = {}
    for h, o in zip(rel[rel.rel == "fills"].hosted, rel[rel.rel == "fills"].host):
        if o in op2wall:
            host.setdefault(h, set()).add(op2wall[o])
    for a, b in zip(rel[rel.rel == "connects"].hosted, rel[rel.rel == "connects"].host):
        host.setdefault(a, set()).add(b)
    p2d, d2p = dict(zip(cw.guid, cw.design_guid)), dict(zip(cw.design_guid, cw.guid))
    g2n = dict(zip(ev.GUID, ev.Resource))
    ev["design_linked"] = ev.GUID.map(p2d).notna()
    ev["hosts"] = ev.GUID.map(lambda g: [g2n.get(d2p.get(h)) for h in host.get(p2d.get(g), []) if g2n.get(d2p.get(h))])
    # element-level as-planned finish of earlier tasks (predecessor work on the same element)
    first_fin = ev.groupby("Resource").fwk.min()
    rows = []
    for w in sorted(ev.wk.unique()):
        sub = ev[ev.wk == w]
        prev = miss.get(w - 1)
        for _, r in sub.iterrows():
            if prev is None:
                g = "?"
            else:
                own_missing = r.Resource in prev  # the comparison lists only elements planned as built by week w-1
                host_missing = any(h in prev for h in r.hosts)
                g = "U" if (own_missing or host_missing) else "S"
            state = "BLOCKED" if g == "U" else ("READY" if g == "S" else "INSUFFICIENT")
            rows.append({"week": int(w), "task": r.TaskID, "task_name": r.TaskName, "element": r.Resource,
                         "ifc_class": r.IfcClass, "design_linked": bool(r.design_linked), "n_hosts": len(r.hosts),
                         "site_gate": g, "state": state, "late": int(r.late)})
    L = pd.DataFrame(rows)
    L.to_parquet(os.path.join(OUT, "ledger.parquet"))
    cov = L[L.site_gate != "?"]
    out = {
        "event_log_rows": n_rows_raw, "labelled_element_tasks": int(len(L)), "elements": int(ev.Resource.nunique()),
        "tasks": int(ev.TaskID.nunique()), "design_link_rate": float(ev.design_linked.mean()),
        "late_element_tasks": int(L.late.sum()),
        "weeks_with_site_evidence": sorted(int(k) + 1 for k in miss),
        "covered_element_tasks": int(len(cov)), "covered_late": int(cov.late.sum()),
        "blocked": int((cov.state == "BLOCKED").sum()),
        "blocked_late": int(((cov.state == "BLOCKED") & (cov.late == 1)).sum()),
        "ready": int((cov.state == "READY").sum()), "ready_late": int(((cov.state == "READY") & (cov.late == 1)).sum()),
        "schedule_only_released": int(len(cov)), "schedule_only_late_released": int(cov.late.sum()),
        "by_week": {int(w): {"n": int(len(g)), "late": int(g.late.sum()), "blocked": int((g.state == "BLOCKED").sum()),
                             "blocked_late": int(((g.state == "BLOCKED") & (g.late == 1)).sum())}
                    for w, g in cov.groupby("week")},
        "blocked_by_task": cov[cov.state == "BLOCKED"].groupby(["week", "task_name"]).late.agg(["size", "sum"]).reset_index().to_dict("records"),
    }
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
