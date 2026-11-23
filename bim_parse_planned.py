"""Parse the as-planned IFC with planning data (Synchro export): elements with STEP id, GUID, class,
name, storey, the Synchro resource property set, and task assignments (IfcRelAssignsToProcess).
Writes data/processed/schependomlaan/planned_elements.parquet and planned_tasks.parquet."""
import sys, json
import ifcopenshell, ifcopenshell.util.element as ue
import pandas as pd
f = ifcopenshell.open(sys.argv[1])
rows = []
for e in f.by_type("IfcElement"):
    ps = ue.get_psets(e)
    syn = {k: v for k, v in ps.items() if "ynchro" in k}
    flat = {}
    for k, v in syn.items():
        for kk, vv in v.items():
            if kk != "id":
                flat[f"{kk}"] = vv
    st = ue.get_container(e)
    rows.append({"step_id": e.id(), "guid": e.GlobalId, "ifc_class": e.is_a(), "name": e.Name,
                 "storey": st.Name if st is not None and st.is_a("IfcBuildingStorey") else None,
                 "resource_name": f"{e.Name}_(#{e.id()})" if e.Name else None, **flat})
d = pd.DataFrame(rows)
tasks = []
for rel in f.by_type("IfcRelAssignsToProcess"):
    p = rel.RelatingProcess
    for o in rel.RelatedObjects or []:
        tasks.append({"task_step": p.id(), "task_name": getattr(p, "Name", None), "task_id": getattr(p, "Identification", None) or getattr(p, "TaskId", None),
                      "elem_step": o.id(), "elem_guid": getattr(o, "GlobalId", None)})
t = pd.DataFrame(tasks)
d.to_parquet(sys.argv[2]); t.to_parquet(sys.argv[3])
print(json.dumps({"schema": f.schema, "elements": len(d), "synchro_cols": [c for c in d.columns if c not in ("step_id","guid","ifc_class","name","storey","resource_name")][:20],
                  "task_links": len(t), "tasks": int(t.task_step.nunique()) if len(t) else 0}, indent=1, default=str))
