"""Host-hosted relations from the design IFC plus an ArchiCAD Unique ID crosswalk to the as-planned IFC.
Writes data/processed/schependomlaan/design_relations.parquet and design_uid.parquet."""
import sys, json
import ifcopenshell, ifcopenshell.util.element as ue
import pandas as pd
f = ifcopenshell.open(sys.argv[1])
rows = []
for r in f.by_type("IfcRelFillsElement"):
    rows.append({"rel": "fills", "hosted": r.RelatedBuildingElement.GlobalId, "host": r.RelatingOpeningElement.GlobalId})
for r in f.by_type("IfcRelVoidsElement"):
    rows.append({"rel": "voids", "hosted": r.RelatedOpeningElement.GlobalId, "host": r.RelatingBuildingElement.GlobalId})
for r in f.by_type("IfcRelConnectsPathElements") + f.by_type("IfcRelConnectsElements"):
    try:
        rows.append({"rel": "connects", "hosted": r.RelatedElement.GlobalId, "host": r.RelatingElement.GlobalId})
    except Exception:
        pass
d = pd.DataFrame(rows, columns=["rel", "hosted", "host"])
d.to_parquet(sys.argv[2])
uid = []
for e in f.by_type("IfcElement"):
    ps = ue.get_psets(e)
    u = None
    for k, v in ps.items():
        if "ArchiCADProperties" in k and "Unique ID" in v:
            u = v["Unique ID"]
    uid.append({"guid": e.GlobalId, "ifc_class": e.is_a(), "name": e.Name, "archicad_uid": u})
pd.DataFrame(uid).to_parquet(sys.argv[3])
print(json.dumps({"relations": d.rel.value_counts().to_dict(), "elements": len(uid), "with_uid": sum(1 for x in uid if x["archicad_uid"])}))
