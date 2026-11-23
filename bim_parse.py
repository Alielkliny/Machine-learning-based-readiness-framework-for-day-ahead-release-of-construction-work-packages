"""Parse the Schependomlaan design IFC (no geometry): element GUID, class, name, storey, material.
Writes data/processed/schependomlaan/design_elements.parquet."""
import sys, json
import ifcopenshell, ifcopenshell.util.element as ue
import pandas as pd
f = ifcopenshell.open(sys.argv[1])
rows = []
for e in f.by_type("IfcElement"):
    st = ue.get_container(e)
    mats = ue.get_materials(e)
    rows.append({"guid": e.GlobalId, "ifc_class": e.is_a(), "name": e.Name, "type": (ue.get_type(e).Name if ue.get_type(e) else None),
                 "storey": st.Name if st is not None and st.is_a("IfcBuildingStorey") else None,
                 "material": "; ".join(m.Name for m in mats if hasattr(m, "Name") and m.Name) if mats else None})
d = pd.DataFrame(rows)
d.to_parquet(sys.argv[2])
print(json.dumps({"schema": f.schema, "elements": len(d), "classes": d.ifc_class.value_counts().head(12).to_dict(),
                  "storeys": d.storey.value_counts().to_dict()}, indent=1, default=str))
