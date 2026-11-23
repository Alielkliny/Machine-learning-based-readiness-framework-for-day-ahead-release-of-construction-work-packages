"""Explicit host-hosted relations in the as-planned IFC: fills (window/door -> opening), voids (opening -> wall),
aggregates, connects. Writes data/processed/schependomlaan/relations.parquet."""
import sys, json
import ifcopenshell
import pandas as pd
f = ifcopenshell.open(sys.argv[1])
rows = []
for r in f.by_type("IfcRelFillsElement"):
    rows.append({"rel": "fills", "hosted": r.RelatedBuildingElement.id(), "host": r.RelatingOpeningElement.id()})
for r in f.by_type("IfcRelVoidsElement"):
    rows.append({"rel": "voids", "hosted": r.RelatedOpeningElement.id(), "host": r.RelatingBuildingElement.id()})
for r in f.by_type("IfcRelConnectsElements"):
    rows.append({"rel": "connects", "hosted": r.RelatedElement.id(), "host": r.RelatingElement.id()})
for r in f.by_type("IfcRelAggregates"):
    for o in r.RelatedObjects:
        if o.is_a("IfcElement") and r.RelatingObject.is_a("IfcElement"):
            rows.append({"rel": "aggregates", "hosted": o.id(), "host": r.RelatingObject.id()})
d = pd.DataFrame(rows)
d.to_parquet(sys.argv[2])
print(json.dumps(d.rel.value_counts().to_dict()))
