"""Per-authority reductions and reliability of the readiness probability on the hold-out authorities of the two test
periods, from the saved row files (no refitting). Writes results/post/authority_spread.json and
results/post/reliability.json.

Crew-day loss per authority uses the paper's definition at a 1:1 cost ratio (Eq. 3). The framework score encodes
BLOCKED as -1, a package whose authorization is unresolved as p - 1 and a package with a supported authorization as p,
so the corrected probability of every non-blocked package is recovered as p = score (score >= 0) or score + 1.
Reliability of both the framework and the random forest is measured on the same packages, those the framework did not block.
Usage: python src/post_extra.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "src"))
from splits import is_holdout_authority  # noqa: E402

RIVALS = ["practice_granted", "kpi_supplier_score", "rf_latest", "logit_latest", "constraint_check_ldcc"]


def loss(dec, y):
    return 100.0 * float(np.mean(dec * (1 - y) + (1 - dec) * y))


def main():
    spread, rel = {}, {}
    for split in ("R1", "R2"):
        cols = ["ha", "y", "dec_framework", "sc_framework", "sc_rf_latest"] + [f"dec_{r}" for r in RIVALS]
        d = pd.read_parquet(os.path.join(R, "results", "test", f"{split}.parquet"), columns=cols)
        h = d[d["ha"].map(is_holdout_authority)].copy()
        summ = json.load(open(os.path.join(R, "results", "test", f"{split}_heldout_summary.json"), encoding="utf8"))
        pooled = loss(h["dec_framework"].values, h["y"].values)
        if abs(pooled - summ["pooled"]["framework"]["crew_day_loss"]) > 1e-6:
            raise SystemExit(f"{split}: pooled loss {pooled} does not match the stored summary")
        per = []
        for ha, g in h.groupby("ha"):
            y = g["y"].values
            lf = loss(g["dec_framework"].values, y)
            row = {"n": int(len(g)), "framework": lf}
            for r in RIVALS:
                lr = loss(g[f"dec_{r}"].values, y)
                row[r] = lr
                row["red_" + r] = (lr - lf) / lr if lr > 0 else None
            per.append(row)
        per_df = pd.DataFrame(per)
        out = {"n_authorities": int(len(per_df)), "per_authority": per}
        for r in RIVALS:
            v = per_df["red_" + r]
            out["share_positive_" + r] = float((v > 0).mean())
            out["median_" + r] = float(v.median())
            out["q25_" + r] = float(v.quantile(0.25))
            out["q75_" + r] = float(v.quantile(0.75))
        worst = min(per, key=lambda r: r["red_rf_latest"])
        out["worst_rf"] = {"red": worst["red_rf_latest"], "n": worst["n"], "framework": worst["framework"],
                           "rf_latest": worst["rf_latest"]}
        out["n_negative_rf"] = int((per_df["red_rf_latest"] <= 0).sum())
        spread[split] = out

        blocked = h["sc_framework"].values == -1.0
        p_fw = np.where(h["sc_framework"].values >= 0, h["sc_framework"].values, h["sc_framework"].values + 1.0)
        rows = {}
        for name, p, mask in (("framework", p_fw, ~blocked), ("rf_latest", h["sc_rf_latest"].values, ~blocked)):
            x = pd.DataFrame({"p": p[mask], "y": h["y"].values[mask]})
            x["bin"] = pd.qcut(x["p"], 10, labels=False, duplicates="drop")
            g = x.groupby("bin").agg(p=("p", "mean"), o=("y", "mean"), n=("y", "size"))
            ece = float(np.average(np.abs(g["p"] - g["o"]), weights=g["n"]))
            rows[name] = {"mean_pred": g["p"].tolist(), "observed": g["o"].tolist(), "n": g["n"].astype(int).tolist(),
                          "ece": ece, "n_rows": int(mask.sum())}
        rows["blocked_share"] = float(blocked.mean())
        rows["blocked_start_rate"] = float(h["y"].values[blocked].mean())
        rel[split] = rows
        print(split, "authorities", len(per_df), {r: round(out["share_positive_" + r], 3) for r in RIVALS},
              "ECE fw", round(rows["framework"]["ece"], 3), "rf", round(rows["rf_latest"]["ece"], 3))
    json.dump(spread, open(os.path.join(R, "results", "post", "authority_spread.json"), "w", encoding="utf8"), indent=1)
    json.dump(rel, open(os.path.join(R, "results", "post", "reliability.json"), "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    main()
