"""Post-hoc analyses from saved per-row results (no refitting):
  * cost-ratio sensitivity: crew-day loss at wasted-mobilisation to lost-start cost ratios 0.5 to 3, with each
    probabilistic method at its Bayes threshold c_fr/(c_fr+c_ud) and rule-based methods unchanged;
  * three-state decision output: share of packages and observed start rate per recommendation state;
  * worked example: one held-out package released by the practice rule but not by the framework.
Usage: python -P src/post_analysis.py   (reads results/dev_c2, results/test; writes results/post/*.json)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "results", "post")
PROB = ["framework", "logit_latest", "rf_latest", "lgb_latest", "fw_no_recal"]
RULE = ["naive_release_all", "practice_granted", "constraint_check_ldcc"]


def load_split(name):
    if name == "development":
        return pd.read_parquet(os.path.join(R, "results/dev_c2/fold_rows_c2.parquet"))
    if name == "check_slice":
        return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(R, "results/test/check_slice_g*.parquet")))], ignore_index=True)
    d = pd.read_parquet(os.path.join(R, f"results/test/{name.split('_')[0]}.parquet"))
    return d[d["cut"] == "heldout_authority"].reset_index(drop=True)


def loss(rel, y, cfr, cud):
    rel = np.asarray(rel, bool)
    return 100.0 * (cfr * np.sum(rel & (y == 0)) + cud * np.sum(~rel & (y == 1))) / len(y)


def prob_of(d, m):
    sc = d["sc_" + m].values.astype(float)
    if m in ("framework", "fw_no_recal"):
        # framework scores encode blocked as -1 and unauthorised as p - 1; only authorised, unblocked packages can pass
        return np.where(sc >= 0, sc, -1.0)
    return sc


def main():
    os.makedirs(OUT, exist_ok=True)
    splits = ["development", "check_slice", "R1_heldout", "R2_heldout"]
    ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    sens, states = {}, {}
    for sp in splits:
        try:
            d = load_split(sp)
        except FileNotFoundError:
            continue
        y = d["y"].values
        sens[sp] = {}
        for c in ratios:
            thr = c / (c + 1.0)
            row = {m: float(loss(prob_of(d, m) >= thr, y, c, 1.0)) for m in PROB}
            for m in RULE:
                row[m] = float(loss(d["dec_" + m].values, y, c, 1.0))
            # the KPI score is the promoter's on-time start rate for granted packages (-1 otherwise), read as a probability
            k = d["sc_kpi_supplier_score"].values.astype(float)
            row["kpi_supplier_score"] = float(loss(k >= thr, y, c, 1.0)) if c != 1.0 else float(loss(d["dec_kpi_supplier_score"].values, y, c, 1.0))
            sens[sp][str(c)] = row
        # three-state output of the framework at equal costs
        p = d["sc_framework"].values.astype(float)
        st = np.where(np.isclose(p, -1.0), "BLOCKED", np.where(d["dec_framework"].values, "READY", "INSUFFICIENT"))
        states[sp] = {s: {"share": float(np.mean(st == s)), "start_rate": float(y[st == s].mean()) if np.any(st == s) else None,
                          "n": int(np.sum(st == s))} for s in ("READY", "INSUFFICIENT", "BLOCKED")}
        states[sp]["practice_released_start_rate"] = float(y[d["dec_practice_granted"].values].mean())
    json.dump(sens, open(os.path.join(OUT, "cost_ratio_sensitivity.json"), "w"), indent=1)
    json.dump(states, open(os.path.join(OUT, "three_state_output.json"), "w"), indent=1)
    print(json.dumps(states, indent=1))


if __name__ == "__main__":
    main()
