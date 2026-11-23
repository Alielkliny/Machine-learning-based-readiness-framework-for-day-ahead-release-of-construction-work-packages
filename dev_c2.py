"""Design C2 (C + online recalibration) on the development folds of src/dev_eval.py.
Adds per-row decisions for C2 and its part ablations to results/dev_eval/fold_rows_g{g}.parquet copies:
  framework_c2      ledger + gates + learned layer + online recalibration (promoter x authority, 28 days, k=30, Bayes threshold)
  c2_no_recal       = design C (the learned layer without recalibration, tuned threshold)
  c2_no_ledger      latest-record learner + authorisation gate + recalibration
  c2_no_gates       ledger learner + recalibration, no gates
  c2_no_learner     gates + shrunk track record + recalibration
Writes results/dev_c2/rows_g{g}.parquet. Usage: python -P src/dev_c2.py g
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev_eval as E  # noqa: E402
import methods as M  # noqa: E402

OUT = os.path.join(E.ROOT, "results", "dev_c2")
RECAL = dict(keys=("prom", "ha"), window=28, k=30.0)


def c2_scores(tr, te, ctx, cfg, seed, use_ledger=True, use_gates=True, use_learner=True, use_recal=True):
    cats = list(M.CAT_NOID)
    num = list(M.NUM_LEDGER if use_ledger else M.NUM_LATEST) + (list(M.NUM_POOL) if use_ledger else [])
    trf, tef, cxf = tr.copy(), te.copy(), ctx.copy()
    if use_gates and use_ledger:
        for d in (trf, tef, cxf):
            gv = M.gate_vector(d)
            for k in M.GATES:
                d[k] = gv[k].values
        num += M.GATES
    if use_learner:
        params = {"num_leaves": cfg["nl"], "min_child_samples": cfg["mc"]}
        mdl, Xtr = M._fit_lgb(trf, num, cats, seed, params)
        p = M._predict_lgb(mdl, Xtr, tef, num, cats)
        pc = M._predict_lgb(mdl, Xtr, cxf, num, cats) if use_recal else None
    else:
        p = te["eb_ph"].values.astype(float)
        pc = ctx["eb_ph"].values.astype(float)
    if use_recal:
        off = M.online_offsets(ctx, pc, te, **RECAL)
        p = 1.0 / (1.0 + np.exp(-(M._logit(p) + off)))
    gte = M.gate_vector(te)
    blocked = (gte[M.GATES] == M.U).any(axis=1).values if (use_gates and use_ledger) else (gte["g_auth"] == M.U).values
    auth = (gte["g_auth"] == M.S).values
    return p, blocked, auth


def main():
    g = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    s = E.load(["2024-03", "2024-04", "2024-05", "2024-06", "2024-07"])
    meta = json.load(open(os.path.join(E.OUT, f"fold_meta_g{g}.json")))[0]
    cfg = meta["sel"]["framework"]["cfg"]
    ms = E.month_start("2024-07")
    tr = s[(s["D"] >= ms - pd.DateOffset(months=4)) & (s["D"] < ms) & (s["grp"] != g)].reset_index(drop=True)
    te = s[(s["D"] >= ms) & (s["D"] < ms + pd.DateOffset(months=1)) & (s["grp"] == g)].reset_index(drop=True)
    ctx = s[(s["D"] >= ms - pd.DateOffset(days=40)) & (s["D"] < ms + pd.DateOffset(months=1)) & (s["grp"] == g)].reset_index(drop=True)
    rows = pd.read_parquet(os.path.join(E.OUT, f"fold_rows_g{g}.parquet"))
    assert (rows["pref"].values == te["pref"].values).all() and (rows["D"].values == te["D"].values).all()
    variants = {"framework_c2": {}, "c2_no_ledger": {"use_ledger": False}, "c2_no_gates": {"use_gates": False},
                "c2_no_learner": {"use_learner": False}}
    for name, kw in variants.items():
        p, blocked, auth = c2_scores(tr, te, ctx, cfg, E.SEED, **kw)
        if kw.get("use_gates", True) is False:
            dec = p >= 0.5
            sc = p
        else:
            dec = ~blocked & auth & (p >= 0.5)
            sc = np.where(blocked, -1.0, np.where(auth, p, p - 1.0))
        rows["dec_" + name], rows["sc_" + name] = dec, sc
    rows["dec_c2_no_recal"], rows["sc_c2_no_recal"] = rows["dec_framework"], rows["sc_framework"]
    rows.to_parquet(os.path.join(OUT, f"rows_g{g}.parquet"))
    print(json.dumps({"fold": g, "n": len(rows), "sec": round(time.time() - t0, 1),
                      "cost": {k: float(M.crew_day_loss(rows["dec_" + k].values, rows["y"].values))
                               for k in list(variants) + ["c2_no_recal", "kpi_supplier_score"]}}, indent=1))


if __name__ == "__main__":
    main()
