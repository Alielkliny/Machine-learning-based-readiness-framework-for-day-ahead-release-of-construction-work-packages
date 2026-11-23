"""Frozen-design scoring of the check slice and the sealed reserves.

Frozen after development (results/dev_eval/fold_meta_g*.json, majority choice across folds, median threshold):
  rf_latest     min_samples_leaf 20, max_features 0.3, identities, 200 trees, threshold 0.54
  logit_latest  C 10, identities, threshold 0.53
  lgb_latest    num_leaves 63, min_child_samples 100, identities, threshold 0.51
  framework     design C2: num_leaves 63, min_child_samples 300, no identities, online recalibration
                (promoter x authority, 28 days, k 30), Bayes threshold 0.5
  fw_no_recal   design C (threshold 0.50); fw_no_ledger, fw_no_gates, fw_no_learner: C2 minus one part
  kpi           threshold chosen on training data, as in development
Usage: python -P src/test_eval.py SPLIT [g]
  SPLIT check_slice (needs g): train 2024-04..07 on development groups != g, score 2024-08 group g
  SPLIT R1: train 2024-05..08 on development authorities, score 2024-09-02..2024-10-31, all authorities
  SPLIT R2: train 2024-08..11 on development authorities, score 2024-12-02..2025-01-24, all authorities
Writes results/test/{SPLIT}[_g].parquet with per-row decisions and scores, and a cut column
(heldout_authority for A_H, development_authority otherwise).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev_eval as E  # noqa: E402
import dev_c2 as C2  # noqa: E402
import methods as M  # noqa: E402
from splits import is_holdout_authority  # noqa: E402

OUT = os.path.join(E.ROOT, "results", "test")
SEED = E.SEED
FROZEN = {
    "rf_latest": ({"leaf": 20, "mf": 0.3, "n": 200}, True, 0.54),
    "logit_latest": ({"C": 10.0}, True, 0.53),
    "lgb_latest": ({"nl": 63, "mc": 100}, True, 0.51),
    "framework_cfg": {"nl": 63, "mc": 300},
    "fw_no_recal_thr": 0.50,
}


def load_all(months):
    fr = []
    for m in months:
        d = pd.read_parquet(os.path.join(E.LED, f"snap_{m.replace('-', '_')}.parquet"))
        fr.append(d[d["L"] == 1])
    s = pd.concat(fr, ignore_index=True)
    s = M.add_derived(s)
    s["heldout"] = s["ha"].map(is_holdout_authority)
    s["grp"] = s["ha"].map(lambda h: int(hashlib.sha256(f"{h}|dev-fold".encode()).hexdigest(), 16) % E.NGROUP)
    return s


def score(tr, te, ctx):
    y = te["y"].values
    res = pd.DataFrame({"pref": te["pref"], "D": te["D"], "prom": te["prom"], "ha": te["ha"], "y": y,
                        "granted": (te["pstatus"] == "granted").values,
                        "cut": np.where(te["heldout"], "heldout_authority", "development_authority"),
                        "month": te["D"].dt.strftime("%Y-%m"), "grp": te["grp"]})
    res["dec_naive_release_all"], res["sc_naive_release_all"] = True, 1.0
    for name, fn in (("practice_granted", M.policy_practice), ("constraint_check_ldcc", M.policy_constraint_check),
                     ("kpi_supplier_score", M.policy_kpi)):
        r, p = fn(tr, te)
        res["dec_" + name], res["sc_" + name] = r, p
    for name, kind in (("rf_latest", "rf"), ("logit_latest", "logit")):
        cfg, ids, thr = FROZEN[name]
        p = E.fit_predict_sklearn(kind, cfg, tr, te, ids, SEED)
        res["sc_" + name], res["dec_" + name] = p, p >= thr
    cfg, ids, thr = FROZEN["lgb_latest"]
    p = E.lgb_latest(tr, te, cfg, ids, SEED)
    res["sc_lgb_latest"], res["dec_lgb_latest"] = p, p >= thr
    fcfg = FROZEN["framework_cfg"]
    variants = {"framework": {}, "fw_no_ledger": {"use_ledger": False}, "fw_no_gates": {"use_gates": False},
                "fw_no_learner": {"use_learner": False}, "fw_no_recal": {"use_recal": False}}
    for name, kw in variants.items():
        p, blocked, auth = C2.c2_scores(tr, te, ctx, fcfg, SEED, **kw)
        thr = FROZEN["fw_no_recal_thr"] if name == "fw_no_recal" else 0.5
        if kw.get("use_gates", True) is False:
            res["dec_" + name], res["sc_" + name] = p >= thr, p
        else:
            res["dec_" + name] = ~blocked & auth & (p >= thr)
            res["sc_" + name] = np.where(blocked, -1.0, np.where(auth, p, p - 1.0))
    rk = res.groupby(["prom", "ha", "D"])["sc_framework"].rank(ascending=False, method="first")
    k = res.groupby(["prom", "ha", "D"])["y"].transform("sum")
    res["dec_hindsight_topk"], res["sc_hindsight_topk"] = (rk <= k).values, res["sc_framework"]
    return res


def main():
    split = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if split == "check_slice":
        g = int(sys.argv[2])
        s = load_all(["2024-04", "2024-05", "2024-06", "2024-07", "2024-08"])
        s = s[~s["heldout"]]
        a, b = pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-31")
        tr = s[(s["D"] >= "2024-04-01") & (s["D"] < a) & (s["grp"] != g)].reset_index(drop=True)
        te = s[(s["D"] >= a) & (s["D"] <= b) & (s["grp"] == g)].reset_index(drop=True)
        ctx = s[(s["D"] >= a - pd.Timedelta(days=40)) & (s["D"] <= b) & (s["grp"] == g)].reset_index(drop=True)
        tag = f"check_slice_g{g}"
    else:
        if split == "R1":
            months, tr_a, a, b = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10"], "2024-05-01", "2024-09-02", "2024-10-31"
        elif split == "R2":
            months, tr_a, a, b = ["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"], "2024-08-01", "2024-12-02", "2025-01-24"
        else:
            raise SystemExit("unknown split")
        s = load_all(months)
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tr = s[(s["D"] >= tr_a) & (s["D"] < a - pd.Timedelta(days=1)) & (~s["heldout"])].reset_index(drop=True)
        tr = tr[tr["D"] < pd.Timestamp(months[4] + "-01")].reset_index(drop=True)
        te = s[(s["D"] >= a) & (s["D"] <= b)].reset_index(drop=True)
        ctx = s[(s["D"] >= a - pd.Timedelta(days=40)) & (s["D"] <= b)].reset_index(drop=True)
        tag = split
    res = score(tr, te, ctx)
    res.to_parquet(os.path.join(OUT, f"{tag}.parquet"))
    meta = {"split": tag, "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_auth": int(te["ha"].nunique()),
            "train_from": str(tr["D"].min().date()), "train_to": str(tr["D"].max().date()),
            "test_from": str(te["D"].min().date()), "test_to": str(te["D"].max().date()),
            "frozen": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()},
            "sec": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(OUT, f"{tag}_meta.json"), "w"), indent=1, default=str)
    print(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
