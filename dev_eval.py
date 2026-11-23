"""Decisive development evaluation of design C (frozen protocol).

Authority-grouped rolling-origin folds inside the development authorities:
  * development authorities split into 3 groups by sha256("<swa>|dev-fold") mod 3;
  * scored months m in {2024-05, 2024-06, 2024-07};
  * fold (m, g): train on planned days in the four months before m, authorities not in g;
    score planned days in m, authorities in g (unseen in that fold's training).
Inner tuning per fold: inner-train = training rows before month m-1, inner-validation = month m-1
(training authorities); small fixed grids; the decision threshold of every score-based method is
chosen on inner validation by crew-day loss. Rivals get the identity setting (with or without
promoter and authority identity) as a grid option.
Writes results/dev_eval/fold_rows.parquet (per-row decisions and scores) and results/dev_eval/summary.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods as M  # noqa: E402
from splits import is_holdout_authority  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LED = os.path.join(ROOT, "data", "processed", "ledger")
OUT = os.path.join(ROOT, "results", "dev_eval")
SEED = 20261108
MONTHS = ["2024-07"]  # amended D31: one month, three authority groups (compute)
NGROUP = 3


def month_start(m):
    return pd.Timestamp(m + "-01")


def load(months):
    fr = []
    for m in months:
        f = os.path.join(LED, f"snap_{m.replace('-', '_')}.parquet")
        d = pd.read_parquet(f)
        d = d[d["L"] == 1]
        fr.append(d)
    s = pd.concat(fr, ignore_index=True)
    s = s[~s["ha"].map(is_holdout_authority)].reset_index(drop=True)
    s = M.add_derived(s)
    s["grp"] = s["ha"].map(lambda h: int(hashlib.sha256(f"{h}|dev-fold".encode()).hexdigest(), 16) % NGROUP)
    return s


def tune_threshold(p, y, grid=np.linspace(0.30, 0.70, 41)):
    best, bt = None, 0.5
    for t in grid:
        rel = p >= t
        c = np.sum(rel & (y == 0)) + np.sum(~rel & (y == 1))
        if best is None or c < best:
            best, bt = c, t
    return float(bt)


def cats_for(ids):
    base = [c for c in M.CAT_LATEST if c not in ("prom", "ha")]
    return base + (["prom", "ha"] if ids else [])


def fit_predict_sklearn(kind, cfg, tr, te, ids, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cats = cats_for(ids)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
                             ("num", StandardScaler(), M.NUM_LATEST)])
    if kind == "rf":
        mdl = RandomForestClassifier(n_estimators=cfg.get("n", 200), min_samples_leaf=cfg["leaf"],
                                     max_features=cfg["mf"], max_samples=0.5, n_jobs=3, random_state=seed)
    else:
        mdl = LogisticRegression(max_iter=3000, C=cfg["C"])
    pipe = make_pipeline(pre, mdl)
    pipe.fit(tr[cats + M.NUM_LATEST], tr["y"])
    return pipe.predict_proba(te[cats + M.NUM_LATEST])[:, 1]


def lgb_latest(tr, te, cfg, ids, seed):
    return M.policy_lgb(tr, te, M.NUM_LATEST, cats_for(ids), seed=seed,
                        params={"num_leaves": cfg["nl"], "min_child_samples": cfg["mc"]})[1]


def framework_scores(tr, te, cfg, seed, **kw):
    rel, score, state, gv, p = M.framework_v2(tr, te, use_recal=False, seed=seed, return_detail=True,
                                               params={"num_leaves": cfg["nl"], "min_child_samples": cfg["mc"]}, **kw)
    return rel, score, state, gv, p


GRIDS = {
    "rf": [{"leaf": l, "mf": mf, "n": 100} for l in (20, 100) for mf in ("sqrt", 0.3)],
    "logit": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "lgb": [{"nl": nl, "mc": mc} for nl in (31, 63) for mc in (100, 300)],
}


def run_fold(s, m, g, seed=SEED):
    ms = month_start(m)
    tr = s[(s["D"] >= ms - pd.DateOffset(months=4)) & (s["D"] < ms) & (s["grp"] != g)].reset_index(drop=True)
    te = s[(s["D"] >= ms) & (s["D"] < ms + pd.DateOffset(months=1)) & (s["grp"] == g)].reset_index(drop=True)
    itr = tr[tr["D"] < ms - pd.DateOffset(months=1)].reset_index(drop=True)
    iva = tr[tr["D"] >= ms - pd.DateOffset(months=1)].reset_index(drop=True)
    yv = iva["y"].values
    sel = {}
    t0 = time.time()

    def pick(name, fn, grid, id_options=(True, False)):
        best = None
        for ids in id_options:
            for cfg in grid:
                p = fn(itr, iva, cfg, ids)
                th = tune_threshold(p, yv)
                c = M.crew_day_loss(p >= th, yv)
                if best is None or c < best[0]:
                    best = (c, cfg, ids, th)
        sel[name] = {"inner_cost": best[0], "cfg": best[1], "ids": best[2], "thr": best[3]}
        return best

    sub = itr.sample(n=min(len(itr), 120000), random_state=seed) if len(itr) > 120000 else itr
    # random forest tuned on a subsample of inner-train for speed (100 trees), refit with 200 trees
    best_rf = None
    for ids in (True, False):
        for cfg in GRIDS["rf"]:
            p = fit_predict_sklearn("rf", cfg, sub, iva, ids, seed)
            th = tune_threshold(p, yv)
            c = M.crew_day_loss(p >= th, yv)
            if best_rf is None or c < best_rf[0]:
                best_rf = (c, cfg, ids, th)
    sel["rf_latest"] = {"inner_cost": best_rf[0], "cfg": best_rf[1], "ids": best_rf[2], "thr": best_rf[3]}
    pick("logit_latest", lambda a, b, cfg, ids: fit_predict_sklearn("logit", cfg, a, b, ids, seed), GRIDS["logit"])
    pick("lgb_latest", lambda a, b, cfg, ids: lgb_latest(a, b, cfg, ids, seed), GRIDS["lgb"])
    pick("framework", lambda a, b, cfg, ids: framework_scores(a, b, cfg, seed)[4], GRIDS["lgb"], id_options=(False,))

    y = te["y"].values
    gr = (te["pstatus"] == "granted").values
    res = pd.DataFrame({"pref": te["pref"], "D": te["D"], "prom": te["prom"], "ha": te["ha"], "y": y,
                        "granted": gr, "month": m, "grp": g})
    # rule-based and score-based rivals
    res["dec_naive_release_all"] = True
    res["sc_naive_release_all"] = 1.0
    r, p = M.policy_practice(tr, te)
    res["dec_practice_granted"], res["sc_practice_granted"] = r, p
    r, p = M.policy_constraint_check(tr, te)
    res["dec_constraint_check_ldcc"], res["sc_constraint_check_ldcc"] = r, p
    r, p = M.policy_kpi(tr, te)
    res["dec_kpi_supplier_score"], res["sc_kpi_supplier_score"] = r, p
    for name, kind in (("rf_latest", "rf"), ("logit_latest", "logit")):
        cfg = dict(sel[name]["cfg"])
        if kind == "rf":
            cfg["n"] = 200
        p = fit_predict_sklearn(kind, cfg, tr, te, sel[name]["ids"], seed)
        res["sc_" + name] = p
        res["dec_" + name] = p >= sel[name]["thr"]
    p = lgb_latest(tr, te, sel["lgb_latest"]["cfg"], sel["lgb_latest"]["ids"], seed)
    res["sc_lgb_latest"], res["dec_lgb_latest"] = p, p >= sel["lgb_latest"]["thr"]
    cfg = sel["framework"]["cfg"]
    thr = sel["framework"]["thr"]
    _, score, state, gv, pf = framework_scores(tr, te, cfg, seed)
    blocked = (gv[M.GATES] == M.U).any(axis=1).values
    auth = (gv["g_auth"] == M.S).values
    res["sc_framework"] = score
    res["dec_framework"] = ~blocked & auth & (pf >= thr)
    res["state_framework"] = np.where(blocked, "BLOCKED", np.where(res["dec_framework"], "READY", "INSUFFICIENT"))
    # ablations at the framework's tuned settings
    for name, kw in (("fw_no_ledger", {"use_ledger": False}), ("fw_no_gates", {"use_gates": False}),
                     ("fw_no_learner", {"use_learner": False})):
        _, sc, st, gvv, pp = framework_scores(tr, te, cfg, seed, **kw)
        if name == "fw_no_gates":
            d = pp >= thr
        elif name == "fw_no_ledger":
            d = (~(gvv["g_auth"] == M.U).values) & ((gvv["g_auth"] == M.S).values) & (pp >= thr)
        else:
            d = (~(gvv[M.GATES] == M.U).any(axis=1).values) & ((gvv["g_auth"] == M.S).values) & (pp >= thr)
        res["sc_" + name], res["dec_" + name] = sc, d
    # hindsight ceiling of the decision layer: top-k by framework score, k = actual starts per promoter-day
    rk = res.groupby(["prom", "ha", "D"])["sc_framework"].rank(ascending=False, method="first")
    k = res.groupby(["prom", "ha", "D"])["y"].transform("sum")
    res["dec_hindsight_topk"] = (rk <= k).values
    res["sc_hindsight_topk"] = res["sc_framework"]
    meta = {"month": m, "grp": g, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_auth": int(te["ha"].nunique()), "sel": sel, "sec": round(time.time() - t0, 1)}
    return res, meta


def main():
    os.makedirs(OUT, exist_ok=True)
    only_g = int(sys.argv[1]) if len(sys.argv) > 1 else None
    need = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07"]
    s = load(need)
    print("loaded", s.shape, flush=True)
    allres, metas = [], []
    done = set()
    tag = "" if only_g is None else f"_g{only_g}"
    part = os.path.join(OUT, f"fold_rows_partial{tag}.parquet")
    if os.path.exists(part):
        prev = pd.read_parquet(part)
        allres.append(prev)
        done = set(zip(prev["month"], prev["grp"]))
        metas = json.load(open(os.path.join(OUT, f"fold_meta_partial{tag}.json")))
    for m in MONTHS:
        for g in range(NGROUP):
            if only_g is not None and g != only_g:
                continue
            if (m, g) in done:
                continue
            res, meta = run_fold(s, m, g)
            allres.append(res)
            metas.append(meta)
            pd.concat(allres, ignore_index=True).to_parquet(part)
            json.dump(metas, open(os.path.join(OUT, f"fold_meta_partial{tag}.json"), "w"), indent=1, default=str)
            print(json.dumps({k: meta[k] for k in ("month", "grp", "n_train", "n_test", "sec")}), flush=True)
    rows = pd.concat(allres, ignore_index=True)
    rows.to_parquet(os.path.join(OUT, f"fold_rows{tag}.parquet"))
    json.dump(metas, open(os.path.join(OUT, f"fold_meta{tag}.json"), "w"), indent=1, default=str)
    print("done", rows.shape)


if __name__ == "__main__":
    main()
