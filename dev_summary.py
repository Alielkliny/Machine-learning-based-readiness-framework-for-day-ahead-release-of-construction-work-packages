"""Summarise a fold-rows file: pooled and per-fold metrics, promoter-day cluster bootstrap of the
framework's relative lead on crew-day loss over every comparator, Holm-adjusted p-values against the
10% margin and against zero.

Usage: python -P src/dev_summary.py results/dev_eval/fold_rows.parquet results/dev_eval/summary.json [split_name]
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RIVALS = ["naive_release_all", "practice_granted", "constraint_check_ldcc", "kpi_supplier_score",
          "logit_latest", "rf_latest"]
ABLATIONS_ALL = ["lgb_latest", "fw_no_ledger", "fw_no_gates", "fw_no_learner", "fw_no_recal"]
REF = ["hindsight_topk"]
B = 1000
SEED = 20261108
MARGIN = 0.10


def holm(pvals):
    names = list(pvals)
    order = sorted(names, key=lambda k: pvals[k])
    m = len(order)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def main():
    rows = pd.read_parquet(sys.argv[1]) if not sys.argv[1].count(",") else pd.concat([pd.read_parquet(f) for f in sys.argv[1].split(",")], ignore_index=True)
    out_path = sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "development"
    if len(sys.argv) > 4:
        rows = rows[rows["cut"] == sys.argv[4]].reset_index(drop=True)
    global ABLATIONS
    ABLATIONS = [a for a in ABLATIONS_ALL if "dec_" + a in rows.columns]
    methods = ["framework"] + RIVALS + ABLATIONS + REF
    y = rows["y"].values
    granted = rows["granted"].values.astype(bool)
    pooled = {}
    for mth in methods:
        d = rows["dec_" + mth].values.astype(bool)
        sc = rows["sc_" + mth].values.astype(float)
        n_rel = d.sum()
        pooled[mth] = {
            "crew_day_loss": float(100.0 * (np.sum(d & (y == 0)) + np.sum(~d & (y == 1))) / len(y)),
            "auc": float(roc_auc_score(y, sc)) if np.unique(sc).size > 1 else 0.5,
            "unsupported_release_rate": float(np.sum(d & ~granted) / n_rel) if n_rel else 0.0,
            "release_rate": float(d.mean()),
            "false_ready_rate": float(np.sum(d & (y == 0)) / n_rel) if n_rel else float("nan"),
            "missed_start_rate": float(np.sum(~d & (y == 1)) / max(1, np.sum(y == 1))),
        }
    # per fold
    per_fold = {}
    for (m, g), r in rows.groupby(["month", "grp"]):
        r = r.reset_index(drop=True)
        yy = r["y"].values
        per_fold[f"{m}_g{g}"] = {mth: float(100.0 * (np.sum(r["dec_" + mth].values & (yy == 0)) +
                                                     np.sum(~r["dec_" + mth].values & (yy == 1))) / len(yy))
                                 for mth in methods}
    # cluster bootstrap on promoter-day clusters
    cl = rows["prom"].astype(str) + "|" + rows["ha"].astype(str) + "|" + rows["D"].astype(str)
    codes, uniq = pd.factorize(cl)
    K = len(uniq)
    nk = np.bincount(codes, minlength=K).astype(np.float64)
    costk = {}
    for mth in methods:
        d = rows["dec_" + mth].values.astype(bool)
        e = (d & (y == 0)) | (~d & (y == 1))
        costk[mth] = np.bincount(codes, weights=e.astype(float), minlength=K)
    rng = np.random.default_rng(SEED)
    boot = {mth: np.empty(B) for mth in methods}
    for b in range(B):
        w = rng.poisson(1.0, K).astype(np.float64)
        tot = (w * nk).sum()
        for mth in methods:
            boot[mth][b] = 100.0 * (w * costk[mth]).sum() / tot
    for mth in methods:
        pooled[mth]["ci95_cost"] = [float(np.quantile(boot[mth], 0.025)), float(np.quantile(boot[mth], 0.975))]
    lead = {}
    p_margin, p_zero = {}, {}
    for r in RIVALS + ABLATIONS:
        rel = (boot[r] - boot["framework"]) / boot[r]
        point = (pooled[r]["crew_day_loss"] - pooled["framework"]["crew_day_loss"]) / pooled[r]["crew_day_loss"]
        lead[r] = {"lead_rel": float(point), "ci95": [float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))]}
        p_margin[r] = float((np.sum(rel <= MARGIN) + 1) / (B + 1))
        p_zero[r] = float((np.sum(rel <= 0.0) + 1) / (B + 1))
    hm = holm({k: p_margin[k] for k in RIVALS})
    hz = holm({k: p_zero[k] for k in RIVALS})
    for r in RIVALS:
        lead[r]["p_holm_margin"] = hm[r]
        lead[r]["p_holm_zero"] = hz[r]
        lead[r]["win_10pct"] = bool(lead[r]["lead_rel"] >= MARGIN and hm[r] < 0.05)
    for r in ABLATIONS:
        lead[r]["p_margin"] = p_margin[r]
        lead[r]["p_zero"] = p_zero[r]
    # secondary metrics: cluster-bootstrap p-values for AUC (B_AUC resamples) and unsupported release rate
    B_AUC = 200
    rng2 = np.random.default_rng(SEED + 1)
    unsup_num = {m: np.bincount(codes, weights=(rows["dec_" + m].values.astype(bool) & ~granted).astype(float), minlength=K)
                 for m in methods}
    unsup_den = {m: np.bincount(codes, weights=rows["dec_" + m].values.astype(float), minlength=K) for m in methods}
    auc_diff = {r: np.empty(B_AUC) for r in RIVALS + ABLATIONS}
    uns_diff = {r: np.empty(B_AUC) for r in RIVALS + ABLATIONS}
    for b in range(B_AUC):
        w = rng2.poisson(1.0, K).astype(np.float64)
        wr = w[codes]
        def wauc(m):
            sc = rows["sc_" + m].values.astype(float)
            return roc_auc_score(y, sc, sample_weight=wr) if np.unique(sc).size > 1 else 0.5
        def wuns(m):
            den = (w * unsup_den[m]).sum()
            return (w * unsup_num[m]).sum() / den if den > 0 else 0.0
        af, uf = wauc("framework"), wuns("framework")
        for r in RIVALS + ABLATIONS:
            auc_diff[r][b] = af - wauc(r)
            uns_diff[r][b] = wuns(r) - uf
    p_auc = {r: float((np.sum(auc_diff[r] <= 0) + 1) / (B_AUC + 1)) for r in RIVALS + ABLATIONS}
    p_uns = {r: float(min(1.0, 2 * min((np.sum(uns_diff[r] <= 0) + 1) / (B_AUC + 1), (np.sum(uns_diff[r] >= 0) + 1) / (B_AUC + 1))))
             for r in RIVALS + ABLATIONS}
    ha_auc, ha_uns = holm({k: p_auc[k] for k in RIVALS}), holm({k: p_uns[k] for k in RIVALS})
    for r in RIVALS:
        lead[r]["p_holm_auc"] = ha_auc[r]
        lead[r]["p_holm_unsupported"] = ha_uns[r]
    for r in ABLATIONS:
        lead[r]["p_auc"] = p_auc[r]
        lead[r]["p_unsupported"] = p_uns[r]
    summary = {"split": split, "n_rows": int(len(rows)), "n_clusters": int(K), "n_authorities": int(rows["ha"].nunique()),
               "start_rate": float(y.mean()), "pooled": pooled, "lead_vs": lead, "per_fold_crew_day_loss": per_fold,
               "bootstrap": {"B": B, "seed": SEED, "cluster": "promoter-authority-day", "margin": MARGIN}}
    json.dump(summary, open(out_path, "w"), indent=1)
    print(json.dumps({"pooled_cost": {k: round(v["crew_day_loss"], 3) for k, v in pooled.items()},
                      "lead": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "ci95"} | {"ci95": [round(x, 4) for x in v["ci95"]]} for k, v in lead.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
