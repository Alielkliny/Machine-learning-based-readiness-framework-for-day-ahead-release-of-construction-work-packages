"""Additional analyses from saved per-row results and ledger snapshots (no refitting).
  1. Authority-level inference on the hold-out authorities: authority-cluster bootstrap of the pooled crew-day loss
     reduction of the framework against each rival (B = 2000), Holm adjustment over the six rivals within each test
     period, and sign and Wilcoxon signed-rank tests on the per-authority reductions.
  2. Promoter overlap: share of hold-out package-days whose promoter held packages in the development authorities
     in the training months of the test models.
  3. Gates: observed start rate by gate state on the hold-out authorities, and the gate cost (framework minus the
     ungated variant) split into packages without a granted permit, granted packages with a recorded failure and
     the remaining packages.
  4. Non-starts: hold-out packages that did not start on the planned day, split into started later within the planned
     start interval, cancelled, and no start recorded.
  5. BIM demonstration: constraint check on the as-planned schedule (release when every earlier task on the element
     and on its design hosts is scheduled to finish before the week).
Writes results/post/review_extra.json. Usage: python src/review_extra.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "src"))
import methods as M  # noqa: E402
from splits import is_holdout_authority  # noqa: E402

OUT = os.path.join(R, "results", "post", "review_extra.json")
RIVALS = ["naive_release_all", "practice_granted", "constraint_check_ldcc", "kpi_supplier_score", "logit_latest",
          "rf_latest"]
SPLITS = {"R1": {"score": ["2024_09", "2024_10"], "train": ["2024_05", "2024_06", "2024_07", "2024_08"]},
          "R2": {"score": ["2024_12", "2025_01"], "train": ["2024_08", "2024_09", "2024_10", "2024_11"]}}
GATE_COLS = ["pref", "D", "pstatus", "deemed", "sib_newer_planned", "date_changed", "s58_n", "st_rc_other_granted",
             "act_rc", "sib_inprog", "load_ha", "s_ph_cap", "nd_cap", "y", "yw", "cancelled_later", "start_et"]
B, SEED = 2000, 20261109


def snap(month, cols):
    return pd.read_parquet(os.path.join(R, "data", "processed", "ledger", f"snap_{month}.parquet"), columns=cols)


def loss_rows(d, m):
    r = d["dec_" + m].values.astype(bool)
    y = d["y"].values.astype(int)
    return (r & (y == 0)).astype(float) + (~r & (y == 1)).astype(float)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def authority_inference(d):
    ha = d["ha"].values
    auth = np.unique(ha)
    idx = np.searchsorted(auth, ha)
    n = np.bincount(idx, minlength=len(auth)).astype(float)
    L = {m: np.bincount(idx, weights=loss_rows(d, m), minlength=len(auth)) for m in ["framework"] + RIVALS}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(auth), size=(B, len(auth)))
    out = {"n_authorities": int(len(auth)), "B": B, "seed": SEED, "cluster": "highway authority"}
    ps = []
    for m in RIVALS:
        pooled = 1 - L["framework"].sum() / L[m].sum()
        boot = 1 - L["framework"][draws].sum(1) / L[m][draws].sum(1)
        p = (1 + np.sum(boot <= 0)) / (B + 1)
        per = 1 - L["framework"] / np.where(L[m] > 0, L[m], np.nan)
        diff = (L[m] - L["framework"]) / n
        pos = int(np.sum(diff > 0))
        nz = int(np.sum(diff != 0))
        out[m] = {"pooled_reduction": float(pooled), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                  "p_one_sided": float(p), "n_positive": pos, "n_nonzero": nz,
                  "sign_p_one_sided": float(stats.binomtest(pos, nz, 0.5, alternative="greater").pvalue),
                  "wilcoxon_p_one_sided": float(stats.wilcoxon(diff[diff != 0], alternative="greater").pvalue),
                  "median_authority_reduction": float(np.nanmedian(per))}
        ps.append(p)
    for m, a in zip(RIVALS, holm(np.array(ps))):
        out[m]["p_holm"] = float(a)
    out["framework_loss"] = float(100 * L["framework"].sum() / n.sum())
    return out


def main():
    res = {"note": "hold-out authorities only; cost ratio 1:1"}
    for split, cfg in SPLITS.items():
        d = pd.read_parquet(os.path.join(R, "results", "test", f"{split}.parquet"))
        d = d[d["cut"] == "heldout_authority"].reset_index(drop=True)
        r = {"n_rows": int(len(d)), "authority": authority_inference(d)}
        # 2. promoter overlap with the training rows of the test models
        tr = pd.concat([snap(m, ["prom", "ha"]) for m in cfg["train"]], ignore_index=True)
        tr = tr[~tr["ha"].map(is_holdout_authority)]
        dev_prom = set(tr["prom"].unique())
        r["promoter_overlap_share"] = float(d["prom"].isin(dev_prom).mean())
        r["promoter_overlap_promoters"] = [int(d["prom"].isin(dev_prom).sum()), int(len(d))]
        # 3. gates on the hold-out rows
        g = pd.concat([snap(m, GATE_COLS) for m in cfg["score"]], ignore_index=True)
        g["D"] = pd.to_datetime(g["D"])
        d["D"] = pd.to_datetime(d["D"])
        g = g.drop_duplicates(["pref", "D"])
        j = d.merge(g, on=["pref", "D"], how="left", suffixes=("", "_s"))
        r["join_rate"] = float(j["pstatus"].notna().mean())
        r["y_agreement"] = float((j["y"] == j["y_s"]).mean())
        j["cap_ph"] = j["s_ph_cap"] / j["nd_cap"].clip(lower=1)
        j["load_ratio"] = j["load_ha"] / np.maximum(j["cap_ph"], 0.5)
        j["deemed"] = j["deemed"].astype(float)
        gv = M.gate_vector(j)
        anyF = (gv[M.GATES] == M.U).any(axis=1)
        authS = gv["g_auth"] == M.S
        groups = {"granted_no_failure": authS & ~anyF,
                  "granted_access_failure": authS & (gv["g_access"] == M.U),
                  "granted_schedule_failure": authS & (gv["g_sched"] == M.U),
                  "not_granted": ~authS}
        r["start_rate_by_gate"] = {k: {"n": int(v.sum()), "start_rate": float(j.loc[v, "y"].mean()) if v.sum() else None}
                                   for k, v in groups.items()}
        lf, lg = loss_rows(j, "framework"), loss_rows(j, "fw_no_gates")
        cats = {"no_granted_permit": ~authS, "granted_with_recorded_failure": authS & anyF, "other": authS & ~anyF}
        r["gate_cost"] = {"total": float(100 * (lf.sum() - lg.sum()) / len(j))}
        for k, v in cats.items():
            v = v.values
            r["gate_cost"][k] = float(100 * (lf[v].sum() - lg[v].sum()) / len(j))
        blocked = anyF.values
        reasons = {"authorization_failure": (gv["g_auth"] == M.U).values,
                   "restriction_only": blocked & (gv["g_auth"] != M.U).values & (gv["g_access"] == M.U).values
                   & (gv["g_sched"] != M.U).values,
                   "schedule_failure": blocked & (gv["g_auth"] != M.U).values & (gv["g_sched"] == M.U).values}
        r["blocked_split"] = {"n_blocked": int(blocked.sum())}
        for k, v in reasons.items():
            r["blocked_split"][k] = {"n": int(v.sum()), "share_of_blocked": float(v.sum() / max(blocked.sum(), 1)),
                                     "start_rate": float(j.loc[v, "y"].mean()) if v.sum() else None,
                                     "share_of_blocked_starts": float(j.loc[v, "y"].sum() / max(j.loc[blocked, "y"].sum(), 1))}
        ng_rel = j["dec_fw_no_gates"].astype(bool)
        r["gate_cost"]["ungated_releases_without_grant"] = int((ng_rel & ~authS).sum())
        r["gate_cost"]["ungated_releases_without_grant_started"] = float(j.loc[ng_rel & ~authS, "y"].mean())
        # 4. non-starts of granted packages
        ns = j[(j["y"] == 0) & authS]
        r["non_starts_granted"] = {"n": int(len(ns)),
                                   "started_later_in_interval": float((ns["yw"] == 1).mean()),
                                   "cancelled": float(((ns["yw"] == 0) & (ns["cancelled_later"] == 1)).mean()),
                                   "no_start_recorded": float(((ns["yw"] == 0) & (ns["cancelled_later"] == 0) & ns["start_et"].isna()).mean()),
                                   "started_after_interval": float(((ns["yw"] == 0) & (ns["cancelled_later"] == 0) & ns["start_et"].notna()).mean())}
        res[split] = r
        print(split, json.dumps({k: v for k, v in r.items() if k != "authority"}, indent=0)[:1500])
        print(split, "authority", {m: (round(r["authority"][m]["pooled_reduction"], 4), round(r["authority"][m]["p_holm"], 4),
                                       r["authority"][m]["n_positive"]) for m in RIVALS})
    # 4b. promoter overlap on the development folds (same definition as the test periods)
    import hashlib
    rows = pd.read_parquet(os.path.join(R, "results", "dev_c2", "fold_rows_c2.parquet"), columns=["prom", "ha", "grp"])
    tr = pd.concat([snap(m, ["prom", "ha", "D"]) for m in ["2024_03", "2024_04", "2024_05", "2024_06"]], ignore_index=True)
    tr = tr[~tr["ha"].map(is_holdout_authority)]
    tr["grp"] = tr["ha"].map(lambda h: int(hashlib.sha256(f"{h}|dev-fold".encode()).hexdigest(), 16) % 3)
    hit = 0
    for g in range(3):
        seen = set(tr.loc[tr["grp"] != g, "prom"])
        hit += int(rows.loc[rows["grp"] == g, "prom"].isin(seen).sum())
    res["dev_promoter_overlap_share"] = hit / len(rows)
    print("dev promoter overlap", round(res["dev_promoter_overlap_share"], 4))
    # 5. BIM constraint check on the as-planned schedule
    L = pd.read_parquet(os.path.join(R, "results", "bim_demo", "ledger.parquet"))
    cov = L[L["site_gate"] != "?"]
    raw = os.path.join(R, "data", "raw", "schependomlaan")
    ev = pd.read_csv(os.path.join(raw, "Event log", "Eventlog with As Built information",
                                  "Schependomlaan eventlog as planned data incl as built data.csv"))
    ev = ev[ev["Element on time or not"].notna()].copy()
    ev["s"] = pd.to_datetime(ev.TaskStart, dayfirst=True)
    ev["f"] = pd.to_datetime(ev.TaskFinish, dayfirst=True)
    ev["wk"] = ev.s.dt.isocalendar().week.astype(int)
    ev["fwk"] = ev.f.dt.isocalendar().week.astype(int)
    pro = os.path.join(R, "data", "processed", "schependomlaan")
    cw = pd.read_parquet(os.path.join(pro, "crosswalk.parquet"))
    rel = pd.read_parquet(os.path.join(pro, "design_relations.parquet"))
    op2wall = dict(zip(rel[rel.rel == "voids"].hosted, rel[rel.rel == "voids"].host))
    host = {}
    for h, o in zip(rel[rel.rel == "fills"].hosted, rel[rel.rel == "fills"].host):
        if o in op2wall:
            host.setdefault(h, set()).add(op2wall[o])
    for a, b in zip(rel[rel.rel == "connects"].hosted, rel[rel.rel == "connects"].host):
        host.setdefault(a, set()).add(b)
    p2d, d2p = dict(zip(cw.guid, cw.design_guid)), dict(zip(cw.design_guid, cw.guid))
    g2n = dict(zip(ev.GUID, ev.Resource))
    n2g = dict(zip(ev.Resource, ev.GUID))
    tasks_of = {k: v[["s", "fwk"]].values.tolist() for k, v in ev.groupby("Resource")}
    start_of = dict(zip(zip(ev.Resource, ev.TaskID), ev.s))
    # schedule constraint check: the finish-to-start predecessors that the 4D schedule records (IfcRelSequence in the
    # as-planned IFC) must be planned to finish before the week of the element-task
    import ifcopenshell
    f_ifc = ifcopenshell.open(os.path.join(raw, "As Planned models", "IFC Schependomlaan incl planningsdata.ifc"))
    preds_of = {}
    for rs in f_ifc.by_type("IfcRelSequence"):
        preds_of.setdefault(rs.RelatedProcess.TaskId, set()).add(rs.RelatingProcess.TaskId)
    task_fin = ev.groupby("TaskID")["f"].max().to_dict()
    task_start = ev.groupby("TaskID")["s"].min().to_dict()
    released = []
    for _, row in cov.iterrows():
        pr = [p for p in preds_of.get(row.task, set()) if p in task_fin]
        released.append(all(task_fin[p] <= task_start[row.task] for p in pr))
    cov = cov.assign(cc_release=released)
    both = [(a, b) for b, ps in preds_of.items() for a in ps if a in task_fin and b in task_start]
    res["bim_sequence_links"] = {"links": len(f_ifc.by_type("IfcRelSequence")), "tasks_with_predecessor": len(preds_of),
                                 "links_in_event_log": len(both),
                                 "links_satisfied_by_plan": int(sum(task_fin[a] <= task_start[b] for a, b in both)),
                                 "covered_with_predecessor": int(cov.task.map(lambda t: t in preds_of).sum())}
    res["bim_constraint_check"] = {"released": int(cov.cc_release.sum()), "released_late": int((cov.cc_release & (cov.late == 1)).sum()),
                                   "covered": int(len(cov)), "covered_late": int(cov.late.sum()),
                                   "held_on_time": int((~cov.cc_release & (cov.late == 0)).sum()),
                                   "held_late": int((~cov.cc_release & (cov.late == 1)).sum())}
    fw_rel = cov.state == "READY"
    rules = {"schedule_only": np.ones(len(cov), bool), "constraint_check": cov.cc_release.values, "framework": fw_rel.values}
    res["bim_by_week"] = {int(w): {k: {"released_on_time": int((r[m] & (g.late.values == 0)).sum()),
                                       "released_late": int((r[m] & (g.late.values == 1)).sum()),
                                       "held_late": int((~r[m] & (g.late.values == 1)).sum()),
                                       "held_on_time": int((~r[m] & (g.late.values == 0)).sum())}
                                   for k, r in rules.items()}
                          for w, g in cov.groupby("week") for m in [cov.index.get_indexer(g.index)]}
    res["bim_loss_per_100"] = {k: float(100 * ((r & (cov.late.values == 1)).sum() + (~r & (cov.late.values == 0)).sum()) / len(cov))
                               for k, r in rules.items()}
    print("BIM", res["bim_constraint_check"], res["bim_loss_per_100"])
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
