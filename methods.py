"""Release policies scored on the same package snapshots.

Every policy returns, per snapshot row, a release decision (bool) and a readiness
score (float, higher = more likely to start). The proposed framework also returns
a three-state recommendation (READY, BLOCKED, INSUFFICIENT) and its gate vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

CAT_LATEST = ["cat", "tm", "ttro", "tsens", "collab", "footway", "loc_type",
              "act_type", "road_cat", "prom", "ha", "pstatus"]
NUM_LATEST = ["dur_days", "has_time", "deemed", "lead_days"]

# ledger features: reconciliation of the permit's own record history (supersession,
# temporal validity), sibling permits, same-street evidence and promoter workload
NUM_LEDGER = NUM_LATEST + [
    "n_events", "n_alt", "n_ref", "n_grant", "h_since_first", "h_since_last",
    "h_since_grant", "h_since_alt", "date_changed", "sib_n", "sib_inprog", "sib_done",
    "sib_failed", "sib_newer_planned", "st_n", "st_other_prom", "st_inprog",
    "st_rc_other_granted", "act_n", "act_rc", "s58_n", "load_ha",
    "load_all", "load_usrn", "dow", "lv_inprog", "lv_overrun", "lv_n_cday", "lv_s_cday",
    "lv_rate", "lv_backlog"]
NUM_POOL = ["eb_ph", "eb_p", "n_ph", "cap_ph", "load_ratio", "ebw_ph", "ebw_p"]

GATES = ["g_auth", "g_sched", "g_access", "g_pred", "g_res"]
S, U, Q = 1, -1, 0  # supported, unsupported (verified failure), unresolved


def add_derived(df, m1=20.0, m2=20.0, win_cap_days=20.0):
    d = df.copy()
    d["D"] = pd.to_datetime(d["D"])
    d["deemed"] = d["deemed"].astype(float)
    d["has_time"] = d["has_time"].astype(float)
    d["dow"] = pd.to_datetime(d["D"]).dt.dayofweek.astype(float)
    g = (d["s_g"] / d["n_g"].clip(lower=1)).fillna(0.5)
    d["eb_p"] = (d["s_p"] + m1 * g) / (d["n_p"] + m1)
    d["eb_ph"] = (d["s_ph"] + m2 * d["eb_p"]) / (d["n_ph"] + m2)
    gw = (d["sw_g"] / d["nw_g"].clip(lower=1)).fillna(0.6) if "sw_g" in d else g
    if "sw_p" in d:
        d["ebw_p"] = (d["sw_p"] + m1 * gw) / (d["nw_p"] + m1)
        d["ebw_ph"] = (d["sw_ph"] + m2 * d["ebw_p"]) / (d["nw_ph"] + m2)
    d["kpi_p"] = np.where(d["n_p_kpi"] > 0, d["s_p_kpi"] / d["n_p_kpi"].clip(lower=1), np.nan)
    # crew capacity proxy: known starts per working day in the trailing window
    nd = d["nd_cap"].clip(lower=1) if "nd_cap" in d else win_cap_days
    d["cap_ph"] = d["s_ph_cap"] / nd
    d["load_ratio"] = d["load_ha"] / np.maximum(d["cap_ph"], 0.5)
    # live adherence on the cutoff day and backlog of unstarted cutoff-day packages
    d["lv_rate"] = np.where(d["lv_n_cday"] > 0, d["lv_s_cday"] / d["lv_n_cday"].clip(lower=1), np.nan)
    d["lv_backlog"] = d["lv_n_cday"] - d["lv_s_cday"] + d["lv_overrun"]
    if "lead_days" not in d:
        d["lead_days"] = 1.0
    d["lead_days"] = d["lead_days"].astype(float)
    if "ps_hour" in d:
        d["ps_hour"] = d["ps_hour"].astype(float)
    for c in CAT_LATEST:
        d[c] = d[c].fillna("NA").astype(str)
    return d


def gate_vector(d, cap_tol=1.5):
    """Rule-based gates from the reconciled ledger (three-valued)."""
    gv = pd.DataFrame(index=d.index)
    ps = d["pstatus"]
    gv["g_auth"] = np.select([ps.isin(["granted"]) | ((ps == "submitted") & (d["deemed"] > 0)),
                              ps.isin(["refused", "revoked", "cancelled", "closed"])], [S, U], Q)
    # schedule validity: the planned date was changed by the latest record, or a
    # newer planned sibling permit of the same works supersedes this one
    gv["g_sched"] = np.select([d["sib_newer_planned"] > 0, d["date_changed"] > 0], [U, Q], S)
    # access: another promoter's granted or active road closure on the same street,
    # or a street restriction in force
    gv["g_access"] = np.select([d["s58_n"] > 0, (d["st_rc_other_granted"] > 0) | (d["act_rc"] > 0)], [U, Q], S)
    # predecessors: an earlier phase of the same works still in progress
    gv["g_pred"] = np.where(d["sib_inprog"] > 0, Q, S)
    # resources: promoter workload on D above its recent start capacity
    gv["g_res"] = np.where(d["load_ratio"] > cap_tol, Q, S)
    return gv


# ---------------------------------------------------------------- comparators

def policy_naive(tr, te):
    return np.ones(len(te), bool), np.ones(len(te))


def policy_practice(tr, te):
    rel = (te["pstatus"] == "granted").values
    return rel, rel.astype(float)


def policy_constraint_check(tr, te):
    """Structured constraint checking on the current linked records (LDCC-style):
    release only when every constraint holds; no temporal reasoning or uncertainty."""
    ok = ((te["pstatus"] == "granted") & (te["st_rc_other_granted"] == 0) & (te["act_rc"] == 0)
          & (te["s58_n"] == 0) & (te["sib_inprog"] == 0)).values
    return ok, ok.astype(float)


def _cost_threshold(score, y, c_fr=1.0, c_ud=1.0, grid=None):
    grid = np.unique(np.quantile(score[~np.isnan(score)], np.linspace(0, 1, 101))) if grid is None else grid
    best, bt = None, None
    for t in grid:
        rel = score >= t
        cost = c_fr * np.sum(rel & (y == 0)) + c_ud * np.sum(~rel & (y == 1))
        if best is None or cost < best:
            best, bt = cost, t
    return bt


def policy_kpi(tr, te, c_fr=1.0, c_ud=1.0):
    """Supplier-performance release: granted and promoter's trailing on-time start
    rate at or above a threshold chosen on training data."""
    k_tr = tr["kpi_p"].fillna(tr["kpi_p"].median()).values
    k_te = te["kpi_p"].fillna(tr["kpi_p"].median()).values
    g_tr = (tr["pstatus"] == "granted").values
    t = _cost_threshold(np.where(g_tr, k_tr, -1.0), tr["y"].values, c_fr, c_ud,
                        grid=np.linspace(0, 1, 101))
    score = np.where((te["pstatus"] == "granted").values, k_te, -1.0)
    return score >= t, score


def _latest_matrix(df):
    return df[CAT_LATEST + NUM_LATEST]


def _ohe_pipe(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT_LATEST),
        ("num", StandardScaler(), NUM_LATEST)])
    return make_pipeline(pre, model)


def policy_logit_latest(tr, te, c_fr=1.0, c_ud=1.0, seed=0):
    m = _ohe_pipe(LogisticRegression(max_iter=2000, C=1.0))
    m.fit(_latest_matrix(tr), tr["y"])
    p = m.predict_proba(_latest_matrix(te))[:, 1]
    return p >= c_fr / (c_fr + c_ud), p


def policy_rf_latest(tr, te, c_fr=1.0, c_ud=1.0, seed=0):
    m = _ohe_pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=3,
                                         random_state=seed, max_features="sqrt"))
    m.fit(_latest_matrix(tr), tr["y"])
    p = m.predict_proba(_latest_matrix(te))[:, 1]
    return p >= c_fr / (c_fr + c_ud), p


# ---------------------------------------------------------------- learned layer

def _lgb_frame(df, cols_num, cats):
    X = df[cats + cols_num].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X


def _fit_lgb(tr, cols_num, cats, seed=0, params=None):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=seed, verbose=-1, n_jobs=3, max_cat_to_onehot=8, cat_smooth=20)
    if params:
        p.update(params)
    Xtr = _lgb_frame(tr, cols_num, cats)
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr, tr["y"])
    return m, Xtr


def _predict_lgb(m, Xtr, te, cols_num, cats):
    Xte = _lgb_frame(te, cols_num, cats)
    for c in cats:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    return m.predict_proba(Xte)[:, 1]


def policy_lgb(tr, te, cols_num, cats, c_fr=1.0, c_ud=1.0, seed=0, params=None):
    m, Xtr = _fit_lgb(tr, cols_num, cats, seed, params)
    p = _predict_lgb(m, Xtr, te, cols_num, cats)
    return p >= c_fr / (c_fr + c_ud), p


# ---------------------------------------------------------------- proposed framework

def framework(tr, te, c_fr=1.0, c_ud=1.0, seed=0, use_ledger=True, use_learner=True,
              use_pool=True, use_gates=True, params=None, return_detail=False):
    """Evidence-reconciled readiness gating.

    1. gates from the reconciled ledger: any verified failure -> BLOCKED;
    2. learned readiness layer (gradient boosting on ledger + gate vector + pooled
       track record) gives a start probability for packages without a failure;
    3. READY when no gate is unsupported, the authorisation gate is supported and the
       probability reaches the cost-optimal threshold c_fr / (c_fr + c_ud);
       otherwise INSUFFICIENT (verification action on the unresolved gates).
    """
    cats = ["cat", "tm", "ttro", "tsens", "collab", "footway", "loc_type", "act_type",
            "road_cat", "pstatus", "prom", "ha"]
    num = list(NUM_LEDGER if use_ledger else NUM_LATEST)
    if use_pool:
        num += NUM_POOL
    gtr, gte = gate_vector(tr), gate_vector(te)
    trf, tef = tr.copy(), te.copy()
    if use_gates and use_ledger:
        for g in GATES:
            trf[g], tef[g] = gtr[g].values, gte[g].values
        num += GATES
    thr = c_fr / (c_fr + c_ud)
    if use_learner:
        m, Xtr = _fit_lgb(trf, num, cats, seed, params)
        p = _predict_lgb(m, Xtr, tef, num, cats)
    else:
        # rules only: probability from the pooled track record when every gate holds
        p = np.where((gte[GATES] == S).all(axis=1), 1.0, 0.0) if use_gates else np.full(len(te), 1.0)
    if use_gates:
        blocked = (gte[GATES] == U).any(axis=1).values
        if not use_ledger:
            blocked = (gte["g_auth"] == U).values
        auth_ok = (gte["g_auth"] == S).values
        ready = ~blocked & auth_ok & (p >= thr)
        state = np.where(blocked, "BLOCKED", np.where(ready, "READY", "INSUFFICIENT"))
        score = np.where(blocked, -1.0, np.where(auth_ok, p, p - 1.0))
    else:
        ready = p >= thr
        state = np.where(ready, "READY", "INSUFFICIENT")
        score = p
    if return_detail:
        return ready, score, state, gte
    return ready, score


# ---------------------------------------------------------------- metrics

def crew_day_loss(rel, y, c_fr=1.0, c_ud=1.0):
    """Crew-days lost per 100 candidate packages: released packages that did not
    start (wasted mobilisation) plus withheld packages that would have started."""
    rel = np.asarray(rel, bool)
    y = np.asarray(y)
    return 100.0 * (c_fr * np.sum(rel & (y == 0)) + c_ud * np.sum(~rel & (y == 1))) / len(y)


def decision_table(rel, y, granted):
    rel = np.asarray(rel, bool); y = np.asarray(y); granted = np.asarray(granted, bool)
    n_rel = rel.sum()
    return {
        "release_rate": rel.mean(),
        "false_ready_rate": (np.sum(rel & (y == 0)) / n_rel) if n_rel else np.nan,
        "missed_start_rate": np.sum(~rel & (y == 1)) / max(1, np.sum(y == 1)),
        "unsupported_release_rate": (np.sum(rel & ~granted) / n_rel) if n_rel else np.nan,
    }


# ---------------------------------------------------------------- framework v2 (design C)

CAT_NOID = ["cat", "tm", "ttro", "tsens", "collab", "footway", "loc_type", "act_type",
            "road_cat", "pstatus"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def online_offsets(ctx, p_ctx, targets, keys=("prom", "ha"), window=28, k=30.0):
    """Shrunk logit offset per target row from recent known outcomes of the same
    promoter and authority: rows planned in [cday - window, cday) whose start, if any,
    was recorded before the cutoff date. Uses only records available at the cutoff."""
    import duckdb
    h = ctx[list(keys) + ["D", "y", "start_et"]].copy()
    h["p"] = p_ctx
    h["sdate"] = pd.to_datetime(h["start_et"], utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    h["Dp"] = pd.to_datetime(h["D"])
    t = targets[list(keys) + ["D", "cut"]].copy().reset_index(drop=True)
    t["rid"] = np.arange(len(t))
    t["cday"] = pd.to_datetime(t["cut"], utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.register("h", h[list(keys) + ["Dp", "y", "sdate", "p"]])
    con.register("t", t[list(keys) + ["rid", "cday"]])
    kj = " AND ".join(f"h.{c} = t.{c}" for c in keys)
    r = con.execute(f"""
      SELECT t.rid, count(*) AS n, sum(h.p) AS sp,
        sum(CASE WHEN h.y = 1 AND h.sdate < t.cday THEN 1 ELSE 0 END) AS s
      FROM t JOIN h ON {kj} AND h.Dp < t.cday AND h.Dp >= t.cday - INTERVAL {window} DAY
      GROUP BY t.rid""").df()
    out = np.zeros(len(t))
    if len(r):
        q = (r["sp"] / r["n"]).clip(0.02, 0.98)
        a = _logit(((r["s"] + k * q) / (r["n"] + k)).values)
        b = _logit(((r["sp"] + k * q) / (r["n"] + k)).values)
        out[r["rid"].values] = a - b
    return out


def framework_v2(tr, te, ctx=None, c_fr=1.0, c_ud=1.0, seed=0, params=None,
                 use_ledger=True, use_gates=True, use_recal=True, use_learner=True,
                 use_ids=False, return_detail=False):
    """Evidence-reconciled readiness gating with online recalibration.

    Parts: (1) reconciled ledger and gate vector; (2) learned readiness layer on
    identity-free evidence (record history, track record known at the cutoff,
    workload, live adherence, gates); (3) online recalibration of the readiness
    probability from the promoter's recent known outcomes in the same authority.
    Decision: BLOCKED on any unsupported gate; READY when authorisation is supported
    and the recalibrated probability reaches c_fr / (c_fr + c_ud); else INSUFFICIENT.
    ctx: rows of the scored authorities before and during the scored period (history
    for recalibration); defaults to te.
    """
    cats = list(CAT_NOID) + (["prom", "ha"] if use_ids else [])
    num = list(NUM_LEDGER if use_ledger else NUM_LATEST) + (NUM_POOL if use_ledger else [])
    trf, tef = tr.copy(), te.copy()
    gtr, gte = gate_vector(tr), gate_vector(te)
    if use_gates and use_ledger:
        for g in GATES:
            trf[g], tef[g] = gtr[g].values, gte[g].values
        num += GATES
    thr = c_fr / (c_fr + c_ud)
    if use_learner:
        m, Xtr = _fit_lgb(trf, num, cats, seed, params)
        p = _predict_lgb(m, Xtr, tef, num, cats)
    else:
        # no learned layer: readiness from the shrunk track record only
        p = te["eb_ph"].values.copy()
    if use_recal:
        cx = te if ctx is None else ctx
        cxf = cx.copy()
        gcx = gate_vector(cx)
        if use_gates and use_ledger:
            for g in GATES:
                cxf[g] = gcx[g].values
        pc = _predict_lgb(m, Xtr, cxf, num, cats) if use_learner else cx["eb_ph"].values
        off = online_offsets(cx, pc, te)
        p = 1.0 / (1.0 + np.exp(-(_logit(p) + off)))
    if use_gates:
        blocked = (gte[GATES] == U).any(axis=1).values if use_ledger else (gte["g_auth"] == U).values
        auth_ok = (gte["g_auth"] == S).values
        ready = ~blocked & auth_ok & (p >= thr)
        state = np.where(blocked, "BLOCKED", np.where(ready, "READY", "INSUFFICIENT"))
        score = np.where(blocked, -1.0, np.where(auth_ok, p, p - 1.0))
    else:
        ready = p >= thr
        state = np.where(ready, "READY", "INSUFFICIENT")
        score = p
    if return_detail:
        return ready, score, state, gte, p
    return ready, score
