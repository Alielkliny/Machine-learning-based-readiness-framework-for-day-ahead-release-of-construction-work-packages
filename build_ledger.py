"""Evidence ledger at daily cutoffs over a rolling look-ahead (leads 1 to 5 working days).

Stage 1 (base, lead 1 only, outcomes) and stage 2 (all leads, features) run month by
month so memory stays bounded. Each month uses permit events from three months before
to one month after (state and outcome lookahead) plus the authority activity and
section 58 records.

Outputs (data/processed/ledger/):
  base_YYYY_MM.parquet   one row per (permit, planned day D) at lead 1, with outcomes
  snap_YYYY_MM.parquet   one row per (permit, D, lead) with ledger and history features

Usage: python -P src/build_ledger.py base|snap YYYY-MM [YYYY-MM ...]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger as L  # noqa: E402

OUT = os.path.join(L.ROOT, "data", "processed", "ledger")
LEADS = (1, 2, 3, 4, 5)
WIN_DAYS = 6  # planned start interval: D to min(planned end, D + 6)


def month_shift(m, k):
    y, mm = map(int, m.split("-"))
    mm += k
    while mm > 12:
        y, mm = y + 1, mm - 12
    while mm < 1:
        y, mm = y - 1, mm + 12
    return f"{y:04d}-{mm:02d}"


def month_days(m):
    y, mm = map(int, m.split("-"))
    import calendar
    last = calendar.monthrange(y, mm)[1]
    return f"{y:04d}-{mm:02d}-01", f"{y:04d}-{mm:02d}-{last:02d}"


def make_days_leads(con, d_from, d_to, leads):
    lv = ", ".join(f"({x})" for x in leads)
    con.execute(f"""
    CREATE OR REPLACE TABLE days AS
    WITH d AS (SELECT CAST(x AS DATE) AS D FROM range(DATE '{d_from}', DATE '{d_to}' + INTERVAL 1 DAY, INTERVAL 1 DAY) t(x)),
    wd AS (SELECT CAST(x AS DATE) AS W FROM range(DATE '{d_from}' - INTERVAL 14 DAY, DATE '{d_to}' + INTERVAL 1 DAY, INTERVAL 1 DAY) t(x)
           WHERE dayofweek(CAST(x AS DATE)) BETWEEN 1 AND 5),
    lw AS (SELECT d.D, w.W, row_number() OVER (PARTITION BY d.D ORDER BY w.W DESC) AS L
           FROM d JOIN wd w ON w.W < d.D WHERE dayofweek(d.D) BETWEEN 1 AND 5)
    SELECT lw.D, lw.L, lw.W AS cday,
      CAST(timezone('Europe/London', CAST(lw.W AS TIMESTAMP) + INTERVAL 16 HOUR) AS TIMESTAMPTZ) AS cut
    FROM lw JOIN (VALUES {lv}) v(L) USING (L)
    """)


def build(con, with_features=True):
    P = L.PLANNED
    con.execute(f"""
    CREATE OR REPLACE TABLE cand AS
    SELECT d.D, d.L, d.cday, d.cut, s.*
    FROM days d JOIN st s ON s.ps = d.D AND s.et <= d.cut AND (s.et_next IS NULL OR s.et_next > d.cut)
    WHERE s.cat IN {P} AND s.wstatus = 'planned'
    """)
    # outcomes from the permit's records up to D + 14 days
    con.execute(f"""
    CREATE OR REPLACE TABLE outc AS
    WITH last AS (
      SELECT c.pref, c.D, c.L, arg_max(s.as_local, s.et) AS as_last, arg_max(s.wstatus, s.et) AS wst_last,
        min(CASE WHEN s.as_local IS NOT NULL THEN s.et END) AS start_et
      FROM cand c JOIN st s ON s.pref = c.pref AND s.et <= CAST(c.D AS TIMESTAMPTZ) + INTERVAL 15 DAY
      GROUP BY c.pref, c.D, c.L)
    SELECT l.pref, l.D, l.L, l.as_last, l.start_et,
      CASE WHEN l.as_last IS NOT NULL AND CAST(l.as_last AS DATE) <= l.D THEN 1 ELSE 0 END AS y,
      CASE WHEN l.as_last IS NOT NULL AND CAST(l.as_last AS DATE) <= least(c.pe, l.D + {WIN_DAYS}) THEN 1 ELSE 0 END AS yw,
      CASE WHEN l.wst_last = 'cancelled' THEN 1 ELSE 0 END AS cancelled_later
    FROM last l JOIN cand c ON c.pref = l.pref AND c.D = l.D AND c.L = l.L
    """)
    if not with_features:
        con.execute("""
        CREATE OR REPLACE TABLE out AS
        SELECT c.D, c.L, c.cut, c.pref, c.prom, c.ha, c.cat, c.pstatus, o.y, o.yw, o.cancelled_later, o.start_et, o.as_last
        FROM cand c JOIN outc o ON o.pref = c.pref AND o.D = c.D AND o.L = c.L
        """)
        return
    con.execute("""
    CREATE OR REPLACE TABLE sib AS
    WITH s AS (
      SELECT c.pref, c.D, c.L, o.pref AS opref, o.wstatus AS ow, o.pstatus AS op
      FROM cand c JOIN st o ON o.wref = c.wref AND o.pref <> c.pref
       AND o.et <= c.cut AND (o.et_next IS NULL OR o.et_next > c.cut))
    SELECT pref, D, L, count(*) AS sib_n,
      sum(CASE WHEN ow='in_progress' THEN 1 ELSE 0 END) AS sib_inprog,
      sum(CASE WHEN ow='completed' THEN 1 ELSE 0 END) AS sib_done,
      sum(CASE WHEN op IN ('refused','cancelled','revoked') OR ow='cancelled' THEN 1 ELSE 0 END) AS sib_failed,
      sum(CASE WHEN opref > pref AND ow='planned' THEN 1 ELSE 0 END) AS sib_newer_planned
    FROM s GROUP BY pref, D, L
    """)
    con.execute("""
    CREATE OR REPLACE TABLE street AS
    WITH o AS (
      SELECT c.pref, c.D, c.L, c.prom, o.prom AS oprom, o.wstatus AS ow, o.pstatus AS op, o.tm AS otm
      FROM cand c JOIN st o ON o.usrn = c.usrn AND o.wref <> c.wref
       AND o.et <= c.cut AND (o.et_next IS NULL OR o.et_next > c.cut)
       AND o.ps <= c.D AND o.pe >= c.D AND o.wstatus IN ('planned','in_progress'))
    SELECT pref, D, L, count(*) AS st_n,
      sum(CASE WHEN oprom <> prom THEN 1 ELSE 0 END) AS st_other_prom,
      sum(CASE WHEN ow='in_progress' THEN 1 ELSE 0 END) AS st_inprog,
      sum(CASE WHEN otm='road_closure' AND oprom <> prom AND (op='granted' OR ow='in_progress') THEN 1 ELSE 0 END) AS st_rc_other_granted
    FROM o GROUP BY pref, D, L
    """)
    con.execute("""
    CREATE OR REPLACE TABLE astreet AS
    SELECT c.pref, c.D, c.L, count(*) AS act_n, sum(CASE WHEN a.atm='road_closure' THEN 1 ELSE 0 END) AS act_rc
    FROM cand c JOIN act a ON a.usrn = c.usrn AND a.et <= c.cut AND (a.et_next IS NULL OR a.et_next > c.cut)
      AND NOT a.acanc AND a.a_s <= c.D AND a.a_e >= c.D
    GROUP BY c.pref, c.D, c.L
    """)
    con.execute("""
    CREATE OR REPLACE TABLE sstreet AS
    SELECT c.pref, c.D, c.L, count(*) AS s58_n
    FROM cand c JOIN s58 s ON s.usrn = c.usrn AND s.et <= c.cut AND (s.et_next IS NULL OR s.et_next > c.cut)
      AND s.sstatus = 'in_force' AND s.s_s <= c.D AND s.s_e >= c.D
    GROUP BY c.pref, c.D, c.L
    """)
    con.execute("""
    CREATE OR REPLACE TABLE live AS
    WITH cc AS (SELECT DISTINCT cday, cut FROM days),
    x AS (
      SELECT cc.cday, cc.cut, s.prom, s.ha, s.wstatus, s.ps, s.pe, s.as_local, s.cat
      FROM cc JOIN st s ON s.et <= cc.cut AND (s.et_next IS NULL OR s.et_next > cc.cut)
      WHERE s.wstatus = 'in_progress' OR (s.ps = cc.cday AND s.wstatus IN ('planned','in_progress','completed')))
    SELECT cday, prom, ha,
      count(*) FILTER (WHERE wstatus='in_progress') AS lv_inprog,
      count(*) FILTER (WHERE wstatus='in_progress' AND pe < cday) AS lv_overrun,
      count(*) FILTER (WHERE ps = cday AND cat IN ('minor','standard','major')) AS lv_n_cday,
      count(*) FILTER (WHERE ps = cday AND cat IN ('minor','standard','major') AND wstatus IN ('in_progress','completed')
                       AND CAST(as_local AS DATE) <= cday) AS lv_s_cday
    FROM x GROUP BY cday, prom, ha
    """)
    con.execute("""
    CREATE OR REPLACE TABLE load AS
    SELECT pref, D, L,
      count(*) OVER (PARTITION BY prom, ha, D, L) AS load_ha,
      count(*) OVER (PARTITION BY prom, D, L) AS load_all,
      count(*) OVER (PARTITION BY usrn, D, L) AS load_usrn
    FROM cand
    """)
    con.execute("""
    CREATE OR REPLACE TABLE out AS
    SELECT c.D, c.L, c.cday, c.cut, c.pref, c.wref, c.prom, c.ha, c.usrn, c.cat, c.tm, c.pstatus, c.etype AS last_etype,
      c.deemed, c.ttro, c.tsens, c.collab, c.footway, c.loc_type, c.act_type, c.road_cat, c.has_time,
      c.x, c.y_bng, c.ps_hour, TRY_CAST(c.psuffix AS INTEGER) AS psuffix,
      (c.pe - c.ps) AS dur_days, (c.D - c.cday) AS lead_days, c.k AS n_events, c.n_alt, c.n_ref, c.n_grant,
      date_diff('hour', c.et_first, c.cut) AS h_since_first,
      date_diff('hour', c.et, c.cut) AS h_since_last,
      date_diff('hour', c.et_grant, c.cut) AS h_since_grant,
      date_diff('hour', c.et_alt, c.cut) AS h_since_alt,
      CASE WHEN c.ps_prev IS NOT NULL AND c.ps_prev <> c.ps THEN 1 ELSE 0 END AS date_changed,
      coalesce(s.sib_n,0) sib_n, coalesce(s.sib_inprog,0) sib_inprog, coalesce(s.sib_done,0) sib_done,
      coalesce(s.sib_failed,0) sib_failed, coalesce(s.sib_newer_planned,0) sib_newer_planned,
      coalesce(t.st_n,0) st_n, coalesce(t.st_other_prom,0) st_other_prom, coalesce(t.st_inprog,0) st_inprog,
      coalesce(t.st_rc_other_granted,0) st_rc_other_granted,
      coalesce(a.act_n,0) act_n, coalesce(a.act_rc,0) act_rc, coalesce(x.s58_n,0) s58_n,
      l.load_ha, l.load_all, l.load_usrn,
      coalesce(v.lv_inprog,0) lv_inprog, coalesce(v.lv_overrun,0) lv_overrun,
      coalesce(v.lv_n_cday,0) lv_n_cday, coalesce(v.lv_s_cday,0) lv_s_cday,
      o.y, o.yw, o.cancelled_later, o.start_et, o.as_last
    FROM cand c
    LEFT JOIN live v ON v.cday = c.cday AND v.prom = c.prom AND v.ha = c.ha
    LEFT JOIN sib s ON s.pref = c.pref AND s.D = c.D AND s.L = c.L
    LEFT JOIN street t ON t.pref = c.pref AND t.D = c.D AND t.L = c.L
    LEFT JOIN astreet a ON a.pref = c.pref AND a.D = c.D AND a.L = c.L
    LEFT JOIN sstreet x ON x.pref = c.pref AND x.D = c.D AND x.L = c.L
    LEFT JOIN load l ON l.pref = c.pref AND l.D = c.D AND l.L = c.L
    LEFT JOIN outc o ON o.pref = c.pref AND o.D = c.D AND o.L = c.L
    """)


def history(con, base_glob, win_eb=56, win_kpi=91, win_cap=28):
    """Track record known at each cutoff, from lead-1 base rows of earlier planned days."""
    con.execute(f"""
    CREATE OR REPLACE TABLE hb AS
    SELECT prom, ha, D AS Dp, y, yw, start_et, as_last, cancelled_later FROM read_parquet('{base_glob}')
    """)
    con.execute("CREATE OR REPLACE TABLE tgt AS SELECT DISTINCT prom, ha, D, L, cday, cut FROM out")
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_ph AS
    SELECT t.prom, t.ha, t.D, t.L,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb}) AS n_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.y = 1 AND h.start_et <= t.cut) AS s_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.Dp + {WIN_DAYS} < t.cday) AS nw_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.Dp + {WIN_DAYS} < t.cday AND h.yw = 1 AND h.start_et <= t.cut) AS sw_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_cap} AND h.y = 1 AND h.start_et <= t.cut) AS s_ph_cap
    FROM tgt t JOIN hb h ON h.prom = t.prom AND h.ha = t.ha AND h.Dp < t.cday AND h.Dp >= t.cday - {max(win_eb, win_cap)}
    GROUP BY t.prom, t.ha, t.D, t.L
    """)
    con.execute("CREATE OR REPLACE TABLE tgt_p AS SELECT DISTINCT prom, D, L, cday, cut FROM tgt")
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_p AS
    SELECT t.prom, t.D, t.L,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb}) AS n_p,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.y = 1 AND h.start_et <= t.cut) AS s_p,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.Dp + {WIN_DAYS} < t.cday) AS nw_p,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.Dp + {WIN_DAYS} < t.cday AND h.yw = 1 AND h.start_et <= t.cut) AS sw_p,
      count(*) AS n_p_kpi,
      count(*) FILTER (WHERE h.y = 1 AND h.start_et <= t.cut) AS s_p_kpi
    FROM tgt_p t JOIN hb h ON h.prom = t.prom AND h.Dp < t.cday AND h.Dp >= t.cday - {win_kpi}
    GROUP BY t.prom, t.D, t.L
    """)
    con.execute("CREATE OR REPLACE TABLE tgt_g AS SELECT DISTINCT D, L, cday, cut FROM tgt")
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_g AS
    SELECT t.D, t.L, count(*) AS n_g, count(*) FILTER (WHERE h.y = 1 AND h.start_et <= t.cut) AS s_g,
      count(*) FILTER (WHERE h.Dp + {WIN_DAYS} < t.cday) AS nw_g,
      count(*) FILTER (WHERE h.Dp + {WIN_DAYS} < t.cday AND h.yw = 1 AND h.start_et <= t.cut) AS sw_g,
      count(DISTINCT h.Dp) FILTER (WHERE h.Dp >= t.cday - {win_cap}) AS nd_cap
    FROM tgt_g t JOIN hb h ON h.Dp < t.cday AND h.Dp >= t.cday - {win_eb}
    GROUP BY t.D, t.L
    """)
    con.execute("""
    CREATE OR REPLACE TABLE outh AS
    SELECT o.*, coalesce(a.n_ph,0) n_ph, coalesce(a.s_ph,0) s_ph, coalesce(a.nw_ph,0) nw_ph, coalesce(a.sw_ph,0) sw_ph,
      coalesce(a.s_ph_cap,0) s_ph_cap,
      coalesce(b.n_p,0) n_p, coalesce(b.s_p,0) s_p, coalesce(b.nw_p,0) nw_p, coalesce(b.sw_p,0) sw_p,
      coalesce(b.n_p_kpi,0) n_p_kpi, coalesce(b.s_p_kpi,0) s_p_kpi,
      coalesce(g.n_g,0) n_g, coalesce(g.s_g,0) s_g, coalesce(g.nw_g,0) nw_g, coalesce(g.sw_g,0) sw_g, coalesce(g.nd_cap,0) nd_cap
    FROM out o LEFT JOIN hist_ph a ON a.prom=o.prom AND a.ha=o.ha AND a.D=o.D AND a.L=o.L
    LEFT JOIN hist_p b ON b.prom=o.prom AND b.D=o.D AND b.L=o.L
    LEFT JOIN hist_g g ON g.D=o.D AND g.L=o.L
    """)


def run(stage, month):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ev_months = [month_shift(month, k) for k in (-3, -2, -1, 0, 1)]
    con = L.connect(threads=4, mem="8GB")
    L.load_events(con, ev_months)
    L.load_aux(con, ev_months)
    d0, d1 = month_days(month)
    tag = month.replace("-", "_")
    if stage == "base":
        make_days_leads(con, d0, d1, (1,))
        build(con, with_features=False)
        f = os.path.join(OUT, f"base_{tag}.parquet").replace("\\", "/")
        con.execute(f"COPY out TO '{f}' (FORMAT parquet)")
    else:
        make_days_leads(con, d0, d1, LEADS)
        build(con, with_features=True)
        history(con, os.path.join(OUT, "base_*.parquet").replace("\\", "/"))
        f = os.path.join(OUT, f"snap_{tag}.parquet").replace("\\", "/")
        con.execute(f"COPY outh TO '{f}' (FORMAT parquet)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{f}')").fetchone()[0]
    avail = [m for m in ev_months if os.path.exists(os.path.join(L.EV, f"permit_{m.replace('-', '_')}.parquet"))]
    print(f"{stage} {month} rows={n} event_months={avail} sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "snap1":
        LEADS = (1,)
        stage = "snap"
    for m in sys.argv[2:]:
        run(stage, m)
