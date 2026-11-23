"""Evidence ledger: as-of reconstruction of planned street-works packages at daily cutoffs.

Builds, for each planned day D (Mon to Fri) and each planned works permit whose
state available at the cutoff (16:00 London on the previous working day)
proposes a start on D, one snapshot row with
  * the permit's own as-of record (status, dates, history counts),
  * sibling permits of the same works (phases, superseded permits),
  * same-street evidence (other permits, authority activities, section 58),
  * promoter workload on D,
  * the outcome (actual start by the end of D, from records up to D + 7 days).
Only records with event_time <= cutoff enter the features (availability time).

Usage (module): build_snapshots(months, d_from, d_to, out_path)
"""
from __future__ import annotations

import glob
import os

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "data", "processed", "events")
PLANNED = ("minor", "standard", "major")


def _files(topic, months):
    out = []
    for m in months:
        f = os.path.join(EV, f"{topic}_{m.replace('-', '_')}.parquet")
        if os.path.exists(f):
            out.append(f.replace("\\", "/"))
    return out


def connect(threads=4, mem="8GB"):
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET TimeZone='UTC';")
    return con


def load_events(con, months):
    files = _files("permit", months)
    if not files:
        raise FileNotFoundError("no permit parquet for months %s" % months)
    flist = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""
    CREATE OR REPLACE TABLE ev AS
    SELECT
      object_reference AS pref,
      work_reference_number AS wref,
      promoter_swa_code AS prom,
      highway_authority_swa_code AS ha,
      usrn,
      work_category_ref AS cat,
      traffic_management_type_ref AS tm,
      CAST(event_time AS TIMESTAMPTZ) AS et,
      CAST(event_reference AS BIGINT) AS eref,
      event_type AS etype,
      permit_status AS pstatus,
      work_status_ref AS wstatus,
      CAST(timezone('Europe/London', CAST(proposed_start_date AS TIMESTAMPTZ)) AS DATE) AS ps,
      CAST(timezone('Europe/London', CAST(proposed_end_date AS TIMESTAMPTZ)) AS DATE) AS pe,
      proposed_start_time IS NOT NULL AS has_time,
      timezone('Europe/London', CAST(actual_start_date_time AS TIMESTAMPTZ)) AS as_local,
      is_deemed = 'Yes' AS deemed,
      is_ttro_required AS ttro,
      is_traffic_sensitive AS tsens,
      collaboration_type_ref AS collab,
      close_footway_ref AS footway,
      works_location_type AS loc_type,
      activity_type AS act_type,
      road_category AS road_cat,
      permit_conditions AS conds,
      TRY_CAST(regexp_extract(works_location_coordinates, '([0-9.]+) ([0-9.]+)', 1) AS DOUBLE) AS x,
      TRY_CAST(regexp_extract(works_location_coordinates, '([0-9.]+) ([0-9.]+)', 2) AS DOUBLE) AS y_bng,
      hour(timezone('Europe/London', CAST(proposed_start_time AS TIMESTAMPTZ))) AS ps_hour,
      CAST(right(object_reference, 2) AS VARCHAR) AS psuffix
    FROM read_parquet([{flist}])
    WHERE object_type = 'PERMIT' OR object_type IS NULL
    """)
    # one row per (pref, eref); drop exact duplicates across monthly files
    con.execute("""
    CREATE OR REPLACE TABLE ev AS
    SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY pref, eref ORDER BY et) rn FROM ev) WHERE rn = 1
    """)
    con.execute("ALTER TABLE ev DROP COLUMN rn")
    # state intervals: each event's state is valid from et until the next event of the same permit
    con.execute("""
    CREATE OR REPLACE TABLE st AS
    SELECT *,
      lead(et) OVER w AS et_next,
      row_number() OVER w AS k,
      sum(CASE WHEN etype='PERMIT_ALTERATION_GRANTED' THEN 1 ELSE 0 END) OVER w AS n_alt,
      sum(CASE WHEN etype='PERMIT_REFUSED' THEN 1 ELSE 0 END) OVER w AS n_ref,
      sum(CASE WHEN etype='PERMIT_GRANTED' THEN 1 ELSE 0 END) OVER w AS n_grant,
      min(et) OVER w AS et_first,
      max(CASE WHEN etype='PERMIT_GRANTED' THEN et END) OVER w AS et_grant,
      max(CASE WHEN etype='PERMIT_ALTERATION_GRANTED' THEN et END) OVER w AS et_alt,
      lag(ps) OVER w AS ps_prev
    FROM ev
    WINDOW w AS (PARTITION BY pref ORDER BY et, eref ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)


def load_aux(con, months):
    act = _files("activity", months)
    s58 = _files("section_58", months)
    if act:
        flist = ", ".join(f"'{f}'" for f in act)
        con.execute(f"""
        CREATE OR REPLACE TABLE act AS
        SELECT object_reference AS aref, od_usrn AS usrn, od_highway_authority_swa_code AS ha,
          od_activity_type AS atype, od_traffic_management_type AS atm, od_cancelled='Yes' AS acanc,
          CAST(timezone('Europe/London', CAST(od_start_date AS TIMESTAMPTZ)) AS DATE) AS a_s,
          CAST(timezone('Europe/London', CAST(od_end_date AS TIMESTAMPTZ)) AS DATE) AS a_e,
          CAST(event_time AS TIMESTAMPTZ) AS et, CAST(event_reference AS BIGINT) AS eref
        FROM read_parquet([{flist}], union_by_name=true)
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE act (aref VARCHAR, usrn VARCHAR, ha VARCHAR, atype VARCHAR, atm VARCHAR, acanc BOOLEAN, a_s DATE, a_e DATE, et TIMESTAMPTZ, eref BIGINT)")
    if s58:
        flist = ", ".join(f"'{f}'" for f in s58)
        con.execute(f"""
        CREATE OR REPLACE TABLE s58 AS
        SELECT object_reference AS sref, od_usrn AS usrn, od_section_58_status AS sstatus,
          od_section_58_extent AS sextent,
          CAST(timezone('Europe/London', CAST(od_start_date AS TIMESTAMPTZ)) AS DATE) AS s_s,
          CAST(timezone('Europe/London', CAST(od_end_date AS TIMESTAMPTZ)) AS DATE) AS s_e,
          CAST(event_time AS TIMESTAMPTZ) AS et, CAST(event_reference AS BIGINT) AS eref
        FROM read_parquet([{flist}], union_by_name=true)
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE s58 (sref VARCHAR, usrn VARCHAR, sstatus VARCHAR, sextent VARCHAR, s_s DATE, s_e DATE, et TIMESTAMPTZ, eref BIGINT)")
    for t, key in (("act", "aref"), ("s58", "sref")):
        con.execute(f"""
        CREATE OR REPLACE TABLE {t} AS SELECT *, lead(et) OVER (PARTITION BY {key} ORDER BY et, eref) AS et_next FROM {t}
        """)


def make_days(con, d_from, d_to):
    con.execute(f"""
    CREATE OR REPLACE TABLE days AS
    WITH d AS (SELECT CAST(x AS DATE) AS D FROM range(DATE '{d_from}', DATE '{d_to}' + INTERVAL 1 DAY, INTERVAL 1 DAY) t(x))
    SELECT D,
      CAST(timezone('Europe/London', CAST(D - CASE WHEN dayofweek(D)=1 THEN 3 ELSE 1 END AS TIMESTAMP) + INTERVAL 16 HOUR) AS TIMESTAMPTZ) AS cut
    FROM d WHERE dayofweek(D) BETWEEN 1 AND 5
    """)


def build_snapshots(con):
    # candidate packages: planned category, state at cut proposes start D, works still planned
    con.execute(f"""
    CREATE OR REPLACE TABLE cand AS
    SELECT d.D, d.cut, s.*
    FROM days d JOIN st s ON s.ps = d.D AND s.et <= d.cut AND (s.et_next IS NULL OR s.et_next > d.cut)
    WHERE s.cat IN {PLANNED} AND s.wstatus = 'planned'
    """)
    # outcome: first non-reverted start known within D + 7 days; start date <= D counts as started
    con.execute("""
    CREATE OR REPLACE TABLE outc AS
    WITH last AS (
      SELECT c.pref, c.D, arg_max(s.as_local, s.et) AS as_local_last, arg_max(s.wstatus, s.et) AS wst_last,
        min(CASE WHEN s.as_local IS NOT NULL THEN s.et END) AS start_et
      FROM cand c JOIN st s ON s.pref = c.pref AND s.et <= CAST(c.D AS TIMESTAMPTZ) + INTERVAL 8 DAY
      GROUP BY c.pref, c.D)
    SELECT pref, D, as_local_last, wst_last, start_et,
      CASE WHEN as_local_last IS NOT NULL AND CAST(as_local_last AS DATE) <= D THEN 1 ELSE 0 END AS y,
      CASE WHEN as_local_last IS NOT NULL AND CAST(as_local_last AS DATE) <= D + 1 THEN 1 ELSE 0 END AS y1,
      CASE WHEN wst_last = 'cancelled' THEN 1 ELSE 0 END AS cancelled_later
    FROM last
    """)
    # sibling permits of the same works as of cut (other permit refs, latest state each)
    con.execute("""
    CREATE OR REPLACE TABLE sib AS
    WITH s AS (
      SELECT c.pref, c.D, o.pref AS opref, o.wstatus AS ow, o.pstatus AS op, o.et AS oet
      FROM cand c JOIN st o ON o.wref = c.wref AND o.pref <> c.pref
       AND o.et <= c.cut AND (o.et_next IS NULL OR o.et_next > c.cut))
    SELECT pref, D, count(*) AS sib_n,
      sum(CASE WHEN ow='in_progress' THEN 1 ELSE 0 END) AS sib_inprog,
      sum(CASE WHEN ow='completed' THEN 1 ELSE 0 END) AS sib_done,
      sum(CASE WHEN op IN ('refused','cancelled','revoked') OR ow='cancelled' THEN 1 ELSE 0 END) AS sib_failed,
      sum(CASE WHEN opref > pref THEN 1 ELSE 0 END) AS sib_newer,
      sum(CASE WHEN opref > pref AND ow='planned' THEN 1 ELSE 0 END) AS sib_newer_planned
    FROM s GROUP BY pref, D
    """)
    # same-street other permits overlapping D, state as of cut
    con.execute("""
    CREATE OR REPLACE TABLE street AS
    WITH o AS (
      SELECT c.pref, c.D, c.prom, o.prom AS oprom, o.wstatus AS ow, o.pstatus AS op, o.tm AS otm, o.cat AS ocat
      FROM cand c JOIN st o ON o.usrn = c.usrn AND o.wref <> c.wref
       AND o.et <= c.cut AND (o.et_next IS NULL OR o.et_next > c.cut)
       AND o.ps <= c.D AND o.pe >= c.D AND o.wstatus IN ('planned','in_progress'))
    SELECT pref, D, count(*) AS st_n,
      sum(CASE WHEN oprom <> prom THEN 1 ELSE 0 END) AS st_other_prom,
      sum(CASE WHEN ow='in_progress' THEN 1 ELSE 0 END) AS st_inprog,
      sum(CASE WHEN otm='road_closure' THEN 1 ELSE 0 END) AS st_rc,
      sum(CASE WHEN otm='road_closure' AND oprom <> prom AND (op='granted' OR ow='in_progress') THEN 1 ELSE 0 END) AS st_rc_other_granted,
      sum(CASE WHEN op='granted' OR ow='in_progress' THEN 1 ELSE 0 END) AS st_granted,
      sum(CASE WHEN ocat IN ('immediate_urgent','immediate_emergency') THEN 1 ELSE 0 END) AS st_immediate
    FROM o GROUP BY pref, D
    """)
    con.execute("""
    CREATE OR REPLACE TABLE astreet AS
    SELECT c.pref, c.D, count(*) AS act_n,
      sum(CASE WHEN a.atm='road_closure' THEN 1 ELSE 0 END) AS act_rc
    FROM cand c JOIN act a ON a.usrn = c.usrn AND a.et <= c.cut AND (a.et_next IS NULL OR a.et_next > c.cut)
      AND NOT a.acanc AND a.a_s <= c.D AND a.a_e >= c.D
    GROUP BY c.pref, c.D
    """)
    con.execute("""
    CREATE OR REPLACE TABLE sstreet AS
    SELECT c.pref, c.D, count(*) AS s58_n
    FROM cand c JOIN s58 s ON s.usrn = c.usrn AND s.et <= c.cut AND (s.et_next IS NULL OR s.et_next > c.cut)
      AND s.sstatus = 'in_force' AND s.s_s <= c.D AND s.s_e >= c.D
    GROUP BY c.pref, c.D
    """)
    # promoter's live execution state at the cutoff, same authority: works in progress,
    # overrunning works (in progress past their planned end), and packages planned for
    # the cutoff day with how many of them had started by the cutoff
    con.execute("""
    CREATE OR REPLACE TABLE live AS
    WITH dd AS (SELECT D, cut, CAST(timezone('Europe/London', cut) AS DATE) AS cday FROM days),
    x AS (
      SELECT dd.D, s.prom, s.ha, s.wstatus, s.ps, s.pe, s.as_local, dd.cday, s.cat
      FROM dd JOIN st s ON s.et <= dd.cut AND (s.et_next IS NULL OR s.et_next > dd.cut)
      WHERE s.wstatus = 'in_progress' OR (s.ps = dd.cday AND s.wstatus IN ('planned','in_progress','completed')))
    SELECT D, prom, ha,
      count(*) FILTER (WHERE wstatus='in_progress') AS lv_inprog,
      count(*) FILTER (WHERE wstatus='in_progress' AND pe < cday) AS lv_overrun,
      count(*) FILTER (WHERE ps = cday AND cat IN ('minor','standard','major')) AS lv_n_cday,
      count(*) FILTER (WHERE ps = cday AND cat IN ('minor','standard','major') AND wstatus IN ('in_progress','completed')
                       AND CAST(as_local AS DATE) <= cday) AS lv_s_cday
    FROM x GROUP BY D, prom, ha
    """)
    # promoter workload on D (candidates of the same promoter, same authority and all)
    con.execute("""
    CREATE OR REPLACE TABLE load AS
    SELECT pref, D,
      count(*) OVER (PARTITION BY prom, ha, D) AS load_ha,
      count(*) OVER (PARTITION BY prom, D) AS load_all,
      count(*) OVER (PARTITION BY usrn, D) AS load_usrn
    FROM cand
    """)
    con.execute("""
    CREATE OR REPLACE TABLE snap AS
    SELECT c.D, c.cut, c.pref, c.wref, c.prom, c.ha, c.usrn, c.cat, c.tm, c.pstatus, c.etype AS last_etype,
      c.deemed, c.ttro, c.tsens, c.collab, c.footway, c.loc_type, c.act_type, c.road_cat, c.has_time,
      c.x, c.y_bng, c.ps_hour, TRY_CAST(c.psuffix AS INTEGER) AS psuffix, c.conds,
      (c.pe - c.ps) AS dur_days, c.k AS n_events, c.n_alt, c.n_ref, c.n_grant,
      date_diff('hour', c.et_first, c.cut) AS h_since_first,
      date_diff('hour', c.et, c.cut) AS h_since_last,
      date_diff('hour', c.et_grant, c.cut) AS h_since_grant,
      date_diff('hour', c.et_alt, c.cut) AS h_since_alt,
      CASE WHEN c.ps_prev IS NOT NULL AND c.ps_prev <> c.ps THEN 1 ELSE 0 END AS date_changed,
      coalesce(s.sib_n,0) sib_n, coalesce(s.sib_inprog,0) sib_inprog, coalesce(s.sib_done,0) sib_done,
      coalesce(s.sib_failed,0) sib_failed, coalesce(s.sib_newer,0) sib_newer, coalesce(s.sib_newer_planned,0) sib_newer_planned,
      coalesce(t.st_n,0) st_n, coalesce(t.st_other_prom,0) st_other_prom, coalesce(t.st_inprog,0) st_inprog,
      coalesce(t.st_rc,0) st_rc, coalesce(t.st_rc_other_granted,0) st_rc_other_granted,
      coalesce(t.st_granted,0) st_granted, coalesce(t.st_immediate,0) st_immediate,
      coalesce(a.act_n,0) act_n, coalesce(a.act_rc,0) act_rc, coalesce(x.s58_n,0) s58_n,
      l.load_ha, l.load_all, l.load_usrn,
      coalesce(v.lv_inprog,0) lv_inprog, coalesce(v.lv_overrun,0) lv_overrun,
      coalesce(v.lv_n_cday,0) lv_n_cday, coalesce(v.lv_s_cday,0) lv_s_cday,
      o.y, o.y1, o.cancelled_later, o.start_et
    FROM cand c
    LEFT JOIN live v ON v.D = c.D AND v.prom = c.prom AND v.ha = c.ha
    LEFT JOIN sib s ON s.pref = c.pref AND s.D = c.D LEFT JOIN street t ON t.pref = c.pref AND t.D = c.D
    LEFT JOIN astreet a ON a.pref = c.pref AND a.D = c.D LEFT JOIN sstreet x ON x.pref = c.pref AND x.D = c.D
    LEFT JOIN load l ON l.pref = c.pref AND l.D = c.D LEFT JOIN outc o ON o.pref = c.pref AND o.D = c.D
    """)
    return con.execute("SELECT count(*) FROM snap").fetchone()[0]


def history_features(con, win_eb=56, win_kpi=91, win_cap=28):
    """Track-record features known at each cutoff, from earlier packages only.

    A package planned for day D' counts in the history of day D when D' is before the
    cutoff date; it counts as started only when its start record was entered before
    the cutoff (start_et <= cut). Windows are in calendar days before the cutoff date.
    """
    con.execute("""
    CREATE OR REPLACE TABLE hist_src AS
    SELECT prom, ha, D AS Dp, start_et, y FROM snap
    """)
    con.execute("""
    CREATE OR REPLACE TABLE tgt AS
    SELECT DISTINCT prom, ha, D, cut, CAST(timezone('Europe/London', cut) AS DATE) AS cday FROM snap
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_ph AS
    SELECT t.prom, t.ha, t.D,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb}) AS n_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.y = 1 AND h.start_et <= t.cut) AS s_ph,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_cap} AND h.y = 1 AND h.start_et <= t.cut) AS s_ph_cap
    FROM tgt t JOIN hist_src h ON h.prom = t.prom AND h.ha = t.ha
      AND h.Dp < t.cday AND h.Dp >= t.cday - {max(win_eb, win_cap)}
    GROUP BY t.prom, t.ha, t.D
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE tgt_p AS SELECT DISTINCT prom, D, cut, cday FROM tgt
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_p AS
    SELECT t.prom, t.D,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb}) AS n_p,
      count(*) FILTER (WHERE h.Dp >= t.cday - {win_eb} AND h.y = 1 AND h.start_et <= t.cut) AS s_p,
      count(*) AS n_p_kpi,
      count(*) FILTER (WHERE h.y = 1 AND h.start_et <= t.cut) AS s_p_kpi
    FROM tgt_p t JOIN hist_src h ON h.prom = t.prom AND h.Dp < t.cday AND h.Dp >= t.cday - {win_kpi}
    GROUP BY t.prom, t.D
    """)
    con.execute("""
    CREATE OR REPLACE TABLE tgt_g AS SELECT DISTINCT D, cut, cday FROM tgt
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE hist_g AS
    SELECT t.D, count(*) AS n_g, count(*) FILTER (WHERE h.y = 1 AND h.start_et <= t.cut) AS s_g
    FROM tgt_g t JOIN hist_src h ON h.Dp < t.cday AND h.Dp >= t.cday - {win_eb}
    GROUP BY t.D
    """)
    con.execute("""
    CREATE OR REPLACE TABLE snaph AS
    SELECT s.*, coalesce(a.n_ph,0) n_ph, coalesce(a.s_ph,0) s_ph, coalesce(a.s_ph_cap,0) s_ph_cap,
      coalesce(b.n_p,0) n_p, coalesce(b.s_p,0) s_p, coalesce(b.n_p_kpi,0) n_p_kpi, coalesce(b.s_p_kpi,0) s_p_kpi,
      g.n_g, g.s_g
    FROM snap s LEFT JOIN hist_ph a USING (prom, ha, D) LEFT JOIN hist_p b USING (prom, D) LEFT JOIN hist_g g USING (D)
    """)


def months_between(a, b):
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    out = []
    while (y, m) <= (y2, m2):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out
