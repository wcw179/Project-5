import os
import sys
import sqlite3
import shutil
import json
from datetime import datetime


SQL_PREVIEW_STATS = {
    "pre": {
        "total": "SELECT COUNT(*) FROM bars;",
        "per_symbol": "SELECT symbol, COUNT(*) FROM bars GROUP BY symbol;",
        "dup_slots": (
            """
            WITH s AS (
              SELECT symbol,
                     REPLACE(SUBSTR(time,1,19), 'T',' ') AS nt
              FROM bars
            )
            SELECT COUNT(*)
            FROM (
              SELECT symbol, nt, COUNT(*) AS c
              FROM s
              GROUP BY symbol, nt
              HAVING c>1
            ) t;
            """
        ),
    },
    "post": {
        "total": "SELECT COUNT(*) FROM bars;",
        "per_symbol": "SELECT symbol, COUNT(*) FROM bars GROUP BY symbol;",
        "off_grid": (
            """
            SELECT COUNT(*) FROM (
              SELECT REPLACE(SUBSTR(time,1,19), 'T',' ') AS nt FROM bars
            ) s
            WHERE CAST(strftime('%M', nt) AS INTEGER) % 5 != 0;
            """
        ),
        "dup_slots": (
            """
            WITH s AS (
              SELECT symbol,
                     REPLACE(SUBSTR(time,1,19), 'T',' ') AS nt
              FROM bars
            )
            SELECT COUNT(*)
            FROM (
              SELECT symbol, nt, COUNT(*) AS c
              FROM s
              GROUP BY symbol, nt
              HAVING c>1
            ) t;
            """
        ),
    },
}


def backup_db(db_path: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.dedupe-backup-{ts}{ext}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def fetchall_dicts(cur) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_stats(conn: sqlite3.Connection, phase: str):
    cur = conn.cursor()
    out = {"phase": phase}
    out["total"] = cur.execute(SQL_PREVIEW_STATS[phase]["total"]).fetchone()[0]
    out["per_symbol"] = fetchall_dicts(cur.execute(SQL_PREVIEW_STATS[phase]["per_symbol"]))
    if phase == "pre":
        out["dup_slots"] = cur.execute(SQL_PREVIEW_STATS[phase]["dup_slots"]).fetchone()[0]
    else:
        out["off_grid"] = cur.execute(SQL_PREVIEW_STATS[phase]["off_grid"]).fetchone()[0]
        out["dup_slots"] = cur.execute(SQL_PREVIEW_STATS[phase]["dup_slots"]).fetchone()[0]
    return out


def dedupe_normalize(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Create a temp table of winners per (symbol, normalized time)
    cur.execute("DROP TABLE IF EXISTS __winners__;")
    cur.execute(
        """
        CREATE TEMP TABLE __winners__ AS
        WITH base AS (
          SELECT rowid AS rid,
                 symbol,
                 time,
                 REPLACE(SUBSTR(time,1,19), 'T',' ') AS nt,
                 COALESCE(volume, 0.0) AS vol,
                 COALESCE(high, 0.0) - COALESCE(low, 0.0) AS range,
                 CASE WHEN INSTR(time, 'T') > 0 THEN 1 ELSE 0 END AS iso_pref
          FROM bars
        ), ranked AS (
          SELECT rid, symbol, nt,
                 ROW_NUMBER() OVER (
                   PARTITION BY symbol, nt
                   ORDER BY vol DESC, range DESC, iso_pref DESC, rid DESC
                 ) AS rn
          FROM base
        )
        SELECT b.rid, b.symbol, b.nt
        FROM base b
        JOIN ranked r ON r.rid = b.rid
        WHERE r.rn = 1;
        """
    )

    # Delete all non-winner rows (those whose (symbol, nt) appears in winners but row not in winners)
    # Do deletion first to avoid unique conflicts when updating time
    cur.execute(
        """
        DELETE FROM bars
        WHERE rowid NOT IN (SELECT rid FROM __winners__)
          AND EXISTS (
            SELECT 1
            FROM __winners__ w
            WHERE w.symbol = bars.symbol
              AND w.nt = REPLACE(SUBSTR(bars.time,1,19), 'T',' ')
          );
        """
    )

    # Normalize remaining times precisely to 'YYYY-MM-DD HH:MM:SS'
    cur.execute(
        """
        UPDATE bars
        SET time = (
          SELECT w.nt FROM __winners__ w
          WHERE w.rid = bars.rowid
        )
        WHERE rowid IN (SELECT rid FROM __winners__)
          AND time <> (
            SELECT w.nt FROM __winners__ w WHERE w.rid = bars.rowid
          );
        """
    )

    # Safety: ensure no duplicates remain after normalization
    # (Optional) Could add a UNIQUE INDEX(symbol, time) if not already enforced


def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    backup_path = backup_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF;")
        pre = get_stats(conn, "pre")

        conn.execute("BEGIN")
        dedupe_normalize(conn)
        conn.commit()

        post = get_stats(conn, "post")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "backup_path": backup_path,
                    "stats_pre": pre,
                    "stats_post": post,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(json.dumps({"status": "error", "message": str(e), "backup_path": backup_path}, ensure_ascii=False))
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/dedupe_normalize_bars.py <path_to_sqlite_db>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

