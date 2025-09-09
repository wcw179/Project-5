import os
import sys
import sqlite3
import shutil
import json
from datetime import datetime


def backup_db(db_path: str, tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.{tag}-{ts}{ext}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/dedupe_normalize_one_symbol_fast.py <db_path> <symbol>"}, ensure_ascii=False))
        return 1
    db_path = sys.argv[1]
    symbol = sys.argv[2]

    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    backup_path = backup_db(db_path, f"dedupe-fast-{symbol}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # Pre-stats
        total_before = cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?;", (symbol,)).fetchone()[0]
        dup_slots_before = cur.execute(
            """
            WITH s AS (
              SELECT REPLACE(SUBSTR(time,1,19),'T',' ') AS nt
              FROM bars WHERE symbol=?
            )
            SELECT COUNT(*) FROM (
              SELECT nt, COUNT(*) AS c FROM s GROUP BY nt HAVING c>1
            )
            """,
            (symbol,)
        ).fetchone()[0]

        conn.execute("BEGIN")

        # 1) Winners for this symbol
        cur.execute("DROP TABLE IF EXISTS __winners_fast__;")
        cur.execute(
            """
            CREATE TEMP TABLE __winners_fast__ AS
            WITH base AS (
              SELECT rowid AS rid,
                     REPLACE(SUBSTR(time,1,19),'T',' ') AS nt,
                     COALESCE(volume, 0.0) AS vol,
                     (COALESCE(high, 0.0) - COALESCE(low, 0.0)) AS rng,
                     CASE WHEN INSTR(time, 'T') > 0 THEN 1 ELSE 0 END AS iso_pref
              FROM bars
              WHERE symbol = :sym
            ), ranked AS (
              SELECT rid, nt,
                     ROW_NUMBER() OVER (
                       PARTITION BY nt
                       ORDER BY vol DESC, rng DESC, iso_pref DESC, rid DESC
                     ) AS rn
              FROM base
            )
            SELECT rid, nt FROM ranked WHERE rn = 1;
            """,
            {"sym": symbol}
        )

        # 2) Materialize rows to keep (normalized time applied)
        cur.execute("DROP TABLE IF EXISTS __keep_rows__;")
        cur.execute(
            f"""
            CREATE TEMP TABLE __keep_rows__ AS
            SELECT '{symbol}' AS symbol,
                   w.nt AS time,
                   b.open, b.high, b.low, b.close, b.volume, b.spread
            FROM bars b
            JOIN __winners_fast__ w ON w.rid = b.rowid;
            """
        )

        # 3) Delete all rows for this symbol in bars
        cur.execute("DELETE FROM bars WHERE symbol=?;", (symbol,))

        # 4) Insert back the cleaned rows
        cur.execute(
            """
            INSERT INTO bars(symbol, time, open, high, low, close, volume, spread)
            SELECT symbol, time, open, high, low, close, volume, spread
            FROM __keep_rows__
            """
        )

        conn.commit()

        # Post-stats
        total_after = cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?;", (symbol,)).fetchone()[0]
        dup_slots_after = cur.execute(
            """
            WITH s AS (
              SELECT REPLACE(SUBSTR(time,1,19),'T',' ') AS nt
              FROM bars WHERE symbol=?
            )
            SELECT COUNT(*) FROM (
              SELECT nt, COUNT(*) AS c FROM s GROUP BY nt HAVING c>1
            )
            """,
            (symbol,)
        ).fetchone()[0]
        off_grid_after = cur.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT REPLACE(SUBSTR(time,1,19),'T',' ') AS nt
              FROM bars WHERE symbol=?
            ) s
            WHERE CAST(strftime('%M', nt) AS INTEGER) % 5 != 0
            """,
            (symbol,)
        ).fetchone()[0]

        # Vacuum for compaction
        try:
            conn.execute("VACUUM;")
        except Exception:
            pass

        print(json.dumps({
            "status": "ok",
            "backup_path": backup_path,
            "symbol": symbol,
            "total_before": total_before,
            "total_after": total_after,
            "removed_rows": total_before - total_after,
            "dup_slots_before": dup_slots_before,
            "dup_slots_after": dup_slots_after,
            "off_grid_after": off_grid_after
        }, ensure_ascii=False, indent=2))
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
    sys.exit(main())

