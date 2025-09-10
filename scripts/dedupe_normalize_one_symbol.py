import os
import sys
import sqlite3
import shutil
import json
from datetime import datetime
from typing import Dict, Any


def backup_db(db_path: str, tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.{tag}-{ts}{ext}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def get_symbol_stats(conn: sqlite3.Connection, symbol: str) -> Dict[str, Any]:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?;", (symbol,)).fetchone()[0]
    dup_slots = cur.execute(
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
    off_grid = cur.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT REPLACE(SUBSTR(time,1,19),'T',' ') AS nt
          FROM bars WHERE symbol=?
        ) s
        WHERE CAST(strftime('%M', nt) AS INTEGER) % 5 != 0
        """,
        (symbol,)
    ).fetchone()[0]
    return {"total": total, "dup_slots": dup_slots, "off_grid": off_grid}


def normalize_and_dedupe_symbol(conn: sqlite3.Connection, symbol: str) -> Dict[str, Any]:
    pre = get_symbol_stats(conn, symbol)
    cur = conn.cursor()

    # winners per (symbol, normalized time)
    cur.execute("DROP TABLE IF EXISTS __winners_one__;")
    cur.execute(
        """
        CREATE TEMP TABLE __winners_one__ AS
        WITH base AS (
          SELECT rowid AS rid,
                 symbol,
                 time,
                 REPLACE(SUBSTR(time,1,19), 'T',' ') AS nt,
                 COALESCE(volume, 0.0) AS vol,
                 (COALESCE(high, 0.0) - COALESCE(low, 0.0)) AS rng,
                 CASE WHEN INSTR(time, 'T') > 0 THEN 1 ELSE 0 END AS iso_pref
          FROM bars
          WHERE symbol = :sym
        ), ranked AS (
          SELECT rid, symbol, nt,
                 ROW_NUMBER() OVER (
                   PARTITION BY symbol, nt
                   ORDER BY vol DESC, rng DESC, iso_pref DESC, rid DESC
                 ) AS rn
          FROM base
        )
        SELECT b.rid, b.symbol, b.nt
        FROM base b
        JOIN ranked r ON r.rid = b.rid
        WHERE r.rn = 1;
        """,
        {"sym": symbol}
    )

    # delete non-winners for this symbol
    cur.execute(
        """
        DELETE FROM bars
        WHERE symbol = :sym
          AND rowid NOT IN (SELECT rid FROM __winners_one__)
          AND REPLACE(SUBSTR(time,1,19), 'T',' ') IN (SELECT nt FROM __winners_one__);
        """,
        {"sym": symbol}
    )

    # normalize time for winners
    cur.execute(
        """
        UPDATE bars
        SET time = (
          SELECT w.nt FROM __winners_one__ w WHERE w.rid = bars.rowid
        )
        WHERE symbol = :sym
          AND rowid IN (SELECT rid FROM __winners_one__)
          AND time <> (
            SELECT w.nt FROM __winners_one__ w WHERE w.rid = bars.rowid
          );
        """,
        {"sym": symbol}
    )

    post = get_symbol_stats(conn, symbol)
    removed = pre["total"] - post["total"]
    return {"symbol": symbol, "pre": pre, "post": post, "removed_rows": removed}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/dedupe_normalize_one_symbol.py <db_path> <symbol>"}, ensure_ascii=False))
        return 1
    db_path = sys.argv[1]
    symbol = sys.argv[2]

    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    backup_path = backup_db(db_path, f"dedupe-{symbol}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN")
        res = normalize_and_dedupe_symbol(conn, symbol)
        conn.commit()

        # Optional compact
        try:
            conn.execute("VACUUM;")
        except Exception:
            pass

        print(json.dumps({"status": "ok", "backup_path": backup_path, **res}, ensure_ascii=False, indent=2))
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

