import os
import sys
import sqlite3
import shutil
import json
from datetime import datetime
from typing import Dict, Any, List


def backup_db(db_path: str, tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.{tag}-{ts}{ext}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def fetch_all_symbols(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol;")
    return [r[0] for r in cur.fetchall()]


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

    # Work in a temp winners table for this symbol
    cur.execute("DROP TABLE IF EXISTS __winners_sym__;")
    cur.execute(
        """
        CREATE TEMP TABLE __winners_sym__ AS
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

    # Delete non-winners in duplicate slots for this symbol
    cur.execute(
        """
        DELETE FROM bars
        WHERE symbol = :sym
          AND rowid NOT IN (SELECT rid FROM __winners_sym__)
          AND REPLACE(SUBSTR(time,1,19), 'T',' ') IN (SELECT nt FROM __winners_sym__);
        """,
        {"sym": symbol}
    )

    # Normalize time for winners
    cur.execute(
        """
        UPDATE bars
        SET time = (
          SELECT w.nt FROM __winners_sym__ w WHERE w.rid = bars.rowid
        )
        WHERE symbol = :sym
          AND rowid IN (SELECT rid FROM __winners_sym__)
          AND time <> (
            SELECT w.nt FROM __winners_sym__ w WHERE w.rid = bars.rowid
          );
        """,
        {"sym": symbol}
    )

    post = get_symbol_stats(conn, symbol)

    removed = pre["total"] - post["total"]
    return {"symbol": symbol, "pre": pre, "post": post, "removed_rows": removed}


def final_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM bars;").fetchone()[0]
    per_symbol = cur.execute("SELECT symbol, COUNT(*) FROM bars GROUP BY symbol ORDER BY symbol;").fetchall()
    dup_slots = cur.execute(
        """
        WITH s AS (
          SELECT symbol, REPLACE(SUBSTR(time,1,19),'T',' ') AS nt FROM bars
        )
        SELECT COUNT(*) FROM (
          SELECT symbol, nt, COUNT(*) AS c FROM s GROUP BY symbol, nt HAVING c>1
        );
        """
    ).fetchone()[0]
    off_grid = cur.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT REPLACE(SUBSTR(time,1,19),'T',' ') AS nt FROM bars
        ) s
        WHERE CAST(strftime('%M', nt) AS INTEGER) % 5 != 0;
        """
    ).fetchone()[0]
    return {
        "total": total,
        "per_symbol": [{"symbol": s, "count": n} for (s, n) in per_symbol],
        "dup_slots": dup_slots,
        "off_grid": off_grid,
    }


def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    # Backup before running
    backup_path = backup_db(db_path, "dedupe-progress-backup")

    conn = sqlite3.connect(db_path)
    try:
        # Ensure any previous crashed transaction is resolved automatically by SQLite
        # Begin per-symbol processing
        symbols = fetch_all_symbols(conn)

        progress: List[Dict[str, Any]] = []
        for sym in symbols:
            conn.execute("BEGIN")
            try:
                res = normalize_and_dedupe_symbol(conn, sym)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(json.dumps({"status": "error", "symbol": sym, "message": str(e)}, ensure_ascii=False))
                return 1
            progress.append(res)
            # Emit progress after each symbol
            print(json.dumps({"status": "progress", **res}, ensure_ascii=False))

        # Optional integrity check
        try:
            ic = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        except Exception:
            ic = "skipped"

        # Vacuum to compact
        try:
            conn.execute("VACUUM;")
        except Exception:
            pass

        summary = final_stats(conn)
        print(json.dumps({
            "status": "ok",
            "backup_path": backup_path,
            "integrity_check": ic,
            "summary": summary,
            "by_symbol": progress
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/dedupe_normalize_bars_progress.py <path_to_sqlite_db>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

