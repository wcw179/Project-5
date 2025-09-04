import os
import sys
import sqlite3
import json


def main(db_path: str, symbol: str, start: str, end_excl: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Count groups with >1 rows per normalized time
        cur.execute(
            """
            WITH s AS (
              SELECT REPLACE(SUBSTR(time,1,19), 'T', ' ') AS nt,
                     open, high, low, close, volume
              FROM bars
              WHERE symbol=? AND time>=? AND time<?
            ), g AS (
              SELECT nt, COUNT(*) AS c,
                     COUNT(DISTINCT printf('%.10f|%.10f|%.10f|%.10f|%.10f',open,high,low,close,volume)) AS d_ohlcv
              FROM s
              GROUP BY nt
              HAVING c>1
            )
            SELECT COUNT(*) AS dup_slots,
                   SUM(CASE WHEN d_ohlcv=1 THEN 1 ELSE 0 END) AS identical_ohlcv_slots,
                   SUM(CASE WHEN d_ohlcv>1 THEN 1 ELSE 0 END) AS differing_ohlcv_slots
            FROM g
            """,
            (symbol, start + " 00:00:00", end_excl + " 00:00:00"),
        )
        dup_slots, ident_slots, diff_slots = cur.fetchone()
        print(json.dumps({
            "status": "ok",
            "symbol": symbol,
            "range": [start, end_excl],
            "duplicate_normalized_slots": dup_slots or 0,
            "identical_ohlcv_slots": ident_slots or 0,
            "differing_ohlcv_slots": diff_slots or 0,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/check_norm_dup_ohlcv.py <db_path> <symbol> <start_YYYY-MM-DD> <end_YYYY-MM-DD>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))

