import os
import sys
import sqlite3
import json
from typing import Any, Dict


def check_symbol(conn: sqlite3.Connection, symbol: str, start: str, end_exclusive: str) -> Dict[str, Any]:
    cur = conn.cursor()
    out: Dict[str, Any] = {"symbol": symbol, "start": start, "end_exclusive": end_exclusive}

    # Total rows and distinct times
    cur.execute(
        """
        SELECT COUNT(*) AS n, COUNT(DISTINCT time) AS n_distinct
        FROM bars
        WHERE symbol=? AND time>=? AND time<?
        """,
        (symbol, start, end_exclusive),
    )
    row = cur.fetchone()
    out["total_rows"] = row[0]
    out["distinct_times"] = row[1]
    out["duplicate_rows"] = row[0] - row[1]

    # Duplicate times and their multiplicities
    cur.execute(
        """
        SELECT time, COUNT(*) AS c
        FROM bars
        WHERE symbol=? AND time>=? AND time<?
        GROUP BY time
        HAVING c>1
        ORDER BY c DESC, time ASC
        LIMIT 50
        """,
        (symbol, start, end_exclusive),
    )
    out["top_duplicate_times"] = [(t, c) for (t, c) in cur.fetchall()]

    # Sum of (count-1) across duplicate times
    cur.execute(
        """
        SELECT COALESCE(SUM(c-1), 0)
        FROM (
          SELECT COUNT(*) AS c
          FROM bars
          WHERE symbol=? AND time>=? AND time<?
          GROUP BY time
          HAVING COUNT(*)>1
        ) s
        """,
        (symbol, start, end_exclusive),
    )
    out["duplicate_rows_via_times"] = cur.fetchone()[0]

    # Exact duplicate rows (same time+OHLCV)
    cur.execute(
        """
        SELECT COALESCE(SUM(c-1), 0)
        FROM (
          SELECT COUNT(*) AS c
          FROM bars
          WHERE symbol=? AND time>=? AND time<?
          GROUP BY symbol, time, open, high, low, close, volume
          HAVING COUNT(*)>1
        ) s
        """,
        (symbol, start, end_exclusive),
    )
    out["exact_duplicate_rows"] = cur.fetchone()[0]

    # Sample rows for first 5 duplicate times
    cur.execute(
        """
        SELECT time
        FROM (
          SELECT time, COUNT(*) AS c
          FROM bars
          WHERE symbol=? AND time>=? AND time<?
          GROUP BY time
          HAVING COUNT(*)>1
          ORDER BY time ASC
          LIMIT 5
        ) t
        """,
        (symbol, start, end_exclusive),
    )
    dup_times = [r[0] for r in cur.fetchall()]

    samples = {}
    for t in dup_times:
        cur.execute(
            """
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND time=?
            ORDER BY rowid
            """,
            (symbol, t),
        )
        samples[t] = [
            {
                "time": r[0],
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in cur.fetchall()
        ]
    out["samples_for_duplicate_times"] = samples

    # Range sanity: min/max time
    cur.execute(
        """
        SELECT MIN(time), MAX(time)
        FROM bars
        WHERE symbol=?
        """,
        (symbol,),
    )
    tmin, tmax = cur.fetchone()
    out["global_time_min"] = tmin
    out["global_time_max"] = tmax

    return out


def main():
    if len(sys.argv) < 5:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": "Usage: python scripts/check_symbol_duplicates.py <db_path> <symbol> <start_YYYY-MM-DD> <end_YYYY-MM-DD> (end exclusive)",
                },
                ensure_ascii=False,
            )
        )
        return 1
    db_path, symbol, start_date, end_date = sys.argv[1:5]

    # normalize inputs
    start = start_date.strip() + " 00:00:00"
    # end exclusive next day midnight
    end_exclusive = end_date.strip() + " 00:00:00"

    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    conn = sqlite3.connect(db_path)
    try:
        res = check_symbol(conn, symbol, start, end_exclusive)
        print(json.dumps({"status": "ok", **res}, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

