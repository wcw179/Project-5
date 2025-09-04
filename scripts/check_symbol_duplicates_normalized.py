import os
import sys
import sqlite3
import json


def main(db_path: str, symbol: str, start_date: str, end_date_exclusive: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    # Normalize inputs to midnight boundaries with space separator
    start = start_date.strip() + " 00:00:00"
    end_excl = end_date_exclusive.strip() + " 00:00:00"

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # Total rows
        cur.execute(
            """
            SELECT COUNT(*)
            FROM bars
            WHERE symbol=? AND time>=? AND time<?
            """,
            (symbol, start, end_excl),
        )
        total_rows = cur.fetchone()[0]

        # Distinct raw times
        cur.execute(
            """
            SELECT COUNT(DISTINCT time)
            FROM bars
            WHERE symbol=? AND time>=? AND time<?
            """,
            (symbol, start, end_excl),
        )
        distinct_raw = cur.fetchone()[0]

        # Distinct normalized times (replace T with space, drop timezone suffix)
        cur.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT DISTINCT REPLACE(SUBSTR(time,1,19), 'T', ' ') AS nt
              FROM bars
              WHERE symbol=? AND time>=? AND time<?
            ) s
            """,
            (symbol, start, end_excl),
        )
        distinct_norm = cur.fetchone()[0]

        # Duplicates by normalized time
        cur.execute(
            """
            SELECT COALESCE(SUM(c-1),0)
            FROM (
              SELECT REPLACE(SUBSTR(time,1,19), 'T', ' ') AS nt, COUNT(*) AS c
              FROM bars
              WHERE symbol=? AND time>=? AND time<?
              GROUP BY nt
              HAVING COUNT(*)>1
            ) t
            """,
            (symbol, start, end_excl),
        )
        dup_by_norm = cur.fetchone()[0]

        # Top duplicate normalized times
        cur.execute(
            """
            SELECT nt, COUNT(*) AS c
            FROM (
              SELECT REPLACE(SUBSTR(time,1,19), 'T', ' ') AS nt
              FROM bars
              WHERE symbol=? AND time>=? AND time<?
            ) s
            GROUP BY nt
            HAVING COUNT(*)>1
            ORDER BY c DESC, nt ASC
            LIMIT 50
            """,
            (symbol, start, end_excl),
        )
        top_dup_norm = cur.fetchall()

        # Count rows that are not on 5-minute grid (minute % 5 != 0)
        # strftime('%M', nt) yields minute as 00..59 on normalized nt
        cur.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT REPLACE(SUBSTR(time,1,19), 'T', ' ') AS nt
              FROM bars
              WHERE symbol=? AND time>=? AND time<?
            ) s
            WHERE CAST(strftime('%M', nt) AS INTEGER) % 5 != 0
            """,
            (symbol, start, end_excl),
        )
        off_grid = cur.fetchone()[0]

        # Count per day summary to spot abnormal days (optional top 5)
        cur.execute(
            """
            SELECT SUBSTR(REPLACE(SUBSTR(time,1,19), 'T', ' '), 1, 10) AS d, COUNT(*) AS c
            FROM bars
            WHERE symbol=? AND time>=? AND time<?
            GROUP BY d
            ORDER BY c DESC
            LIMIT 5
            """,
            (symbol, start, end_excl),
        )
        top_days = cur.fetchall()

        print(
            json.dumps(
                {
                    "status": "ok",
                    "symbol": symbol,
                    "range": [start, end_excl],
                    "total_rows": total_rows,
                    "distinct_raw_times": distinct_raw,
                    "distinct_normalized_times": distinct_norm,
                    "duplicate_rows_by_normalized_time": dup_by_norm,
                    "top_duplicate_normalized_times": top_dup_norm,
                    "off_5min_grid_rows": off_grid,
                    "top_days_by_rowcount": top_days,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/check_symbol_duplicates_normalized.py <db_path> <symbol> <start_YYYY-MM-DD> <end_YYYY-MM-DD>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))

