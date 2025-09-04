import os
import sys
import sqlite3
import json
from typing import List, Dict, Any

M5_NAME_HINTS = ["m5", "5m", "5_min", "5min", "bar5", "bars5"]
TIME_COL_HINTS = ["time", "timestamp", "datetime", "date", "ts"]
OPEN_COL_HINTS = ["open", "o"]
HIGH_COL_HINTS = ["high", "h"]
LOW_COL_HINTS = ["low", "l"]
CLOSE_COL_HINTS = ["close", "c"]
VOL_COL_HINTS = ["volume", "vol", "v"]
SYMBOL_COL_HINTS = ["symbol", "instrument", "ticker", "pair", "asset"]
TF_COL_HINTS = ["timeframe", "interval", "tf", "resolution", "bar_size", "granularity", "timeframe_min", "tf_min", "interval_min"]

M5_VALUES = ["M5", "m5", "5m", "5", "5_min", "5min"]
M5_INTS = [5]

TARGET_SYMBOLS = ["XAUUSDm", "EURUSDm", "GBPUSDm", "USDJPYm"]


def get_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in cur.fetchall()]


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [row[1] for row in cur.fetchall()]


def has_any(cols: List[str], hints: List[str]) -> bool:
    lc = [c.lower() for c in cols]
    return any(h in lc for h in hints)


def find_col(cols: List[str], hints: List[str]) -> str:
    lc_map = {c.lower(): c for c in cols}
    for h in hints:
        if h in lc_map:
            return lc_map[h]
    # Try partial
    for c in cols:
        cl = c.lower()
        for h in hints:
            if h in cl:
                return c
    return ""


def is_bar_candidate(table: str, cols: List[str]) -> bool:
    lc = table.lower()
    # Require O/H/L/C columns
    has_ohlc = (
        find_col(cols, OPEN_COL_HINTS)
        and find_col(cols, HIGH_COL_HINTS)
        and find_col(cols, LOW_COL_HINTS)
        and find_col(cols, CLOSE_COL_HINTS)
    )
    has_time = bool(find_col(cols, TIME_COL_HINTS))
    return has_ohlc and has_time


def is_m5_table_name(table: str) -> bool:
    tl = table.lower()
    return any(h in tl for h in M5_NAME_HINTS)


def count_m5(conn: sqlite3.Connection) -> Dict[str, Any]:
    tables = get_tables(conn)
    result: Dict[str, Any] = {"symbols": {}, "tables": []}
    for s in TARGET_SYMBOLS:
        result["symbols"][s] = 0

    for t in tables:
        cols = get_columns(conn, t)
        if not is_bar_candidate(t, cols):
            continue

        symbol_col = find_col(cols, SYMBOL_COL_HINTS)
        tf_col = find_col(cols, TF_COL_HINTS)
        time_col = find_col(cols, TIME_COL_HINTS)

        treat_as_m5 = False
        where_clauses = []
        params: List[Any] = []

        if tf_col:
            # Try to see if this table has any m5 rows at all
            # We'll build WHERE for timeframe variants
            tf_checks = []
            tf_checks.append(f"LOWER({tf_col}) IN ({','.join(['?']*len(M5_VALUES))})")
            params.extend([v.lower() for v in M5_VALUES])
            # Numeric tf columns
            tf_checks.append(f"{tf_col} IN ({','.join(['?']*len(M5_INTS))})")
            params.extend(M5_INTS)
            where_clauses.append("(" + " OR ".join(tf_checks) + ")")
        else:
            # No tf column; decide by table name
            if is_m5_table_name(t):
                treat_as_m5 = True
            else:
                continue  # cannot assert m5; skip

        if not symbol_col:
            # If no symbol column, skip; ambiguous
            continue

        breakdown = {"table": t, "counts": {}}
        for sym in TARGET_SYMBOLS:
            try:
                q = f"SELECT COUNT(*) FROM '{t}' WHERE {symbol_col}=?"
                p = [sym]
                if tf_col:
                    q += " AND " + " AND ".join(where_clauses)
                    p += params
                elif treat_as_m5:
                    # nothing extra
                    pass
                else:
                    continue
                cur = conn.execute(q, p)
                n = cur.fetchone()[0]
                breakdown["counts"][sym] = n
                result["symbols"][sym] += n
            except sqlite3.Error:
                breakdown["counts"][sym] = None
        result["tables"].append(breakdown)

    return result


def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1
    conn = sqlite3.connect(db_path)
    try:
        summary = count_m5(conn)
        print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/count_m5_rows.py <path_to_sqlite_db>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

