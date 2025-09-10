import os
import sys
import sqlite3
import json

SYMBOLS = ["XAUUSDm", "EURUSDm", "GBPUSDm", "USDJPYm"]


def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Detect if bars table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bars';")
        row = cur.fetchone()
        if not row:
            print(json.dumps({"status": "error", "message": "Table 'bars' not found"}, ensure_ascii=False))
            return 1
        # Count total and per symbol
        out = {"status": "ok", "table": "bars", "total": 0, "per_symbol": {}}
        cur.execute("SELECT COUNT(*) FROM bars")
        out["total"] = cur.fetchone()[0]
        for sym in SYMBOLS:
            cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?", (sym,))
            out["per_symbol"][sym] = cur.fetchone()[0]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/count_bars_per_symbol.py <path_to_sqlite_db>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

