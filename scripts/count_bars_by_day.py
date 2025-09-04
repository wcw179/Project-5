import os
import sys
import sqlite3
import json


def main(db_path: str, symbol: str, day: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1
    start = f"{day} 00:00:00"
    # end exclusive next day
    from datetime import datetime, timedelta
    d = datetime.strptime(day, "%Y-%m-%d")
    end = (d + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), MIN(time), MAX(time) FROM bars WHERE symbol=? AND time>=? AND time<?",
            (symbol, start, end),
        )
        n, tmin, tmax = cur.fetchone()
        print(json.dumps({"status": "ok", "symbol": symbol, "day": day, "count": n, "min_time": tmin, "max_time": tmax}, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/count_bars_by_day.py <db_path> <symbol> <YYYY-MM-DD>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))

