import os
import sys
import sqlite3
import json

def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name;")
        items = cur.fetchall()
        out = []
        for it in items:
            name = it[0]
            typ = it[1]
            info = {"name": name, "type": typ}
            try:
                cur.execute(f"PRAGMA table_info('{name}')")
                cols = [dict(cid=row[0], name=row[1], type=row[2], notnull=row[3], dflt=row[4], pk=row[5]) for row in cur.fetchall()]
                info["columns"] = cols
            except sqlite3.Error as e:
                info["columns_error"] = str(e)
            try:
                cur.execute(f"SELECT COUNT(*) AS n FROM '{name}'")
                info["row_count"] = cur.fetchone()[0]
            except sqlite3.Error as e:
                info["row_count_error"] = str(e)
            out.append(info)
        print(json.dumps({"status": "ok", "objects": out}, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/inspect_db.py <path_to_sqlite_db>"}, ensure_ascii=False))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

