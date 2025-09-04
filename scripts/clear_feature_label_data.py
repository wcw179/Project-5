import os
import sys
import sqlite3
import shutil
from datetime import datetime
import json


def main(db_path: str):
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}))
        return 1

    # Backup first
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.backup-{ts}{ext}"
    try:
        shutil.copy2(db_path, backup_path)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Failed to create backup: {e}"}))
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # Get all user tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]

        # Identify feature/label tables by name
        target_tables = []
        for t in tables:
            tl = t.lower()
            if ("feature" in tl) or ("label" in tl):
                target_tables.append(t)

        # Also detect tables that may not have feature/label in name but in columns
        additional_tables = []
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = [row[1].lower() for row in cur.fetchall()]
                if any(("feature" in c or "label" in c) for c in cols):
                    if t not in target_tables:
                        additional_tables.append(t)
            except sqlite3.Error:
                pass

        candidates = target_tables + additional_tables
        # Deduplicate preserving order
        seen = set()
        filtered = []
        for t in candidates:
            if t not in seen:
                seen.add(t)
                filtered.append(t)

        # Prepare summary
        summary = {
            "status": "ok",
            "backup_path": backup_path,
            "cleared_tables": [],
            "skipped_tables": [],
        }

        # Delete rows in each candidate table
        conn.execute("BEGIN")
        for t in filtered:
            try:
                cur.execute(f"SELECT COUNT(*) FROM '{t}'")
                before = cur.fetchone()[0]
                cur.execute(f"DELETE FROM '{t}'")
                summary["cleared_tables"].append({"table": t, "rows_deleted": before})
            except sqlite3.Error as e:
                summary["skipped_tables"].append({"table": t, "reason": str(e)})
        conn.commit()

        # Optionally reset autoincrement sequences
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence';")
            if cur.fetchone():
                for t in filtered:
                    try:
                        cur.execute("DELETE FROM sqlite_sequence WHERE name=?", (t,))
                    except sqlite3.Error:
                        pass
                conn.commit()
        except sqlite3.Error:
            pass

        # Vacuum to reclaim space (outside transaction)
        try:
            conn.execute("VACUUM")
        except sqlite3.Error:
            pass

        print(json.dumps(summary, ensure_ascii=False))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python clear_feature_label_data.py <path_to_sqlite_db>"}))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

