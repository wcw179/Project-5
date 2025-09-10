import os
import sys
import csv
import sqlite3
import shutil
import json
from datetime import datetime
from typing import Optional, Dict, Any


def backup_db(db_path: str, tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.{tag}-{ts}{ext}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def detect_delimiter(path: str) -> str:
    with open(path, 'r', newline='', encoding='utf-8') as f:
        sample = f.read(4096)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ','


def parse_date_ymd(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # Accept formats: YYYY-MM-DD or YYYYMMDD
    try:
        if '-' in s:
            dt = datetime.strptime(s, "%Y-%m-%d")
        else:
            dt = datetime.strptime(s, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_time_hms(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().replace('T', ' ').replace('Z', '')
    # Accept H:MM:SS or HH:MM:SS
    parts = s.split(' ')
    t = parts[-1]  # if a full datetime slipped in, take the time part
    segs = t.split(':')
    if len(segs) == 2:
        segs.append('00')
    if len(segs) != 3:
        return None
    try:
        h = int(segs[0])
        m = int(segs[1])
        sec = int(segs[2])
    except Exception:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        return None
    return f"{h:02d}:{m:02d}:{sec:02d}"


def norm_time_5min_from_date_time(date_str: str, time_str: str) -> Optional[str]:
    d = parse_date_ymd(date_str)
    t = parse_time_hms(time_str)
    if not d or not t:
        return None
    # Build datetime and ensure 5-minute grid
    try:
        dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    if dt.minute % 5 != 0:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def norm_time_5min(s: str) -> Optional[str]:
    # Backward-compatible single string normalization
    if not s:
        return None
    s = s.strip().replace('T', ' ').replace('Z', '')
    # Try full datetime first
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        # Try YYYY-MM-DD HH:MM
        try:
            dt = datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
            dt = dt.replace(second=0)
        except Exception:
            return None
    if dt.minute % 5 != 0:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return None
    # Remove thousand separators if any
    s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return None


def main(csv_path: str, db_path: str, default_symbol: Optional[str] = None):
    if not os.path.isfile(csv_path):
        print(json.dumps({"status": "error", "message": f"CSV not found: {csv_path}"}, ensure_ascii=False))
        return 1
    if not os.path.isfile(db_path):
        print(json.dumps({"status": "error", "message": f"Database not found: {db_path}"}, ensure_ascii=False))
        return 1

    # Backup DB
    backup_path = backup_db(db_path, "pre-import")

    delimiter = detect_delimiter(csv_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Verify bars table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bars';")
        if not cur.fetchone():
            print(json.dumps({"status": "error", "message": "bars table not found in DB"}, ensure_ascii=False))
            return 1

        # Prepare insertion
        insert_sql = "INSERT OR IGNORE INTO bars(symbol, time, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)"

        total_rows = 0
        inserted = 0
        ignored = 0
        skipped = 0
        first_time = None
        last_time = None

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            headers = next(reader, None)
            if headers is None:
                print(json.dumps({"status": "error", "message": "CSV is empty"}, ensure_ascii=False))
                return 1
            # Build header map (case-insensitive)
            hmap: Dict[str, int] = {}
            for idx, h in enumerate(headers):
                hmap[h.strip().lower()] = idx

            # Identify columns
            sym_idx = hmap.get('symbol')
            # Prefer combined datetime/timestamp; otherwise use separate date + time
            date_idx = hmap.get('date')
            time_only_idx = None
            for k in ('datetime', 'timestamp', 'time'):
                if k in hmap:
                    time_only_idx = hmap[k]
                    break
            open_idx = hmap.get('open')
            high_idx = hmap.get('high')
            low_idx = hmap.get('low')
            close_idx = hmap.get('close')
            vol_idx = hmap.get('volume') or hmap.get('vol')

            if (time_only_idx is None and date_idx is None) or open_idx is None or high_idx is None or low_idx is None or close_idx is None:
                print(json.dumps({"status": "error", "message": "CSV must contain at least time/date and open/high/low/close columns"}, ensure_ascii=False))
                return 1

            batch: list = []
            BATCH_SIZE = 10000

            for row in reader:
                total_rows += 1
                try:
                    sym = row[sym_idx].strip() if sym_idx is not None else (default_symbol or 'EURUSDm')
                    if not sym:
                        sym = default_symbol or 'EURUSDm'

                    # Build normalized 5-min timestamp
                    tnorm = None
                    if date_idx is not None and time_only_idx is not None:
                        tnorm = norm_time_5min_from_date_time(row[date_idx], row[time_only_idx])
                    elif time_only_idx is not None:
                        tnorm = norm_time_5min(row[time_only_idx])
                    if not tnorm:
                        skipped += 1
                        continue

                    o = to_float(row[open_idx])
                    h = to_float(row[high_idx])
                    l = to_float(row[low_idx])
                    c = to_float(row[close_idx])
                    v = to_float(row[vol_idx]) if vol_idx is not None else 0.0
                    if None in (o, h, l, c):
                        skipped += 1
                        continue

                    batch.append((sym, tnorm, o, h, l, c, v))

                    if first_time is None:
                        first_time = tnorm
                    last_time = tnorm

                    if len(batch) >= BATCH_SIZE:
                        cur.executemany(insert_sql, batch)
                        inserted += cur.rowcount if cur.rowcount != -1 else 0
                        batch.clear()
                except Exception:
                    skipped += 1
                    continue

            if batch:
                cur.executemany(insert_sql, batch)
                inserted += cur.rowcount if cur.rowcount != -1 else 0

        conn.commit()

        # Post counts for symbol
        sym_to_report = default_symbol or 'EURUSDm'
        try:
            count_after = cur.execute("SELECT COUNT(*) FROM bars WHERE symbol=?;", (sym_to_report,)).fetchone()[0]
        except Exception:
            count_after = None

        print(json.dumps({
            "status": "ok",
            "backup_path": backup_path,
            "csv": csv_path,
            "db": db_path,
            "symbol": sym_to_report,
            "rows_read": total_rows,
            "inserted_or_ignored": inserted,  # SQLite's rowcount for executemany may be -1; this is best-effort
            "skipped": skipped,
            "first_time": first_time,
            "last_time": last_time,
            "count_in_db_after": count_after
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: python scripts/import_csv_to_bars.py <csv_path> <db_path> [symbol]"}, ensure_ascii=False))
        sys.exit(1)
    csv_path = sys.argv[1]
    db_path = sys.argv[2]
    symbol = sys.argv[3] if len(sys.argv) > 3 else 'EURUSDm'
    sys.exit(main(csv_path, db_path, symbol))

