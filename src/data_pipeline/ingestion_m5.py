"""
M5 real-time and batch ingestion pipeline (V4)

Responsibilities:
- Connect to data sources (broker APIs, CSV, or message bus) to ingest M5 OHLCV bars.
- Normalize timestamps to UTC 'YYYY-MM-DD HH:MM:SS' on a 5-minute grid.
- Validate schema and basic sanity (non-negative volume, OHLC consistency).
- Upsert into SQLite 'bars' table with natural key (symbol, time).
- Emits hooks for feature pipelines.

Note: Implementation is a stub; wire to actual sources in Task 1.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "m5_trading.db"


@dataclass
class Bar:
    symbol: str
    time: datetime  # normalized UTC, minute % 5 == 0
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation: required columns, 5m grid, OHLC consistency."""
    required = {"symbol", "time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    # Ensure datetime and 5-minute grid
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    if (df["time"].dt.minute % 5 != 0).any():
        raise ValueError("Found rows off 5-minute grid")

    # OHLC sanity
    bad = (df["low"] > df["high"]) | (df[["open", "close"]].max(axis=1) > df["high"]) | (df[["open", "close"]].min(axis=1) < df["low"])  # noqa: E501
    if bad.any():
        raise ValueError("Found inconsistent OHLC rows")

    # Sort and de-dup within batch
    df = df.sort_values(["symbol", "time"]).drop_duplicates(["symbol", "time"], keep="last")
    return df


def upsert_bars(rows: Iterable[Bar], db_path: Optional[Path] = None) -> int:
    """Upsert bars into SQLite. Returns number of rows written (best-effort)."""
    db_path = db_path or DB_PATH
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return 0
    df = _validate_bars(df)

    with sqlite3.connect(str(db_path)) as conn:
        # ensure table exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
              symbol TEXT NOT NULL,
              time   DATETIME NOT NULL,
              open   REAL NOT NULL,
              high   REAL NOT NULL,
              low    REAL NOT NULL,
              close  REAL NOT NULL,
              volume REAL NOT NULL DEFAULT 0,
              PRIMARY KEY (symbol, time)
            )
            """
        )
        # Use INSERT OR REPLACE for idempotent upserts
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO bars(symbol, time, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    r.symbol,
                    r.time.strftime("%Y-%m-%d %H:%M:%S"),
                    float(r.open),
                    float(r.high),
                    float(r.low),
                    float(r.close),
                    float(r.volume or 0.0),
                )
                for r in rows
            ],
        )
        conn.commit()
        return cur.rowcount if cur.rowcount != -1 else len(df)


# TODO: implement concrete ingestors (MT5, IB, Kafka, CSV Folder Watcher) per Task 1.4.

