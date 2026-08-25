"""
Binance Quant Trader V3 - SQLite Logger
========================================
Persistent trade logging, PnL tracking, and audit trail via SQLite.
"""

import sqlite3
import json
import time
import threading
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("trader.db")


class TradeDB:
    """SQLite-backed trade and event logger."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    position_side TEXT,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    avg_price REAL,
                    status TEXT NOT NULL,
                    order_id INTEGER,
                    client_order_id TEXT,
                    realized_pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    commission_asset TEXT,
                    leverage INTEGER,
                    margin_type TEXT,
                    entry_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    atr_value REAL,
                    strategy_signal TEXT,
                    notes TEXT,
                    raw_response TEXT
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    position_side TEXT NOT NULL,
                    position_amt REAL NOT NULL DEFAULT 0,
                    entry_price REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0,
                    leverage INTEGER NOT NULL DEFAULT 1,
                    margin_type TEXT NOT NULL DEFAULT 'ISOLATED',
                    liquidation_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    last_update REAL NOT NULL,
                    last_update_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    symbol TEXT,
                    message TEXT,
                    data TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    commission REAL NOT NULL DEFAULT 0,
                    trade_count INTEGER NOT NULL DEFAULT 0,
                    starting_balance REAL,
                    ending_balance REAL
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)
            conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def _utcnow(self):
        return datetime.now(timezone.utc)

    def log_trade(self, **kwargs):
        now = self._utcnow()
        kwargs.setdefault("timestamp", time.time())
        kwargs.setdefault("datetime_utc", now.isoformat())
        if "raw_response" in kwargs and isinstance(kwargs["raw_response"], dict):
            kwargs["raw_response"] = json.dumps(kwargs["raw_response"])

        fields = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        values = list(kwargs.values())

        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO trades ({fields}) VALUES ({placeholders})",
                values
            )
            conn.commit()

    def update_trade(self, order_id: int, **kwargs):
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [order_id]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE trades SET {set_clause} WHERE order_id = ?",
                values
            )
            conn.commit()

    def upsert_position(self, symbol: str, **kwargs):
        now = self._utcnow()
        kwargs["symbol"] = symbol
        kwargs["last_update"] = time.time()
        kwargs["last_update_utc"] = now.isoformat()

        fields = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        updates = ", ".join([f"{k} = excluded.{k}" for k in kwargs.keys() if k != "symbol"])
        values = list(kwargs.values())

        with self._conn() as conn:
            conn.execute(
                f"""INSERT INTO positions ({fields}) VALUES ({placeholders})
                    ON CONFLICT(symbol) DO UPDATE SET {updates}""",
                values
            )
            conn.commit()

    def get_position(self, symbol: str):
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if row:
                cols = [d[0] for d in cursor.description]
                return dict(zip(cols, row))
        return None

    def get_all_positions(self) -> list:
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE position_amt != 0")
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def clear_position(self, symbol: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def log_event(self, event_type: str, symbol: str = None, message: str = "", data: dict = None):
        now = self._utcnow()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, datetime_utc, event_type, symbol, message, data) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), now.isoformat(), event_type, symbol, message,
                 json.dumps(data) if data else None)
            )
            conn.commit()

    def update_daily_pnl(self, realized_pnl: float, commission: float):
        today = self._utcnow().strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO daily_pnl (date, realized_pnl, commission, trade_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(date) DO UPDATE SET
                    realized_pnl = realized_pnl + excluded.realized_pnl,
                    commission = commission + excluded.commission,
                    trade_count = trade_count + 1
            """, (today, realized_pnl, commission))
            conn.commit()

    def get_daily_pnl(self, date: str = None):
        if date is None:
            date = self._utcnow().strftime("%Y-%m-%d")
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
            row = cursor.fetchone()
            if row:
                cols = [d[0] for d in cursor.description]
                return dict(zip(cols, row))
        return None

    def set_state(self, key: str, value):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time())
            )
            conn.commit()

    def get_state(self, key: str, default=None):
        with self._conn() as conn:
            cursor = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return default

    def get_recent_trades(self, symbol: str = None, limit: int = 50) -> list:
        with self._conn() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_today_trade_count(self, symbol: str = None) -> int:
        today_start = self._utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        with self._conn() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE symbol = ? AND timestamp >= ?",
                    (symbol, today_start)
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE timestamp >= ?",
                    (today_start,)
                )
            return cursor.fetchone()[0]

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
