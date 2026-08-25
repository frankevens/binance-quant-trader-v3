"""
Binance Quant Trader V3 - Status Monitor
==========================================
Quick status check: positions, PnL, recent trades, system health.
Run: python monitor.py
"""

import sys
import os
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "trader_data/trades.db"


def get_db():
    if not Path(DB_PATH).exists():
        print(f"Database not found: {DB_PATH}")
        print("Trader may not have been started yet.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def show_positions(conn):
    print("\n" + "=" * 60)
    print("  OPEN POSITIONS")
    print("=" * 60)
    cursor = conn.execute("SELECT * FROM positions WHERE position_amt != 0")
    rows = cursor.fetchall()
    if not rows:
        print("  No open positions")
        return

    total_pnl = 0
    for row in rows:
        pnl = row["unrealized_pnl"]
        total_pnl += pnl
        side = "LONG" if row["position_amt"] > 0 else "SHORT"
        print(f"  {row['symbol']:10s} | {side:5s} | qty={row['position_amt']:+.6f} | "
              f"entry={row['entry_price']:.2f} | uPnL={pnl:+.4f} | "
              f"lev={row['leverage']}x {row['margin_type']}")

    print(f"  {'':10s} | {'Total Unrealized PnL':>40s} = {total_pnl:+.4f} USDT")


def show_daily_pnl(conn):
    print("\n" + "=" * 60)
    print("  TODAY'S PnL")
    print("=" * 60)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cursor = conn.execute("SELECT * FROM daily_pnl WHERE date = ?", (today,))
    row = cursor.fetchone()
    if row:
        print(f"  Date:           {row['date']}")
        print(f"  Realized PnL:   {row['realized_pnl']:+.4f} USDT")
        print(f"  Commission:     {row['commission']:.4f} USDT")
        print(f"  Net PnL:        {row['realized_pnl'] - row['commission']:+.4f} USDT")
        print(f"  Trade Count:    {row['trade_count']}")
    else:
        print("  No trades today")


def show_recent_trades(conn, limit=10):
    print(f"\n" + "=" * 60)
    print(f"  RECENT TRADES (last {limit})")
    print("=" * 60)
    cursor = conn.execute(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    if not rows:
        print("  No trades recorded")
        return

    for row in rows:
        dt = row["datetime_utc"][:19]
        print(f"  {dt} | {row['symbol']:10s} | {row['side']:5s} | "
              f"qty={row['quantity']:.6f} | price={row.get('avg_price', 0):.2f} | "
              f"rPnL={row.get('realized_pnl', 0):+.4f} | {row['status']}")


def show_system_state(conn):
    print("\n" + "=" * 60)
    print("  SYSTEM STATE")
    print("=" * 60)
    cursor = conn.execute("SELECT * FROM system_state")
    rows = cursor.fetchall()
    if not rows:
        print("  No system state recorded")
        return
    for row in rows:
        print(f"  {row['key']}: {row['value']}")


def show_recent_events(conn, limit=20):
    print(f"\n" + "=" * 60)
    print(f"  RECENT EVENTS (last {limit})")
    print("=" * 60)
    cursor = conn.execute(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    if not rows:
        print("  No events recorded")
        return
    for row in rows:
        dt = row["datetime_utc"][:19]
        sym = row["symbol"] or ""
        print(f"  {dt} | {row['event_type']:15s} | {sym:10s} | {row['message'][:60]}")


def show_summary(conn):
    print("\n" + "=" * 60)
    print("  TRADING SUMMARY")
    print("=" * 60)
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losers,
            SUM(realized_pnl) as total_pnl,
            SUM(commission) as total_commission
        FROM trades
        WHERE status IN ('FILLED', 'CLOSED')
    """)
    row = cursor.fetchone()
    if row and row["total_trades"] > 0:
        win_rate = row["winners"] / (row["winners"] + row["losers"]) * 100 if (row["winners"] + row["losers"]) > 0 else 0
        print(f"  Total Trades:    {row['total_trades']}")
        print(f"  Winners:         {row['winners']}")
        print(f"  Losers:          {row['losers']}")
        print(f"  Win Rate:        {win_rate:.1f}%")
        print(f"  Total Realized:  {row['total_pnl'] or 0:+.4f} USDT")
        print(f"  Total Commission:{row['total_commission'] or 0:.4f} USDT")
        net = (row['total_pnl'] or 0) - (row['total_commission'] or 0)
        print(f"  Net PnL:         {net:+.4f} USDT")
    else:
        print("  No completed trades yet")


def main():
    print("=" * 60)
    print("  BINANCE QUANT TRADER V3 - STATUS MONITOR")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    conn = get_db()
    try:
        show_positions(conn)
        show_daily_pnl(conn)
        show_summary(conn)
        show_recent_trades(conn)
        show_recent_events(conn)
        show_system_state(conn)
    finally:
        conn.close()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
