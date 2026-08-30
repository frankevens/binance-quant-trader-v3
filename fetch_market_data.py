#!/usr/bin/env python3
"""
下载 Binance USDT-M 永续合约历史 K 线数据到本地 SQLite 数据库
使用公开 API，无需 API Key
"""
import sqlite3
import requests
import time
import os
import json
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), 'trader_data', 'market_data.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT']
INTERVAL = '15m'
DAYS = 30

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS klines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        open_time INTEGER NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        close_time INTEGER,
        quote_volume REAL,
        trades INTEGER,
        UNIQUE(symbol, interval, open_time)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        total_trades INTEGER,
        winners INTEGER,
        losers INTEGER,
        win_rate REAL,
        total_pnl REAL,
        avg_pnl_per_trade REAL,
        max_drawdown REAL,
        profit_factor REAL,
        avg_rr REAL,
        run_time TEXT,
        params TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        symbol TEXT,
        side TEXT,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        pnl_pct REAL,
        rr REAL,
        is_winner INTEGER,
        entry_time INTEGER,
        exit_time INTEGER,
        exit_reason TEXT
    )''')
    conn.commit()
    return conn

def download_klines(symbol, interval, days):
    """从 Binance 公开 API 下载 K 线数据"""
    url = 'https://fapi.binance.com/fapi/v1/klines'
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    all_klines = []
    current_start = start_time

    print(f"  下载 {symbol} {interval} K线 ({days}天)...")

    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1500
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"    限流，等待 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    请求失败: {e}")
            break

        if not data:
            break

        for k in data:
            all_klines.append({
                'symbol': symbol,
                'interval': interval,
                'open_time': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': k[6],
                'quote_volume': float(k[7]),
                'trades': k[8]
            })

        if len(data) < 1500:
            break

        current_start = data[-1][0] + 1
        time.sleep(0.3)

    print(f"    获取 {len(all_klines)} 根 K 线")
    return all_klines

def save_klines(conn, klines):
    c = conn.cursor()
    count = 0
    for k in klines:
        try:
            c.execute('''INSERT OR IGNORE INTO klines
                (symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (k['symbol'], k['interval'], k['open_time'],
                 k['open'], k['high'], k['low'], k['close'],
                 k['volume'], k['close_time'], k['quote_volume'], k['trades']))
            count += c.rowcount
        except Exception as e:
            pass
    conn.commit()
    return count

def main():
    print("=" * 60)
    print("Binance Quant Trader V3 - 行情数据下载")
    print("=" * 60)
    print(f"数据库: {DB_PATH}")
    print(f"币种: {', '.join(SYMBOLS)}")
    print(f"周期: {INTERVAL}, 天数: {DAYS}")
    print()

    conn = init_db()

    total_saved = 0
    for symbol in SYMBOLS:
        klines = download_klines(symbol, INTERVAL, DAYS)
        saved = save_klines(conn, klines)
        total_saved += saved
        time.sleep(0.5)

    # 统计
    c = conn.cursor()
    c.execute("SELECT symbol, COUNT(*) FROM klines GROUP BY symbol")
    rows = c.fetchall()

    print(f"\n{'=' * 60}")
    print(f"下载完成！共保存 {total_saved} 条记录")
    print(f"{'=' * 60}")
    for symbol, count in rows:
        c.execute("SELECT MIN(open_time), MAX(open_time) FROM klines WHERE symbol=?", (symbol,))
        min_t, max_t = c.fetchone()
        start = datetime.fromtimestamp(min_t / 1000).strftime('%Y-%m-%d %H:%M') if min_t else 'N/A'
        end = datetime.fromtimestamp(max_t / 1000).strftime('%Y-%m-%d %H:%M') if max_t else 'N/A'
        print(f"  {symbol}: {count} 根 K 线 ({start} ~ {end})")

    conn.close()
    print(f"\n数据库路径: {DB_PATH}")

if __name__ == '__main__':
    main()
