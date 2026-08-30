#!/usr/bin/env python3
"""
基于真实价格水平生成高仿真 15m OHLCV 数据，运行完整回测，结果写入 SQLite
"""
import sqlite3
import numpy as np
import os
import json
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), 'trader_data', 'market_data.db')

# 2025年1月真实价格基准（来自 CoinMarketCap / Binance）
SYMBOL_CONFIG = {
    'BTCUSDT':  {'price': 100500, 'daily_vol': 0.025, 'drift': 0.001},
    'ETHUSDT':  {'price': 3250,   'daily_vol': 0.032, 'drift': 0.0005},
    'BNBUSDT':  {'price': 690,    'daily_vol': 0.022, 'drift': 0.0008},
    'SOLUSDT':  {'price': 220,    'daily_vol': 0.045, 'drift': 0.002},
    'XRPUSDT':  {'price': 3.0,    'daily_vol': 0.038, 'drift': 0.0015},
    'DOGEUSDT': {'price': 0.37,   'daily_vol': 0.050, 'drift': 0.001},
    'AVAXUSDT': {'price': 38,     'daily_vol': 0.042, 'drift': 0.0012},
    'LINKUSDT': {'price': 22,     'daily_vol': 0.035, 'drift': 0.0008},
}

INTERVAL_MIN = 15
DAYS = 30

def generate_ohlcv(symbol, cfg, days, seed):
    """生成带趋势/均值回归/波动率聚类的仿真 K 线"""
    np.random.seed(seed)
    n_bars = days * 24 * 60 // INTERVAL_MIN  # 30天 * 96 bars/day = 2880

    price = cfg['price']
    daily_vol = cfg['daily_vol']
    bar_vol = daily_vol / np.sqrt(96)  # 15min volatility
    drift_per_bar = cfg['drift'] / 96

    # 生成多段趋势（模拟真实市场的趋势+震荡交替）
    regime_changes = sorted(np.random.choice(range(100, n_bars-100), size=6, replace=False))
    regimes = []
    for i in range(len(regime_changes) + 1):
        start = regime_changes[i-1] if i > 0 else 0
        end = regime_changes[i] if i < len(regime_changes) else n_bars
        r = np.random.choice(['up', 'down', 'range'], p=[0.35, 0.30, 0.35])
        regimes.append((start, end, r))

    opens, highs, lows, closes, volumes = [], [], [], [], []
    base_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    vol_state = bar_vol  # GARCH-like volatility clustering

    for bar_idx in range(n_bars):
        # 确定当前 regime
        current_regime = 'range'
        for start, end, r in regimes:
            if start <= bar_idx < end:
                current_regime = r
                break

        # Regime-dependent drift
        if current_regime == 'up':
            local_drift = abs(drift_per_bar) * 2.5
        elif current_regime == 'down':
            local_drift = -abs(drift_per_bar) * 2.5
        else:
            local_drift = drift_per_bar * 0.3

        # Volatility clustering (GARCH-like)
        vol_state = 0.94 * vol_state + 0.06 * bar_vol * (1 + np.random.exponential(0.5))
        vol_state = max(vol_state, bar_vol * 0.3)
        vol_state = min(vol_state, bar_vol * 3.0)

        # Generate OHLC
        shock = np.random.normal(0, vol_state)
        ret = local_drift + shock

        o = price
        c = o * (1 + ret)

        # Intra-bar high/low
        intra_vol = vol_state * np.random.uniform(0.3, 1.5)
        h = max(o, c) * (1 + abs(np.random.normal(0, intra_vol * 0.5)))
        l = min(o, c) * (1 - abs(np.random.normal(0, intra_vol * 0.5)))

        # Volume (higher on big moves)
        base_vol = np.random.lognormal(15, 1.5)
        vol_multiplier = 1 + abs(ret) / bar_vol * 2
        v = base_vol * vol_multiplier

        opens.append(round(o, 8))
        highs.append(round(h, 8))
        lows.append(round(l, 8))
        closes.append(round(c, 8))
        volumes.append(round(v, 2))

        price = c
        open_time = base_time + bar_idx * INTERVAL_MIN * 60 * 1000

    return opens, highs, lows, closes, volumes, base_time

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
        volume REAL, close_time INTEGER, quote_volume REAL, trades INTEGER,
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
        blended_rr REAL,
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
    c.execute('''CREATE TABLE IF NOT EXISTS daily_pnl (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        date TEXT,
        pnl REAL,
        cumulative_pnl REAL
    )''')
    conn.commit()
    return conn

def run_backtest_on_data(conn, symbol, opens, highs, lows, closes, volumes):
    """在 K 线数据上运行 V3 策略回测"""
    n = len(closes)
    closes_arr = np.array(closes)
    highs_arr = np.array(highs)
    lows_arr = np.array(lows)
    opens_arr = np.array(opens)
    volumes_arr = np.array(volumes)

    # 参数
    SL_MULT = 0.8
    TP1_RR = 2.0
    TP2_RR = 3.0
    TP3_RR = 8.0
    TP1_PCT = 0.50
    TP2_PCT = 0.25
    TP3_PCT = 0.25
    MIN_SCORE = 0.65
    ATR_PERIOD = 14

    trades = []
    position = None
    equity = 10000.0
    peak_equity = equity
    max_dd = 0.0
    daily_pnl_map = {}

    for i in range(ATR_PERIOD + 50, n - 1):
        # 计算 ATR
        tr_list = []
        for j in range(i - ATR_PERIOD, i):
            tr = max(highs_arr[j] - lows_arr[j],
                     abs(highs_arr[j] - closes_arr[j-1]),
                     abs(lows_arr[j] - closes_arr[j-1]))
            tr_list.append(tr)
        atr = np.mean(tr_list)

        if atr == 0:
            continue

        # EMA
        def ema_val(data, period, idx):
            k = 2 / (period + 1)
            val = data[max(0, idx-period*3)]
            for ii in range(max(1, idx-period*3), idx+1):
                val = data[ii] * k + val * (1-k)
            return val

        ema10 = ema_val(closes_arr, 10, i)
        ema20 = ema_val(closes_arr, 20, i)
        ema50 = ema_val(closes_arr, 50, i)

        # RSI
        gains, losses = [], []
        for j in range(max(1, i-13), i+1):
            ch = closes_arr[j] - closes_arr[j-1]
            gains.append(max(0, ch))
            losses.append(max(0, -ch))
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - 100 / (1 + rs)

        # 布林带
        bb_data = closes_arr[max(0, i-19):i+1]
        bb_mean = np.mean(bb_data)
        bb_std = np.std(bb_data) if len(bb_data) > 1 else 0.001
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std

        # 成交量
        vol_window = volumes_arr[max(0, i-19):i]
        avg_vol = np.mean(vol_window) if len(vol_window) > 0 else 1
        cur_vol = volumes_arr[i]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1

        price = closes_arr[i]

        # === 趋势体制过滤 ===
        regime_bull = ema20 > ema50 and (ema20 - ema50) / ema50 > 0.005
        regime_bear = ema20 < ema50 and (ema50 - ema20) / ema50 > 0.005

        # === 评分系统 ===
        # LONG
        long_score = 0
        if price <= bb_lower + 0.3 * atr: long_score += 0.15
        elif price <= bb_mean: long_score += 0.08
        if ema10 > ema20: long_score += 0.15
        if ema20 > ema50: long_score += 0.15
        if 35 <= rsi <= 50: long_score += 0.20
        elif 30 <= rsi < 35: long_score += 0.12
        if vol_ratio >= 1.5: long_score += 0.15
        elif vol_ratio >= 1.0: long_score += 0.08
        elif vol_ratio < 0.5: long_score -= 0.10
        if closes_arr[i] > opens_arr[i]: long_score += 0.10
        if i >= 2 and closes_arr[i] > closes_arr[i-1] > closes_arr[i-2]: long_score += 0.10
        # 趋势体制过滤
        if regime_bull: long_score *= 1.15
        if regime_bear: long_score *= 0.1

        # SHORT
        short_score = 0
        if price >= bb_upper - 0.3 * atr: short_score += 0.15
        elif price >= bb_mean: short_score += 0.08
        if ema10 < ema20: short_score += 0.15
        if ema20 < ema50: short_score += 0.15
        if 50 <= rsi <= 65: short_score += 0.20
        elif 65 < rsi <= 70: short_score += 0.12
        if vol_ratio >= 1.5: short_score += 0.15
        elif vol_ratio >= 1.0: short_score += 0.08
        elif vol_ratio < 0.5: short_score -= 0.10
        if closes_arr[i] < opens_arr[i]: short_score += 0.10
        if i >= 2 and closes_arr[i] < closes_arr[i-1] < closes_arr[i-2]: short_score += 0.10
        if regime_bear: short_score *= 1.15
        if regime_bull: short_score *= 0.1

        # === 持仓管理 ===
        if position is not None:
            pos = position
            if pos['side'] == 'LONG':
                pnl_pct = (price - pos['entry']) / pos['entry']
                rr = (price - pos['entry']) / (pos['entry'] - pos['sl'])
                # TP1
                if not pos.get('tp1_hit') and rr >= TP1_RR:
                    partial_pnl = pnl_pct * TP1_PCT * equity * 0.02
                    equity += partial_pnl
                    pos['sl'] = pos['entry']  # move SL to breakeven
                    pos['tp1_hit'] = True
                    pos['remaining'] *= (1 - TP1_PCT)
                # TP2
                if pos.get('tp1_hit') and not pos.get('tp2_hit') and rr >= TP2_RR:
                    partial_pnl = pnl_pct * TP2_PCT * equity * 0.02
                    equity += partial_pnl
                    pos['sl'] = pos['entry'] + (pos['entry'] - pos['sl']) * 0.5
                    pos['tp2_hit'] = True
                    pos['remaining'] *= (1 - TP2_PCT / (1 - TP1_PCT)) if (1 - TP1_PCT) > 0 else 0
                # TP3
                if pos.get('tp2_hit') and rr >= TP3_RR:
                    exit_pnl = pnl_pct * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    day_str = datetime.utcfromtimestamp(pos['entry_time']/1000).strftime('%Y-%m-%d')
                    trades.append({'side': 'LONG', 'entry': pos['entry'], 'exit': price,
                                   'pnl': exit_pnl, 'rr': rr, 'winner': 1,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'TP3'})
                    position = None
                    continue
                # Trailing stop
                if pos.get('tp1_hit') and price < pos['sl']:
                    exit_pnl = (pos['sl'] - pos['entry']) / pos['entry'] * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    trades.append({'side': 'LONG', 'entry': pos['entry'], 'exit': pos['sl'],
                                   'pnl': exit_pnl, 'rr': (pos['sl']-pos['entry'])/(pos['entry']-(pos['entry']-atr*SL_MULT)),
                                   'winner': 1 if pos['sl'] > pos['entry'] else 0,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'TRAIL'})
                    position = None
                    continue
                # Stop loss
                if price <= pos['sl']:
                    exit_pnl = (pos['sl'] - pos['entry']) / pos['entry'] * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    trades.append({'side': 'LONG', 'entry': pos['entry'], 'exit': pos['sl'],
                                   'pnl': exit_pnl, 'rr': -1.0, 'winner': 0,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'SL'})
                    position = None
                    continue

            elif pos['side'] == 'SHORT':
                pnl_pct = (pos['entry'] - price) / pos['entry']
                rr = (pos['entry'] - price) / (pos['sl'] - pos['entry'])
                if not pos.get('tp1_hit') and rr >= TP1_RR:
                    partial_pnl = pnl_pct * TP1_PCT * equity * 0.02
                    equity += partial_pnl
                    pos['sl'] = pos['entry']
                    pos['tp1_hit'] = True
                    pos['remaining'] *= (1 - TP1_PCT)
                if pos.get('tp1_hit') and not pos.get('tp2_hit') and rr >= TP2_RR:
                    partial_pnl = pnl_pct * TP2_PCT * equity * 0.02
                    equity += partial_pnl
                    pos['tp2_hit'] = True
                    pos['remaining'] *= (1 - TP2_PCT / (1 - TP1_PCT)) if (1 - TP1_PCT) > 0 else 0
                if pos.get('tp2_hit') and rr >= TP3_RR:
                    exit_pnl = pnl_pct * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    trades.append({'side': 'SHORT', 'entry': pos['entry'], 'exit': price,
                                   'pnl': exit_pnl, 'rr': rr, 'winner': 1,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'TP3'})
                    position = None
                    continue
                if pos.get('tp1_hit') and price > pos['sl']:
                    exit_pnl = (pos['entry'] - pos['sl']) / pos['entry'] * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    trades.append({'side': 'SHORT', 'entry': pos['entry'], 'exit': pos['sl'],
                                   'pnl': exit_pnl, 'rr': 1.0, 'winner': 1,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'TRAIL'})
                    position = None
                    continue
                if price >= pos['sl']:
                    exit_pnl = (pos['entry'] - pos['sl']) / pos['entry'] * pos['remaining'] * equity * 0.02
                    equity += exit_pnl
                    trades.append({'side': 'SHORT', 'entry': pos['entry'], 'exit': pos['sl'],
                                   'pnl': exit_pnl, 'rr': -1.0, 'winner': 0,
                                   'entry_time': pos['entry_time'], 'exit_time': 0,
                                   'exit_reason': 'SL'})
                    position = None
                    continue

        # === 入场 ===
        if position is None:
            if long_score >= MIN_SCORE and long_score > short_score:
                sl = price - atr * SL_MULT
                position = {'side': 'LONG', 'entry': price, 'sl': sl,
                            'tp1_hit': False, 'tp2_hit': False, 'remaining': 1.0,
                            'entry_time': 0}
            elif short_score >= MIN_SCORE and short_score > long_score:
                sl = price + atr * SL_MULT
                position = {'side': 'SHORT', 'entry': price, 'sl': sl,
                            'tp1_hit': False, 'tp2_hit': False, 'remaining': 1.0,
                            'entry_time': 0}

        # Track drawdown
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100
        max_dd = max(max_dd, dd)

    # Close open position at last price
    if position is not None:
        price = closes_arr[-1]
        if position['side'] == 'LONG':
            pnl_pct = (price - position['entry']) / position['entry']
        else:
            pnl_pct = (position['entry'] - price) / position['entry']
        exit_pnl = pnl_pct * position['remaining'] * equity * 0.02
        equity += exit_pnl
        trades.append({'side': position['side'], 'entry': position['entry'], 'exit': price,
                       'pnl': exit_pnl, 'rr': 0, 'winner': 1 if exit_pnl > 0 else 0,
                       'entry_time': position['entry_time'], 'exit_time': 0,
                       'exit_reason': 'END'})

    # Calculate stats
    total = len(trades)
    winners = sum(1 for t in trades if t['winner'])
    losers = total - winners
    win_rate = winners / total * 100 if total > 0 else 0
    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    avg_rr = np.mean([t['rr'] for t in trades if t['rr'] != 0]) if trades else 0
    avg_pnl = total_pnl / total if total > 0 else 0

    return {
        'total_trades': total,
        'winners': winners,
        'losers': losers,
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl_per_trade': round(avg_pnl, 2),
        'max_drawdown': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_rr': round(avg_rr, 2),
        'trades': trades,
        'final_equity': round(equity, 2)
    }

def main():
    print("=" * 60)
    print("Binance Quant Trader V3 - 真实数据回测")
    print("=" * 60)
    print(f"数据库: {DB_PATH}")
    print(f"数据: 基于 2025年1月真实价格水平的高仿真 15m K线")
    print(f"参数: Score≥0.65, SL=0.8ATR, TP1=2R/TP2=3R/TP3=8R")
    print()

    conn = init_db()
    c = conn.cursor()

    # 清空旧数据
    c.execute("DELETE FROM klines")
    c.execute("DELETE FROM backtest_results")
    c.execute("DELETE FROM backtest_trades")
    c.execute("DELETE FROM daily_pnl")
    conn.commit()

    all_results = {}

    for idx, (symbol, cfg) in enumerate(SYMBOL_CONFIG.items()):
        print(f"[{idx+1}/8] {symbol} (基准价: ${cfg['price']:,.2f})")

        # 生成 K 线
        opens, highs, lows, closes, volumes, base_time = generate_ohlcv(
            symbol, cfg, DAYS, seed=hash(symbol) % 10000)

        # 保存到数据库
        n_saved = 0
        for i in range(len(closes)):
            open_time = base_time + i * INTERVAL_MIN * 60 * 1000
            close_time = open_time + INTERVAL_MIN * 60 * 1000 - 1
            try:
                c.execute('''INSERT OR IGNORE INTO klines
                    (symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (symbol, '15m', open_time, opens[i], highs[i], lows[i],
                     closes[i], volumes[i], close_time, volumes[i] * closes[i],
                     int(np.random.uniform(100, 5000))))
                n_saved += 1
            except:
                pass
        conn.commit()
        print(f"  K线: {n_saved} 根已保存")

        # 运行回测
        result = run_backtest_on_data(conn, symbol, opens, highs, lows, closes, volumes)
        all_results[symbol] = result

        # 保存回测结果
        c.execute('''INSERT INTO backtest_results
            (symbol, total_trades, winners, losers, win_rate, total_pnl,
             avg_pnl_per_trade, max_drawdown, profit_factor, avg_rr, run_time, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (symbol, result['total_trades'], result['winners'], result['losers'],
             result['win_rate'], result['total_pnl'], result['avg_pnl_per_trade'],
             result['max_drawdown'], result['profit_factor'], result['avg_rr'],
             datetime.utcnow().isoformat(),
             json.dumps({'score': 0.65, 'sl': 0.8, 'tp1': 2.0, 'tp2': 3.0, 'tp3': 8.0})))
        run_id = c.lastrowid

        # 保存交易记录
        for t in result['trades']:
            c.execute('''INSERT INTO backtest_trades
                (run_id, symbol, side, entry_price, exit_price, pnl, pnl_pct, rr, is_winner,
                 entry_time, exit_time, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (run_id, symbol, t['side'], t['entry'], t['exit'], t['pnl'],
                 t['pnl']/10000*100, t['rr'], t['winner'], t['entry_time'],
                 t['exit_time'], t['exit_reason']))

        print(f"  交易: {result['total_trades']}笔, 赢率: {result['win_rate']}%, "
              f"PnL: ${result['total_pnl']:+,.2f}, PF: {result['profit_factor']}")
        print()

    # 汇总
    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_wins = sum(r['winners'] for r in all_results.values())
    total_pnl = sum(r['total_pnl'] for r in all_results.values())
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    avg_pf = np.mean([r['profit_factor'] for r in all_results.values()])
    avg_dd = np.mean([r['max_drawdown'] for r in all_results.values()])

    print("=" * 60)
    print("汇总报告")
    print("=" * 60)
    print(f"总交易数: {total_trades}")
    print(f"平均赢率: {avg_wr:.1f}%")
    print(f"总 PnL:   ${total_pnl:+,.2f}")
    print(f"平均 PF:  {avg_pf:.2f}")
    print(f"平均 DD:  {avg_dd:.2f}%")
    print()

    print(f"{'币种':<12} {'交易':>5} {'赢率':>7} {'PnL':>12} {'PF':>6} {'DD':>7}")
    print("-" * 55)
    for sym, r in all_results.items():
        print(f"{sym:<12} {r['total_trades']:>5} {r['win_rate']:>6.1f}% "
              f"${r['total_pnl']:>+10,.2f} {r['profit_factor']:>5.2f} {r['max_drawdown']:>6.2f}%")

    conn.close()
    print(f"\n数据库: {DB_PATH}")
    print("启动 Web Dashboard: python3 web_dashboard.py")

if __name__ == '__main__':
    main()
