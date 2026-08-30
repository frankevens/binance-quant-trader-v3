#!/usr/bin/env python3
"""
V4 优化回测引擎
- 止盈目标: +60% 回报率（杠杆后）
- 止损限制: -30% 回报率（杠杆后）
- 10x 杠杆下: TP = +6% 价格变动, SL = -3% 价格变动
- 高入场质量阈值
- 长持仓时间
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json
import os

# 数据库路径
DB_PATH = 'trader_data/market_data.db'

# V4 策略参数
LEVERAGE = 10
TP_RETURN = 0.60  # 60% 回报率（杠杆后）
SL_RETURN = -0.30  # -30% 回报率（杠杆后）
TP_PRICE_PCT = TP_RETURN / LEVERAGE  # 6% 价格变动
SL_PRICE_PCT = abs(SL_RETURN) / LEVERAGE  # 3% 价格变动

# 高入场阈值
MIN_SCORE = 0.75

# 交易对
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT']


def get_klines(conn, symbol):
    """获取 K 线数据"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ?
        ORDER BY open_time ASC
    """, (symbol,))
    rows = cursor.fetchall()
    
    if not rows:
        return None
    
    data = {
        'time': [r[0] for r in rows],
        'open': np.array([float(r[1]) for r in rows]),
        'high': np.array([float(r[2]) for r in rows]),
        'low': np.array([float(r[3]) for r in rows]),
        'close': np.array([float(r[4]) for r in rows]),
        'volume': np.array([float(r[5]) for r in rows])
    }
    return data


def ema(data, period):
    """计算 EMA"""
    out = np.zeros_like(data, dtype=float)
    out[0] = data[0]
    k = 2 / (period + 1)
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out


def rsi(data, period=14):
    """计算 RSI"""
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    for i in range(period, len(data)):
        avg_gain[i] = np.mean(gains[i-period:i])
        avg_loss[i] = np.mean(losses[i-period:i])
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(high, low, close, period=14):
    """计算 ATR"""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    
    atr_val = np.zeros_like(close)
    atr_val[:period] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr_val[i] = (atr_val[i-1] * (period - 1) + tr[i]) / period
    return atr_val


def generate_signal(data, i):
    """生成 V4 高确信度信号"""
    if i < 60:
        return None, 0
    
    close = data['close']
    high = data['high']
    low = data['low']
    
    # 计算指标
    ema20 = ema(close[:i+1], 20)
    ema50 = ema(close[:i+1], 50)
    rsi_val = rsi(close[:i+1], 14)
    atr_val = atr(high[:i+1], low[:i+1], close[:i+1], 14)
    
    # 布林带
    bb_period = 20
    bb_sma = np.mean(close[i-bb_period:i])
    bb_std = np.std(close[i-bb_period:i])
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    
    # 趋势判断
    trend_up = ema20[-1] > ema50[-1]
    trend_down = ema20[-1] < ema50[-1]
    
    # RSI 条件
    rsi_current = rsi_val[-1]
    rsi_oversold = rsi_current < 40
    rsi_overbought = rsi_current > 60
    
    # 价格位置
    price = close[-1]
    near_bb_lower = price <= bb_lower * 1.01
    near_bb_upper = price >= bb_upper * 0.99
    
    # V4 高确信度信号
    long_score = 0
    short_score = 0
    
    # 趋势确认（权重 30%）
    if trend_up:
        long_score += 0.30
    if trend_down:
        short_score += 0.30
    
    # RSI 确认（权重 25%）
    if rsi_oversold and trend_up:
        long_score += 0.25
    if rsi_overbought and trend_down:
        short_score += 0.25
    
    # 布林带确认（权重 25%）
    if near_bb_lower and trend_up:
        long_score += 0.25
    if near_bb_upper and trend_down:
        short_score += 0.25
    
    # 动量确认（权重 20%）
    momentum = (close[-1] - close[-5]) / close[-5] if i >= 5 else 0
    if momentum > 0.01 and trend_up:  # 1% 正向动量
        long_score += 0.20
    if momentum < -0.01 and trend_down:  # -1% 负向动量
        short_score += 0.20
    
    # 决定信号
    if long_score >= MIN_SCORE and long_score > short_score:
        return 'LONG', long_score
    elif short_score >= MIN_SCORE and short_score > long_score:
        return 'SHORT', short_score
    
    return None, max(long_score, short_score)


def simulate_backtest(data, symbol):
    """模拟回测 - 带追踪止损"""
    if data is None:
        return None
    
    trades = []
    position = None
    
    for i in range(60, len(data['close'])):
        # 检查持仓退出
        if position is not None:
            entry_price = position['entry_price']
            current_price = data['close'][i]
            current_high = data['high'][i]
            current_low = data['low'][i]
            
            if position['side'] == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                # 检查最高价是否触及 TP
                max_pnl = (current_high - entry_price) / entry_price
                hit_tp = max_pnl >= TP_PRICE_PCT
                # 检查最低价是否触及 SL
                min_pnl = (current_low - entry_price) / entry_price
                hit_sl = min_pnl <= -SL_PRICE_PCT
                
                # 追踪止损：一旦盈利超过 1%，移动 SL 到成本价
                if position.get('max_pnl', 0) > 0.01:
                    # 追踪止损：从最高点回撤 1.5% 就平仓
                    trailing_sl = position.get('trailing_sl', entry_price)
                    new_trailing_sl = max(trailing_sl, current_high * 0.985)
                    position['trailing_sl'] = new_trailing_sl
                    if current_low <= new_trailing_sl:
                        exit_price = new_trailing_sl
                        pnl_pct = (exit_price - entry_price) / entry_price
                        trade = {
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': position['entry_time'],
                            'exit_time': data['time'][i],
                            'pnl_pct': pnl_pct,
                            'leverage_pnl': pnl_pct * LEVERAGE,
                            'exit_reason': 'TRAIL',
                            'holding_bars': i - position['entry_bar']
                        }
                        trades.append(trade)
                        position = None
                        continue
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price
                max_pnl = (entry_price - current_low) / entry_price
                hit_tp = max_pnl >= TP_PRICE_PCT
                min_pnl = (entry_price - current_high) / entry_price
                hit_sl = min_pnl <= -SL_PRICE_PCT
                
                # 追踪止损
                if position.get('max_pnl', 0) > 0.01:
                    trailing_sl = position.get('trailing_sl', entry_price)
                    new_trailing_sl = min(trailing_sl, current_low * 1.015)
                    position['trailing_sl'] = new_trailing_sl
                    if current_high >= new_trailing_sl:
                        exit_price = new_trailing_sl
                        pnl_pct = (entry_price - exit_price) / entry_price
                        trade = {
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': position['entry_time'],
                            'exit_time': data['time'][i],
                            'pnl_pct': pnl_pct,
                            'leverage_pnl': pnl_pct * LEVERAGE,
                            'exit_reason': 'TRAIL',
                            'holding_bars': i - position['entry_bar']
                        }
                        trades.append(trade)
                        position = None
                        continue
            
            # 更新最大盈利
            position['max_pnl'] = max(position.get('max_pnl', 0), max_pnl)
            
            if hit_tp:
                # TP 以收盘价计算
                leverage_pnl = TP_RETURN
                trade = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': entry_price * (1 + TP_PRICE_PCT) if position['side'] == 'LONG' else entry_price * (1 - TP_PRICE_PCT),
                    'entry_time': position['entry_time'],
                    'exit_time': data['time'][i],
                    'pnl_pct': TP_PRICE_PCT,
                    'leverage_pnl': leverage_pnl,
                    'exit_reason': 'TP',
                    'holding_bars': i - position['entry_bar']
                }
                trades.append(trade)
                position = None
            elif hit_sl:
                leverage_pnl = SL_RETURN
                trade = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': entry_price * (1 - SL_PRICE_PCT) if position['side'] == 'LONG' else entry_price * (1 + SL_PRICE_PCT),
                    'entry_time': position['entry_time'],
                    'exit_time': data['time'][i],
                    'pnl_pct': -SL_PRICE_PCT,
                    'leverage_pnl': leverage_pnl,
                    'exit_reason': 'SL',
                    'holding_bars': i - position['entry_bar']
                }
                trades.append(trade)
                position = None
        
        # 检查入场信号
        if position is None:
            signal, score = generate_signal(data, i)
            if signal is not None:
                position = {
                    'side': signal,
                    'entry_price': data['close'][i],
                    'entry_time': data['time'][i],
                    'entry_bar': i,
                    'score': score,
                    'max_pnl': 0,
                    'trailing_sl': data['close'][i]
                }
    
    # 处理未平仓
    if position is not None:
        current_price = data['close'][-1]
        entry_price = position['entry_price']
        
        if position['side'] == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        leverage_pnl = pnl_pct * LEVERAGE
        
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'entry_time': position['entry_time'],
            'exit_time': data['time'][-1],
            'pnl_pct': pnl_pct,
            'leverage_pnl': leverage_pnl,
            'exit_reason': 'END',
            'holding_bars': len(data['close']) - 1 - position['entry_bar']
        }
        trades.append(trade)
    
    return trades


def calculate_stats(trades):
    """计算统计数据"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_rr': 0,
            'avg_holding_bars': 0
        }
    
    total_trades = len(trades)
    winners = sum(1 for t in trades if t['leverage_pnl'] > 0)
    win_rate = winners / total_trades if total_trades > 0 else 0
    
    total_pnl = sum(t['leverage_pnl'] for t in trades)
    gross_profit = sum(t['leverage_pnl'] for t in trades if t['leverage_pnl'] > 0)
    gross_loss = abs(sum(t['leverage_pnl'] for t in trades if t['leverage_pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # 最大回撤
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['leverage_pnl']
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    
    max_drawdown = max_dd / 10000 * 100 if total_pnl != 0 else 0  # 假设初始资金 10000
    
    # 平均盈亏比
    avg_win = gross_profit / winners if winners > 0 else 0
    avg_loss = gross_loss / (total_trades - winners) if (total_trades - winners) > 0 else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else None
    
    # 平均持仓时间
    avg_holding = sum(t['holding_bars'] for t in trades) / total_trades
    
    return {
        'total_trades': total_trades,
        'winners': winners,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor if profit_factor != float('inf') else None,
        'max_drawdown': max_drawdown,
        'avg_rr': avg_rr,
        'avg_holding_bars': avg_holding
    }


def save_results(results, trades):
    """保存结果到数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建 V4 结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS v4_backtest_results (
            symbol TEXT PRIMARY KEY,
            total_trades INTEGER,
            winners INTEGER,
            win_rate REAL,
            total_pnl REAL,
            profit_factor REAL,
            max_drawdown REAL,
            avg_rr REAL,
            avg_holding_bars REAL,
            tp_target REAL,
            sl_target REAL,
            leverage INTEGER,
            min_score REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS v4_backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            entry_time INTEGER,
            exit_time INTEGER,
            pnl_pct REAL,
            leverage_pnl REAL,
            exit_reason TEXT,
            holding_bars INTEGER
        )
    """)
    
    # 清空旧数据
    cursor.execute("DELETE FROM v4_backtest_results")
    cursor.execute("DELETE FROM v4_backtest_trades")
    
    # 保存结果
    for symbol, stats in results.items():
        cursor.execute("""
            INSERT INTO v4_backtest_results 
            (symbol, total_trades, winners, win_rate, total_pnl, profit_factor, 
             max_drawdown, avg_rr, avg_holding_bars, tp_target, sl_target, leverage, min_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            stats['total_trades'],
            stats['winners'],
            stats['win_rate'],
            stats['total_pnl'],
            stats['profit_factor'],
            stats['max_drawdown'],
            stats['avg_rr'],
            stats['avg_holding_bars'],
            TP_RETURN,
            SL_RETURN,
            LEVERAGE,
            MIN_SCORE
        ))
    
    # 保存交易明细
    for trade in trades:
        cursor.execute("""
            INSERT INTO v4_backtest_trades
            (symbol, side, entry_price, exit_price, entry_time, exit_time,
             pnl_pct, leverage_pnl, exit_reason, holding_bars)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['symbol'],
            trade['side'],
            trade['entry_price'],
            trade['exit_price'],
            trade['entry_time'],
            trade['exit_time'],
            trade['pnl_pct'],
            trade['leverage_pnl'],
            trade['exit_reason'],
            trade['holding_bars']
        ))
    
    conn.commit()
    conn.close()


def main():
    print("=" * 70)
    print("V4 优化回测引擎")
    print("=" * 70)
    print(f"策略参数:")
    print(f"  止盈目标: +{TP_RETURN*100:.0f}% 回报率（杠杆后）= +{TP_PRICE_PCT*100:.1f}% 价格变动")
    print(f"  止损限制: {SL_RETURN*100:.0f}% 回报率（杠杆后）= -{SL_PRICE_PCT*100:.1f}% 价格变动")
    print(f"  杠杆: {LEVERAGE}x")
    print(f"  入场阈值: {MIN_SCORE}")
    print(f"  盈亏比: {TP_RETURN/abs(SL_RETURN):.1f}:1")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    
    results = {}
    all_trades = []
    
    for symbol in SYMBOLS:
        print(f"\n处理 {symbol}...")
        data = get_klines(conn, symbol)
        
        if data is None:
            print(f"  无数据，跳过")
            continue
        
        print(f"  K线数量: {len(data['close'])}")
        
        trades = simulate_backtest(data, symbol)
        if trades:
            stats = calculate_stats(trades)
            results[symbol] = stats
            all_trades.extend(trades)
            
            print(f"  交易数: {stats['total_trades']}")
            print(f"  赢率: {stats['win_rate']*100:.1f}%")
            print(f"  总 PnL: {stats['total_pnl']:+.2f}%")
            print(f"  利润因子: {stats['profit_factor']:.2f}" if stats['profit_factor'] else "  利润因子: N/A")
            print(f"  平均持仓: {stats['avg_holding_bars']:.0f} 根 K线")
        else:
            print(f"  无交易")
    
    conn.close()
    
    # 保存结果
    save_results(results, all_trades)
    
    # 汇总
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    total_trades = sum(s['total_trades'] for s in results.values())
    total_wins = sum(s['winners'] for s in results.values())
    total_pnl = sum(s['total_pnl'] for s in results.values())
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"总交易数: {total_trades}")
    print(f"平均赢率: {avg_wr:.1f}%")
    print(f"总 PnL: {total_pnl:+.2f}%")
    print(f"策略: TP +{TP_RETURN*100:.0f}% / SL {SL_RETURN*100:.0f}%")
    print("=" * 70)
    
    print("\n结果已保存到数据库")


if __name__ == '__main__':
    main()
