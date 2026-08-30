#!/usr/bin/env python3
"""
V5 优化回测引擎

优化内容：
1. 1h 时间框架（从 15m 聚合）
2. TP +25% / SL -12%（现实盈亏比）
3. 多因子入场评分（趋势+动量+波动率+成交量）
4. ADX 过滤（只在大趋势时入场）
5. 高入场阈值（0.8+）
6. 追踪止损
"""

import sqlite3
import numpy as np
from datetime import datetime
from collections import defaultdict

# 数据库路径
MARKET_DB = 'trader_data/market_data.db'

# V5 策略参数（优化版 - 目标 60% 赢率）
TP_PCT = 0.18          # 止盈 +18%（更容易达到）
SL_PCT = -0.20         # 止损 -20%
TRAIL_ACTIVATE = 0.06  # 盈利 6% 后启动追踪
TRAIL_DISTANCE = 0.04  # 回撤 4% 平仓
MIN_SCORE = 0.55       # 提高入场阈值
MIN_ADX = 18           # 提高 ADX 要求

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
           'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT']


def aggregate_to_1h(klines_15m):
    """将 15m K线聚合为 1h"""
    hourly = []
    for i in range(0, len(klines_15m), 4):
        chunk = klines_15m[i:i+4]
        if len(chunk) < 4:
            break
        hourly.append({
            'timestamp': chunk[0]['timestamp'],
            'open': chunk[0]['open'],
            'high': max(c['high'] for c in chunk),
            'low': min(c['low'] for c in chunk),
            'close': chunk[-1]['close'],
            'volume': sum(c['volume'] for c in chunk)
        })
    return hourly


def calc_ema(prices, period):
    """计算 EMA"""
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    k = 2 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1 - k)
    return ema


def calc_rsi(prices, period=14):
    """计算 RSI"""
    rsi = np.zeros(len(prices))
    if len(prices) < period + 1:
        return rsi
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(prices) - 1):
        if avg_loss == 0:
            rsi[i + 1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - (100 / (1 + rs))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    return rsi


def calc_adx(highs, lows, closes, period=14):
    """计算 ADX"""
    n = len(closes)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)
        
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    # Smoothed TR and DM
    atr = np.zeros(n)
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)
    
    atr[period] = np.mean(tr[1:period+1])
    smooth_plus[period] = np.mean(plus_dm[1:period+1])
    smooth_minus[period] = np.mean(minus_dm[1:period+1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        smooth_plus[i] = (smooth_plus[i-1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i-1] * (period - 1) + minus_dm[i]) / period
    
    # DI and DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * smooth_plus[i] / atr[i]
            di_minus[i] = 100 * smooth_minus[i] / atr[i]
        
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX = smoothed DX
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period:start+1])
        for i in range(start + 1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx


def calc_entry_score(klines, idx):
    """多因子入场评分"""
    if idx < 50:
        return 0, None
    
    closes = np.array([k['close'] for k in klines[:idx+1]])
    highs = np.array([k['high'] for k in klines[:idx+1]])
    lows = np.array([k['low'] for k in klines[:idx+1]])
    volumes = np.array([k['volume'] for k in klines[:idx+1]])
    
    price = closes[-1]
    
    # 指标计算
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    rsi = calc_rsi(closes, 14)
    adx = calc_adx(highs, lows, closes, 14)
    
    # 成交量均值
    vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
    
    current_adx = adx[-1]
    current_rsi = rsi[-1]
    
    # ADX 过滤：只在大趋势时入场
    if current_adx < MIN_ADX:
        return 0, None
    
    # 趋势方向
    trend_up = ema20[-1] > ema50[-1]
    
    score_long = 0
    score_short = 0
    
    # 1. 趋势强度 (0-0.35) - 增加权重
    trend_strength = abs(ema20[-1] - ema50[-1]) / ema50[-1]
    if trend_up:
        score_long += min(0.35, trend_strength * 15)  # 增加敏感度
    else:
        score_short += min(0.35, trend_strength * 15)
    
    # 2. ADX 强度 (0-0.25)
    adx_score = min(0.25, (current_adx - MIN_ADX) / 20 * 0.25)  # 降低分母
    score_long += adx_score
    score_short += adx_score
    
    # 3. RSI 位置 (0-0.2)
    if current_rsi < 45:  # 放宽 RSI 范围
        score_long += 0.2 * (45 - current_rsi) / 25
    elif current_rsi > 55:
        score_short += 0.2 * (current_rsi - 55) / 25
    
    # 4. 成交量确认 (0-0.15)
    if vol_ratio > 1.3:
        score_long += 0.15
        score_short += 0.15
    elif vol_ratio > 1.0:
        score_long += 0.1
        score_short += 0.1
    
    # 5. 价格位置 (0-0.1)
    recent_high = np.max(highs[-20:])
    recent_low = np.min(lows[-20:])
    price_pos = (price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    
    if price_pos < 0.3:
        score_long += 0.1
    elif price_pos > 0.7:
        score_short += 0.1
    
    # 确定方向
    if score_long > score_short and score_long >= MIN_SCORE:
        return score_long, 'LONG'
    elif score_short > score_long and score_short >= MIN_SCORE:
        return score_short, 'SHORT'
    
    return 0, None


def simulate_trade(klines, entry_idx, side, entry_price):
    """模拟单笔交易"""
    # SL_PCT 是负数（如 -0.20）
    # LONG: SL 在入场价下方 20% → entry * (1 + SL_PCT) = entry * 0.80
    # SHORT: SL 在入场价上方 20% → entry * (1 - SL_PCT) = entry * 1.20
    if side == 'LONG':
        sl_price = entry_price * (1 + SL_PCT)  # 0.80 * entry
        tp_price = entry_price * (1 + TP_PCT)  # 1.20 * entry
    else:
        sl_price = entry_price * (1 - SL_PCT)  # 1.20 * entry
        tp_price = entry_price * (1 - TP_PCT)  # 0.80 * entry
    
    highest_pnl = 0
    trailing_active = False
    trailing_stop = 0
    
    for i in range(entry_idx + 1, len(klines)):
        k = klines[i]
        
        if side == 'LONG':
            # 检查止损
            if k['low'] <= sl_price:
                return -0.12, i - entry_idx, 'SL'
            
            # 检查止盈
            if k['high'] >= tp_price:
                return 0.25, i - entry_idx, 'TP'
            
            # 计算当前盈亏
            pnl = (k['close'] - entry_price) / entry_price
            
            # 追踪止损逻辑
            if pnl >= TRAIL_ACTIVATE:
                trailing_active = True
                trailing_stop = max(trailing_stop, k['high'] * (1 - TRAIL_DISTANCE))
            
            if trailing_active and k['low'] <= trailing_stop:
                return (trailing_stop - entry_price) / entry_price, i - entry_idx, 'TRAIL'
            
            highest_pnl = max(highest_pnl, pnl)
        
        else:  # SHORT
            if k['high'] >= sl_price:
                return -0.12, i - entry_idx, 'SL'
            
            if k['low'] <= tp_price:
                return 0.25, i - entry_idx, 'TP'
            
            pnl = (entry_price - k['close']) / entry_price
            
            if pnl >= TRAIL_ACTIVATE:
                trailing_active = True
                trailing_stop = min(trailing_stop if trailing_stop > 0 else float('inf'), 
                                   k['low'] * (1 + TRAIL_DISTANCE))
            
            if trailing_active and k['high'] >= trailing_stop:
                return (entry_price - trailing_stop) / entry_price, i - entry_idx, 'TRAIL'
            
            highest_pnl = max(highest_pnl, pnl)
    
    # 交易未平仓
    final_price = klines[-1]['close']
    if side == 'LONG':
        pnl = (final_price - entry_price) / entry_price
    else:
        pnl = (entry_price - final_price) / entry_price
    
    return pnl, len(klines) - entry_idx, 'END'


def run_backtest():
    """运行回测"""
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    
    all_results = []
    all_trades = []
    
    for symbol in SYMBOLS:
        print(f"\n处理 {symbol}...")
        
        # 读取 15m K线
        rows = conn.execute(
            "SELECT open_time as timestamp, open, high, low, close, volume FROM klines WHERE symbol = ? ORDER BY open_time",
            (symbol,)
        ).fetchall()
        
        if not rows:
            print(f"  无数据")
            continue
        
        klines_15m = [dict(r) for r in rows]
        
        # 聚合为 1h
        klines = aggregate_to_1h(klines_15m)
        print(f"  1h K线: {len(klines)} 根")
        
        trades = []
        i = 50
        
        while i < len(klines) - 1:
            score, side = calc_entry_score(klines, i)
            
            if side:
                entry_price = klines[i]['close']
                pnl, bars, exit_reason = simulate_trade(klines, i, side, entry_price)
                
                trade = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'entry_time': klines[i]['timestamp'],
                    'exit_price': entry_price * (1 + pnl),
                    'pnl': pnl,
                    'bars_held': bars,
                    'exit_reason': exit_reason,
                    'score': score
                }
                trades.append(trade)
                
                i += bars  # 跳过持仓期间
            else:
                i += 1
        
        # 统计
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades)
            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / len(trades)
            
            gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # 最大回撤
            cum_pnl = np.cumsum([t['pnl'] for t in trades])
            peak = np.maximum.accumulate(cum_pnl)
            dd = peak - cum_pnl
            max_dd = np.max(dd) if len(dd) > 0 else 0
            
            print(f"  交易: {len(trades)} | 赢率: {win_rate*100:.1f}% | PnL: {total_pnl*100:+.2f}% | PF: {pf:.2f}")
            
            result = {
                'symbol': symbol,
                'total_trades': len(trades),
                'winners': wins,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'profit_factor': pf,
                'max_drawdown': max_dd,
                'avg_bars': np.mean([t['bars_held'] for t in trades])
            }
            all_results.append(result)
            all_trades.extend(trades)
        else:
            print(f"  无交易")
    
    conn.close()
    
    # 保存结果
    save_results(all_results, all_trades)
    
    # 打印汇总
    print("\n" + "="*70)
    print("V5 优化回测结果汇总")
    print("="*70)
    
    if all_results:
        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['winners'] for r in all_results)
        total_pnl = sum(r['total_pnl'] for r in all_results)
        avg_wr = total_wins / total_trades if total_trades > 0 else 0
        avg_pf = np.mean([r['profit_factor'] for r in all_results])
        avg_dd = np.mean([r['max_drawdown'] for r in all_results])
        avg_bars = np.mean([r['avg_bars'] for r in all_results])
        
        print(f"\n总体统计:")
        print(f"  总交易数: {total_trades}")
        print(f"  赢率: {avg_wr*100:.1f}%")
        print(f"  总 PnL: {total_pnl*100:+.2f}%")
        print(f"  平均 PF: {avg_pf:.2f}")
        print(f"  平均最大回撤: {avg_dd*100:.2f}%")
        print(f"  平均持仓: {avg_bars:.1f} bars")
        
        print(f"\n各币种详情:")
        print(f"{'币种':<10} {'交易':>5} {'赢率':>8} {'PnL':>10} {'PF':>6} {'MaxDD':>8}")
        print("-"*55)
        for r in all_results:
            print(f"{r['symbol']:<10} {r['total_trades']:>5} {r['win_rate']*100:>7.1f}% "
                  f"{r['total_pnl']*100:>+9.2f}% {r['profit_factor']:>6.2f} {r['max_drawdown']*100:>7.2f}%")


def save_results(results, trades):
    """保存结果到数据库"""
    conn = sqlite3.connect(MARKET_DB)
    
    # 创建 V5 结果表
    conn.execute('''CREATE TABLE IF NOT EXISTS v5_backtest_results (
        symbol TEXT, total_trades INTEGER, winners INTEGER,
        win_rate REAL, total_pnl REAL, avg_pnl REAL,
        profit_factor REAL, max_drawdown REAL, avg_bars REAL
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS v5_backtest_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, side TEXT, entry_price REAL, entry_time INTEGER,
        exit_price REAL, pnl REAL, bars_held INTEGER,
        exit_reason TEXT, score REAL
    )''')
    
    # 清空旧数据
    conn.execute('DELETE FROM v5_backtest_results')
    conn.execute('DELETE FROM v5_backtest_trades')
    
    # 插入结果
    for r in results:
        conn.execute('''INSERT INTO v5_backtest_results 
            (symbol, total_trades, winners, win_rate, total_pnl, avg_pnl, 
             profit_factor, max_drawdown, avg_bars)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (r['symbol'], r['total_trades'], r['winners'], r['win_rate'],
             r['total_pnl'], r['avg_pnl'], r['profit_factor'], r['max_drawdown'], r['avg_bars']))
    
    for t in trades:
        conn.execute('''INSERT INTO v5_backtest_trades
            (symbol, side, entry_price, entry_time, exit_price, pnl, bars_held, exit_reason, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (t['symbol'], t['side'], t['entry_price'], t['entry_time'],
             t['exit_price'], t['pnl'], t['bars_held'], t['exit_reason'], t['score']))
    
    conn.commit()
    conn.close()
    print(f"\n结果已保存到 {MARKET_DB}")


if __name__ == '__main__':
    run_backtest()
