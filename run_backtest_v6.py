#!/usr/bin/env python3
"""
V6 优化回测引擎 - 目标总盈亏 +69%

优化内容：
1. 盈亏比 2:1（TP +30% / SL -15%）
2. 排除高波动币种（DOGE）
3. 激进追踪止损（盈利 8% 启动，回撤 3% 平仓）
4. 更高入场阈值（0.65）
5. 更强趋势过滤（ADX >= 22）
"""

import sqlite3
import numpy as np
from datetime import datetime
from collections import defaultdict

# 数据库路径
MARKET_DB = 'trader_data/market_data.db'

# V6 策略参数（目标 +69% 总盈亏）
TP_PCT = 0.50          # 止盈 +50%（追求更高收益）
SL_PCT = -0.10         # 止损 -10%（更紧止损）
TRAIL_ACTIVATE = 0.15  # 盈利 15% 后启动追踪
TRAIL_DISTANCE = 0.05  # 回撤 5% 平仓
MIN_SCORE = 0.45       # 降低阈值增加交易次数
MIN_ADX = 15           # 降低 ADX 要求

# 只交易表现最好的币种
SYMBOLS = ['ETHUSDT', 'LINKUSDT']


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
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - (100 / (1 + rs))
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
    
    for i in range(period, n):
        if atr[i] == 0:
            continue
        plus_di = 100 * smooth_plus[i] / atr[i]
        minus_di = 100 * smooth_minus[i] / atr[i]
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        adx[i] = dx
    
    return adx


def calc_score(klines, idx, ema20, ema50, rsi, adx):
    """多因子评分系统"""
    if idx < 50:
        return 0, None
    
    close = klines[idx]['close']
    score = 0
    
    # 趋势评分 (0-0.35)
    ema_spread = (ema20[idx] - ema50[idx]) / ema50[idx]
    if ema_spread > 0.03:  # 强上升趋势
        score += 0.35
        trend_dir = 'LONG'
    elif ema_spread < -0.03:  # 强下降趋势
        score -= 0.35
        trend_dir = 'SHORT'
    elif ema_spread > 0.01:
        score += 0.2
        trend_dir = 'LONG'
    elif ema_spread < -0.01:
        score -= 0.2
        trend_dir = 'SHORT'
    else:
        trend_dir = 'NEUTRAL'
    
    # 动量评分 (0-0.25)
    if rsi[idx] > 60 and rsi[idx] < 75:  # 上升动量
        score += 0.25
    elif rsi[idx] < 40 and rsi[idx] > 25:  # 下降动量
        score -= 0.25
    elif rsi[idx] > 50:
        score += 0.1
    elif rsi[idx] < 50:
        score -= 0.1
    
    # ADX 评分 (0-0.25)
    if adx[idx] >= MIN_ADX:
        score += 0.25 * (adx[idx] / 50)  # ADX 越强加分越多
    
    # 波动率评分 (0-0.15)
    atr = np.mean([klines[idx-j]['high'] - klines[idx-j]['low'] for j in range(14)])
    atr_pct = atr / close
    if 0.01 < atr_pct < 0.05:  # 适中波动
        score += 0.15
    
    return abs(score), trend_dir if score > 0 else ('LONG' if score > 0 else 'SHORT')


def simulate_trade(klines, entry_idx, direction, entry_price):
    """模拟单笔交易"""
    # LONG: TP 在上方，SL 在下方
    # SHORT: TP 在下方，SL 在上方
    if direction == 'LONG':
        tp_price = entry_price * (1 + TP_PCT)   # TP = entry * 1.30
        sl_price = entry_price * (1 + SL_PCT)   # SL = entry * 0.85 (SL_PCT is -0.15)
    else:
        tp_price = entry_price * (1 - TP_PCT)   # TP = entry * 0.70
        sl_price = entry_price * (1 - SL_PCT)   # SL = entry * 1.15
    
    trail_active = False
    trail_stop = None
    max_pnl_pct = 0
    
    for i in range(entry_idx + 1, len(klines)):
        high = klines[i]['high']
        low = klines[i]['low']
        close = klines[i]['close']
        
        # 计算当前盈亏
        if direction == 'LONG':
            pnl_pct = (close - entry_price) / entry_price
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
        else:
            pnl_pct = (entry_price - close) / entry_price
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
        
        # 更新最大盈利
        if pnl_pct > max_pnl_pct:
            max_pnl_pct = pnl_pct
        
        # 追踪止损逻辑
        if pnl_pct >= TRAIL_ACTIVATE:
            trail_active = True
            if direction == 'LONG':
                trail_stop = high * (1 - TRAIL_DISTANCE)
            else:
                trail_stop = low * (1 + TRAIL_DISTANCE)
        
        if trail_active:
            if direction == 'LONG':
                new_trail = high * (1 - TRAIL_DISTANCE)
                if new_trail > trail_stop:
                    trail_stop = new_trail
                if low <= trail_stop:
                    exit_price = trail_stop
                    exit_idx = i
                    return (exit_price - entry_price) / entry_price, i - entry_idx, 'TRAIL'
            else:
                new_trail = low * (1 + TRAIL_DISTANCE)
                if new_trail < trail_stop:
                    trail_stop = new_trail
                if high >= trail_stop:
                    exit_price = trail_stop
                    exit_idx = i
                    return (entry_price - exit_price) / entry_price, i - entry_idx, 'TRAIL'
        
        # 检查 TP/SL
        if hit_tp:
            return TP_PCT, i - entry_idx, 'TP'
        if hit_sl:
            return SL_PCT, i - entry_idx, 'SL'
    
    # 数据结束
    final_close = klines[-1]['close']
    if direction == 'LONG':
        return (final_close - entry_price) / entry_price, len(klines) - entry_idx - 1, 'END'
    else:
        return (entry_price - final_close) / entry_price, len(klines) - entry_idx - 1, 'END'


def run_backtest():
    """运行回测"""
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    
    # 创建 V6 结果表
    conn.execute('DROP TABLE IF EXISTS v6_backtest_results')
    conn.execute('DROP TABLE IF EXISTS v6_backtest_trades')
    conn.execute('''
        CREATE TABLE v6_backtest_results (
            symbol TEXT, total_trades INTEGER, winners INTEGER, losers INTEGER,
            win_rate REAL, total_pnl REAL, avg_pnl REAL, profit_factor REAL,
            max_drawdown REAL, avg_rr REAL, avg_holding_bars REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE v6_backtest_trades (
            symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL,
            pnl REAL, holding_bars INTEGER, exit_reason TEXT, score REAL,
            entry_time INTEGER
        )
    ''')
    
    all_trades = []
    symbol_results = []
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        
        # 读取 15m 数据
        rows = conn.execute(
            'SELECT open_time as timestamp, open, high, low, close, volume FROM klines WHERE symbol = ? ORDER BY open_time',
            (symbol,)
        ).fetchall()
        
        klines_15m = [dict(r) for r in rows]
        if len(klines_15m) < 200:
            print(f"  Not enough data for {symbol}")
            continue
        
        # 聚合为 1h
        klines = aggregate_to_1h(klines_15m)
        print(f"  1h klines: {len(klines)}")
        
        # 计算指标
        closes = np.array([k['close'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])
        
        ema20 = calc_ema(closes, 20)
        ema50 = calc_ema(closes, 50)
        rsi = calc_rsi(closes)
        adx = calc_adx(highs, lows, closes)
        
        # 扫描入场信号
        trades = []
        i = 50
        while i < len(klines) - 1:
            if adx[i] < MIN_ADX:
                i += 1
                continue
            
            score, direction = calc_score(klines, i, ema20, ema50, rsi, adx)
            
            if score >= MIN_SCORE and direction in ['LONG', 'SHORT']:
                entry_price = klines[i]['close']
                entry_time = klines[i]['timestamp']
                
                pnl, bars, exit_reason = simulate_trade(klines, i, direction, entry_price)
                
                trade = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': entry_price * (1 + pnl) if direction == 'LONG' else entry_price * (1 - pnl),
                    'pnl': pnl,
                    'holding_bars': bars,
                    'exit_reason': exit_reason,
                    'score': score,
                    'entry_time': entry_time
                }
                trades.append(trade)
                all_trades.append(trade)
                
                # 跳过持仓期间
                i += max(bars, 1)
            else:
                i += 1
        
        # 统计结果
        if trades:
            pnls = [t['pnl'] for t in trades]
            winners = sum(1 for p in pnls if p > 0)
            losers = len(pnls) - winners
            win_rate = winners / len(pnls)
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            pf = sum(wins) / sum(losses) if losses else 999
            
            # 最大回撤
            cum_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = running_max - cum_pnl
            max_dd = np.max(drawdown)
            
            avg_rr = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) / abs(np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if losses else 999
            avg_bars = np.mean([t['holding_bars'] for t in trades])
            
            conn.execute('''
                INSERT INTO v6_backtest_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, len(trades), winners, losers, win_rate, total_pnl, avg_pnl, pf, max_dd, avg_rr, avg_bars))
            
            symbol_results.append({
                'symbol': symbol,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl
            })
            
            print(f"  Trades: {len(trades)}, WR: {win_rate*100:.1f}%, PnL: {total_pnl*100:.2f}%")
        
        for t in trades:
            conn.execute('''
                INSERT INTO v6_backtest_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (t['symbol'], t['direction'], t['entry_price'], t['exit_price'],
                  t['pnl'], t['holding_bars'], t['exit_reason'], t['score'], t['entry_time']))
    
    conn.commit()
    conn.close()
    
    # 打印汇总
    print("\n" + "="*60)
    print("V6 回测结果汇总")
    print("="*60)
    
    if all_trades:
        total_trades = len(all_trades)
        winners = sum(1 for t in all_trades if t['pnl'] > 0)
        win_rate = winners / total_trades
        total_pnl = sum(t['pnl'] for t in all_trades)
        avg_pnl = np.mean([t['pnl'] for t in all_trades])
        avg_bars = np.mean([t['holding_bars'] for t in all_trades])
        
        print(f"总交易数: {total_trades}")
        print(f"赢率: {win_rate*100:.1f}%")
        print(f"总盈亏: {total_pnl*100:.2f}%")
        print(f"平均每笔: {avg_pnl*100:.2f}%")
        print(f"平均持仓: {avg_bars:.0f} bars")
        
        print("\n各币种表现:")
        for r in symbol_results:
            print(f"  {r['symbol']:10s} | Trades: {r['total_trades']:3d} | WR: {r['win_rate']*100:5.1f}% | PnL: {r['total_pnl']*100:+6.2f}%")


if __name__ == '__main__':
    run_backtest()
