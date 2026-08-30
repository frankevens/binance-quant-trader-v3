"""
Quick win-rate ceiling test - what's the max achievable win rate?
Uses simplified scoring to find the theoretical maximum.
"""
import numpy as np
import itertools

def gen_market(n, regime, vol_base=0.015):
    np.random.seed(42)
    prices = [100.0]
    drift = {'bull': 0.001, 'bear': -0.001, 'range': 0.0, 'volatile': 0.0}[regime]
    vol = {'bull': vol_base, 'bear': vol_base, 'range': vol_base * 0.7, 'volatile': vol_base * 2.2}[regime]
    for i in range(1, n):
        shock = np.random.normal(0, vol)
        if regime == 'range':
            shock -= (prices[-1] - 100) * 0.003
        p = prices[-1] * (1 + drift + shock)
        prices.append(p)
    return np.array(prices)

def ema(data, period):
    out = np.zeros_like(data, dtype=float)
    out[0] = data[0]
    k = 2 / (period + 1)
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def calc_atr_simple(prices, period=14):
    atr = np.zeros(len(prices))
    for i in range(1, len(prices)):
        tr = abs(prices[i] - prices[i-1])
        atr[i] = tr if i < period else atr[i-1] * (period-1)/period + tr/period
    atr[:period] = atr[period] if atr[period] > 0 else 0.001
    return atr

def simulate_simple(prices, min_score, sl_mult, tp_rr):
    """Simplified: only trend + momentum signals"""
    ema10 = ema(prices, 10)
    ema30 = ema(prices, 30)
    atr = calc_atr_simple(prices)
    
    trades = []
    in_pos = False
    entry_p = sl_p = tp_p = 0.0
    pos_dir = 1
    
    for i in range(35, len(prices)):
        if atr[i] <= 0: continue
        p = prices[i]
        
        if in_pos:
            if pos_dir == 1:
                if p >= tp_p:
                    trades.append(tp_rr * sl_mult)
                    in_pos = False
                elif p <= sl_p:
                    trades.append(-sl_mult)
                    in_pos = False
            else:
                if p <= tp_p:
                    trades.append(tp_rr * sl_mult)
                    in_pos = False
                elif p >= sl_p:
                    trades.append(-sl_mult)
                    in_pos = False
            continue
        
        # Simple scoring
        long_score = 0.0
        short_score = 0.0
        
        # Trend
        if ema10[i] > ema30[i]:
            long_score += 0.4
        else:
            short_score += 0.4
        
        # Momentum (price vs EMA)
        if p > ema10[i]:
            long_score += 0.3
        else:
            short_score += 0.3
        
        # Mean reversion (counter-trend)
        if p < ema30[i] * 0.98:
            long_score += 0.3  # Oversold
        elif p > ema30[i] * 1.02:
            short_score += 0.3  # Overbought
        
        # Only trade in direction of trend when score is high enough
        if long_score >= min_score and long_score > short_score:
            in_pos = True; pos_dir = 1; entry_p = p
            sl_p = p - atr[i] * sl_mult
            tp_p = p + atr[i] * sl_mult * tp_rr
        elif short_score >= min_score and short_score > long_score:
            in_pos = True; pos_dir = -1; entry_p = p
            sl_p = p + atr[i] * sl_mult
            tp_p = p - atr[i] * sl_mult * tp_rr
    
    return trades

# Quick scan
scores = [0.3, 0.4, 0.5, 0.6, 0.7]
sls = [0.5, 0.8, 1.0, 1.5]
tps = [1.0, 1.5, 2.0, 3.0, 4.0]

print("Scanning for max win rate...\n")
print(f"{'Score':>6} {'SL':>5} {'TP':>5} | {'WR%':>6} {'R:R':>6} {'Exp':>7} {'Trades':>6} {'Ret%':>8}")
print("-" * 65)

results = []
for sc, sl, tp in itertools.product(scores, sls, tps):
    all_trades = []
    for regime in ['bull', 'bear', 'range', 'volatile']:
        prices = gen_market(1000, regime)
        trades = simulate_simple(prices, sc, sl, tp)
        all_trades.extend(trades)
    
    if not all_trades:
        continue
    
    wins = sum(1 for t in all_trades if t > 0)
    wr = wins / len(all_trades) * 100
    exp = np.mean(all_trades)
    ret = sum(all_trades) * 2
    
    results.append((sc, sl, tp, wr, tp, exp, len(all_trades), ret))

# Sort by win rate
results.sort(key=lambda x: x[3], reverse=True)

for sc, sl, tp, wr, rr, exp, count, ret in results[:30]:
    print(f"{sc:>6.1f} {sl:>5.1f} {tp:>5.1f} | {wr:>5.1f}% {tp:>5.1f}:1 {exp:>+6.3f}R {count:>6} {ret:>+7.1f}%")

# Find best with positive expectancy
print("\n" + "=" * 65)
print("BEST WITH POSITIVE EXPECTANCY:")
print("=" * 65)
pos_results = [r for r in results if r[5] > 0 and r[6] >= 5]
if pos_results:
    pos_results.sort(key=lambda x: (x[3], x[5]), reverse=True)
    best = pos_results[0]
    sc, sl, tp, wr, rr, exp, count, ret = best
    print(f"Score={sc}, SL={sl}, TP={tp}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"R:R: {tp:.1f}:1")
    print(f"Expectancy: {exp:+.3f}R/trade")
    print(f"Trades: {count}")
    print(f"Return: {ret:+.1f}%")
else:
    print("No positive expectancy combos found.")

# Check if 56.8% is achievable
print("\n" + "=" * 65)
print("TARGET 56.8% WIN RATE:")
print("=" * 65)
target = [r for r in results if r[3] >= 56.8 and r[5] > 0]
if target:
    target.sort(key=lambda x: x[3], reverse=True)
    best = target[0]
    sc, sl, tp, wr, rr, exp, count, ret = best
    print(f"ACHIEVABLE! Score={sc}, SL={sl}, TP={tp}")
    print(f"Win Rate: {wr:.1f}%, R:R={tp:.1f}:1, Exp={exp:+.3f}R, Ret={ret:+.1f}%")
else:
    best_wr = max(r[3] for r in results if r[5] > 0) if any(r[5] > 0 for r in results) else 0
    print(f"56.8% NOT achievable with positive expectancy.")
    print(f"Max achievable: {best_wr:.1f}%")
