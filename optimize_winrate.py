"""
Win-rate focused parameter optimization for Binance Quant Trader V3.
Scans a wider range of entry score thresholds (0.60-0.90) to find
the combination that maximizes win rate while maintaining positive expectancy.
"""
import numpy as np
import itertools
import sys

# ---------- Synthetic market data ----------
def gen_market(n, regime, vol_base=0.015):
    np.random.seed(42)
    prices = [100.0]
    volumes = []
    highs = []
    lows = []
    drift = {'bull': 0.0008, 'bear': -0.0008, 'range': 0.0, 'volatile': 0.0}[regime]
    vol = {'bull': vol_base, 'bear': vol_base, 'range': vol_base * 0.7, 'volatile': vol_base * 2.2}[regime]
    for i in range(1, n):
        shock = np.random.normal(0, vol)
        if regime == 'range':
            shock -= (prices[-1] - 100) * 0.005
        elif regime == 'volatile':
            if np.random.random() < 0.05:
                shock *= 3
        ret = drift + shock
        p = prices[-1] * (1 + ret)
        h = p * (1 + abs(np.random.normal(0, vol * 0.3)))
        l = p * (1 - abs(np.random.normal(0, vol * 0.3)))
        prices.append(p); highs.append(h); lows.append(l)
        base_v = {'bull': 1200, 'bear': 1500, 'range': 800, 'volatile': 2000}[regime]
        volumes.append(base_v * (1 + abs(shock) * 20) * np.random.uniform(0.6, 1.5))
    # Ensure all arrays have same length
    min_len = min(len(prices), len(volumes), len(highs), len(lows))
    return np.array(prices[:min_len]), np.array(volumes[:min_len]), np.array(highs[:min_len]), np.array(lows[:min_len])

# ---------- Indicators ----------
def ema(data, period):
    out = np.zeros_like(data, dtype=float)
    out[0] = data[0]
    k = 2 / (period + 1)
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def calc_rsi(prices, period=14):
    rsi = np.full(len(prices), 50.0)
    for i in range(period, len(prices)):
        d = np.diff(prices[i-period:i+1])
        g = np.sum(d[d > 0]) / period
        l = np.abs(np.sum(d[d < 0])) / period
        if l == 0: rsi[i] = 100
        else: rs = g / l; rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def calc_atr(highs, lows, closes, period=14):
    atr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr[i] = tr if i < period else atr[i-1] * (period-1)/period + tr/period
    atr[:period] = atr[period] if atr[period] > 0 else 0.001
    return atr

def bbands(prices, period=20, std_mult=2.0):
    upper = np.copy(prices)
    lower = np.copy(prices)
    mid = np.copy(prices)
    for i in range(period, len(prices)):
        w = prices[i-period:i]
        m = np.mean(w)
        s = np.std(w)
        mid[i] = m; upper[i] = m + std_mult * s; lower[i] = m - std_mult * s
    return upper, mid, lower

def htf_ema(prices, period=20):
    return ema(prices, period)

# ---------- Simulate with regime filter ----------
def simulate(prices, volumes, highs, lows, params):
    min_score = params['min_score']
    sl_mult = params['sl_mult']
    tp1_rr = params['tp1_rr']
    tp2_rr = params['tp2_rr']
    tp3_rr = params['tp3_rr']

    ema20 = ema(prices, 20)
    ema50 = ema(prices, 50)
    rsi = calc_rsi(prices)
    atr = calc_atr(highs, lows, prices)
    bb_u, bb_m, bb_l = bbands(prices)
    htf_e20 = htf_ema(prices, 20)
    htf_e50 = htf_ema(prices, 50)
    vol_ma = np.convolve(volumes, np.ones(20)/20, mode='same')

    trades = []
    in_pos = False
    entry_p = sl_p = tp1_p = tp2_p = tp3_p = trail_p = 0.0
    pos_dir = 1
    tp1_hit = tp2_hit = False
    rem_qty = 1.0

    for i in range(55, len(prices)):
        if atr[i] <= 0: continue
        p = prices[i]

        # Regime detection
        htf_spread = (htf_e20[i] - htf_e50[i]) / htf_e50[i] * 100
        if p > htf_e20[i] > htf_e50[i] and htf_spread > 2:
            regime = 'strong_bull'
        elif htf_e20[i] > htf_e50[i] and p > htf_e50[i]:
            regime = 'bull'
        elif p < htf_e20[i] < htf_e50[i] and htf_spread < -2:
            regime = 'strong_bear'
        elif htf_e20[i] < htf_e50[i] and p < htf_e50[i]:
            regime = 'bear'
        else:
            regime = 'range'

        if in_pos:
            if pos_dir == 1:
                if not tp1_hit and p >= tp1_p:
                    tp1_hit = True; rem_qty = 0.5; sl_p = entry_p
                    pnl = 0.5 * tp1_rr * sl_mult
                    trades.append(pnl); in_pos = False
                elif p <= sl_p:
                    pnl = rem_qty * (sl_p - entry_p) / (atr[i] * sl_mult) * sl_mult
                    trades.append(pnl); in_pos = False
            else:
                if not tp1_hit and p <= tp1_p:
                    tp1_hit = True; rem_qty = 0.5; sl_p = entry_p
                    pnl = 0.5 * tp1_rr * sl_mult
                    trades.append(pnl); in_pos = False
                elif p >= sl_p:
                    pnl = rem_qty * (entry_p - sl_p) / (atr[i] * sl_mult) * sl_mult
                    trades.append(pnl); in_pos = False
            continue

        # LONG scoring
        long_score = 0.0
        if rsi[i] < 35: long_score += 0.15
        elif rsi[i] < 45: long_score += 0.25
        if ema20[i] > ema50[i]: long_score += 0.20
        if p < bb_l[i]: long_score += 0.20
        elif p < bb_m[i]: long_score += 0.10
        if volumes[i] > vol_ma[i] * 1.5: long_score += 0.15
        elif volumes[i] < vol_ma[i] * 0.5: long_score -= 0.10
        if htf_e20[i] > htf_e50[i]: long_score += 0.15
        if regime in ('strong_bull', 'bull'):
            long_score += 0.15
        elif regime in ('strong_bear', 'bear'):
            long_score *= 0.1

        # SHORT scoring
        short_score = 0.0
        if rsi[i] > 65: short_score += 0.15
        elif rsi[i] > 55: short_score += 0.25
        if ema20[i] < ema50[i]: short_score += 0.20
        if p > bb_u[i]: short_score += 0.20
        elif p > bb_m[i]: short_score += 0.10
        if volumes[i] > vol_ma[i] * 1.5: short_score += 0.15
        elif volumes[i] < vol_ma[i] * 0.5: short_score -= 0.10
        if htf_e20[i] < htf_e50[i]: short_score += 0.15
        if regime in ('strong_bear', 'bear'):
            short_score += 0.15
        elif regime in ('strong_bull', 'bull'):
            short_score *= 0.1

        if long_score >= min_score and long_score > short_score:
            in_pos = True; pos_dir = 1; entry_p = p
            sl_p = p - atr[i] * sl_mult
            tp1_p = p + atr[i] * sl_mult * tp1_rr
            tp2_p = p + atr[i] * sl_mult * tp2_rr
            tp3_p = p + atr[i] * sl_mult * tp3_rr
            trail_p = p - atr[i] * 2.0
            tp1_hit = tp2_hit = False; rem_qty = 1.0
        elif short_score >= min_score and short_score > long_score:
            in_pos = True; pos_dir = -1; entry_p = p
            sl_p = p + atr[i] * sl_mult
            tp1_p = p - atr[i] * sl_mult * tp1_rr
            tp2_p = p - atr[i] * sl_mult * tp2_rr
            tp3_p = p - atr[i] * sl_mult * tp3_rr
            trail_p = p + atr[i] * 2.0
            tp1_hit = tp2_hit = False; rem_qty = 1.0

    return trades

def analyze(trades):
    if not trades:
        return {'count': 0, 'wr': 0, 'rr': 0, 'exp': 0, 'pf': 0, 'ret': 0, 'mdd': 0, 'sharpe': 0}
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / len(trades) * 100
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0.01
    rr = avg_w / avg_l if avg_l > 0 else 0
    exp = np.mean(trades)
    gross_w = sum(wins) if wins else 0
    gross_l = abs(sum(losses)) if losses else 0.01
    pf = gross_w / gross_l if gross_l > 0 else 0
    ret = sum(trades) * 2
    peak = 0; mdd = 0; eq = 0
    for t in trades:
        eq += t
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > mdd: mdd = dd
    sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252) if np.std(trades) > 0 else 0
    return {'count': len(trades), 'wr': wr, 'rr': rr, 'exp': exp, 'pf': pf, 'ret': ret, 'mdd': mdd * 2 * 100, 'sharpe': sharpe}

# ---------- Main ----------
if __name__ == '__main__':
    N = 1000
    markets = {}
    for regime in ['bull', 'bear', 'range', 'volatile']:
        p, v, h, l = gen_market(N, regime)
        markets[regime] = (p, v, h, l)

    # Wider score range to find high win rate
    scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sls = [0.5, 0.7, 0.8, 1.0]
    tp1s = [1.0, 1.5, 2.0]
    tp2s = [2.0, 3.0]
    tp3s = [5.0, 8.0]

    combos = list(itertools.product(scores, sls, tp1s, tp2s, tp3s))
    print(f"Scanning {len(combos)} combos (win-rate focused)...\n")

    results = []
    for sc, sl, t1, t2, t3 in combos:
        params = {'min_score': sc, 'sl_mult': sl, 'tp1_rr': t1, 'tp2_rr': t2, 'tp3_rr': t3}
        all_t = []
        per_m = {}
        for regime, (p, v, h, l) in markets.items():
            trades = simulate(p, v, h, l, params)
            all_t.extend(trades)
            per_m[regime] = analyze(trades)
        overall = analyze(all_t)
        overall['params'] = params
        overall['per_market'] = per_m
        results.append(overall)

    # Sort by win rate (primary), then expectancy (secondary)
    results.sort(key=lambda x: (x['wr'], x['exp']), reverse=True)

    print("=" * 90)
    print("  TOP 20 BY WIN RATE (must have positive expectancy)")
    print("=" * 90)
    print(f"{'#':>3} {'Score':>6} {'SL':>5} {'TP1':>5} {'TP2':>5} {'TP3':>5} | {'WR%':>6} {'R:R':>6} {'Exp':>7} {'Trades':>6} {'PF':>6} {'Ret%':>8} {'MDD%':>7}")
    print("-" * 90)
    shown = 0
    for r in results:
        if r['exp'] <= 0: continue
        if r['count'] < 5: continue
        shown += 1
        if shown > 20: break
        p = r['params']
        print(f"{shown:>3} {p['min_score']:>6.2f} {p['sl_mult']:>5.1f} {p['tp1_rr']:>5.1f} {p['tp2_rr']:>5.1f} {p['tp3_rr']:>5.1f} | "
              f"{r['wr']:>5.1f}% {r['rr']:>5.2f}:1 {r['exp']:>+6.3f}R {r['count']:>6} {r['pf']:>5.2f} {r['ret']:>+7.1f}% {r['mdd']:>6.1f}%")

    # Find the best combo that hits 56.8% win rate
    print("\n" + "=" * 90)
    print("  TARGET: 56.8% WIN RATE")
    print("=" * 90)
    target_wr = 56.8
    best = None
    for r in results:
        if r['wr'] >= target_wr and r['exp'] > 0 and r['count'] >= 3:
            best = r
            break

    if best:
        p = best['params']
        print(f"\n  FOUND! Score={p['min_score']}, SL={p['sl_mult']}, TP1={p['tp1_rr']}, TP2={p['tp2_rr']}, TP3={p['tp3_rr']}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  R:R:      {best['rr']:.2f}:1")
        print(f"  Expect:   {best['exp']:+.3f}R/trade")
        print(f"  Trades:   {best['count']}")
        print(f"  PF:       {best['pf']:.2f}")
        print(f"  Return:   {best['ret']:+.1f}%")
        print(f"  MaxDD:    {best['mdd']:.1f}%")
        print(f"\n  Per-market breakdown:")
        for regime in ['bull', 'bear', 'range', 'volatile']:
            m = best['per_market'][regime]
            print(f"    {regime:>10}: WR={m['wr']:.1f}%, R:R={m['rr']:.2f}:1, Exp={m['exp']:+.3f}R, Ret={m['ret']:+.1f}%")
    else:
        # Show the closest we can get
        best_pos = [r for r in results if r['exp'] > 0 and r['count'] >= 3]
        if best_pos:
            best_pos.sort(key=lambda x: x['wr'], reverse=True)
            closest = best_pos[0]
            p = closest['params']
            print(f"\n  56.8% NOT ACHIEVABLE with positive expectancy.")
            print(f"  Best achievable: WR={closest['wr']:.1f}% (Score={p['min_score']}, SL={p['sl_mult']}, TP1={p['tp1_rr']}, TP2={p['tp2_rr']}, TP3={p['tp3_rr']})")
            print(f"  R:R={closest['rr']:.2f}:1, Exp={closest['exp']:+.3f}R, Trades={closest['count']}, Ret={closest['ret']:+.1f}%")
            print(f"\n  Per-market breakdown:")
            for regime in ['bull', 'bear', 'range', 'volatile']:
                m = closest['per_market'][regime]
                print(f"    {regime:>10}: WR={m['wr']:.1f}%, R:R={m['rr']:.2f}:1, Exp={m['exp']:+.3f}R, Ret={m['ret']:+.1f}%")
        else:
            print("\n  No positive expectancy combos found.")

    print()
