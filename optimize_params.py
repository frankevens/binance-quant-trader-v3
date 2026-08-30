#!/usr/bin/env python3
"""
Binance Quant Trader V3 — Parameter Optimization
Sweeps through parameter combinations to find optimal settings.
"""

import numpy as np
import itertools
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ParamSet:
    min_score: float
    sl_mult: float
    tp1_rr: float
    tp2_rr: float
    tp3_rr: float
    trail_mult: float


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    blended_rr: float
    expectancy: float
    total_return: float
    max_drawdown: float
    profit_factor: float


def generate_market_data(n: int = 500, regime: str = "mixed", seed: int = 42) -> np.ndarray:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)

    if regime == "bull":
        drift = 0.0008
        vol = 0.015
    elif regime == "bear":
        drift = -0.0006
        vol = 0.018
    elif regime == "volatile":
        drift = 0.0
        vol = 0.03
    else:
        drift = 0.0002
        vol = 0.015

    returns = rng.normal(drift, vol, n)
    price = 100.0 * np.exp(np.cumsum(returns))

    data = []
    for i in range(n):
        o = price[i]
        h = o * (1 + abs(rng.normal(0, vol * 0.5)))
        l = o * (1 - abs(rng.normal(0, vol * 0.5)))
        c = price[i]
        v = rng.lognormal(10, 1.5)
        data.append([o, h, l, c, v])

    return np.array(data)


def compute_atr(data: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(data)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(
            data[i][1] - data[i][2],
            abs(data[i][1] - data[i - 1][3]),
            abs(data[i][2] - data[i - 1][3]),
        )
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr


def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.full_like(prices, 50.0)
    deltas = np.diff(prices)
    for i in range(period, len(prices)):
        window = deltas[i - period : i]
        gains = np.maximum(window, 0)
        losses = np.maximum(-window, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi


def simulate_backtest(
    data: np.ndarray, params: ParamSet
) -> BacktestResult:
    """Run backtest with given parameters."""
    n = len(data)
    closes = data[:, 3]
    highs = data[:, 1]
    lows = data[:, 2]
    volumes = data[:, 4]

    atr = compute_atr(data)
    ema_fast = compute_ema(closes, 10)
    ema_slow = compute_ema(closes, 30)
    rsi = compute_rsi(closes)

    # Bollinger Bands
    bb_period = 20
    bb_sma = np.convolve(closes, np.ones(bb_period) / bb_period, mode="same")
    bb_std = np.array([
        np.std(closes[max(0, i - bb_period + 1) : i + 1])
        for i in range(len(closes))
    ])
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std

    # Volume MA
    vol_ma = np.convolve(volumes, np.ones(20) / 20, mode="same")

    # Simulate trades
    trades: List[float] = []
    in_position = False
    entry_price = 0.0
    position_side = ""
    entry_atr = 0.0
    partial_closed = False

    warmup = 35

    for i in range(warmup, n - 1):
        if atr[i] <= 0 or np.isnan(atr[i]):
            continue

        price = closes[i]

        # Check exits if in position
        if in_position:
            if position_side == "LONG":
                # Stop loss
                if lows[i] <= entry_price - params.sl_mult * entry_atr:
                    if partial_closed:
                        pnl = -0.5  # remaining 50% hit SL at breakeven
                    else:
                        pnl = -1.0
                    trades.append(pnl)
                    in_position = False
                    continue

                # TP1
                if not partial_closed and highs[i] >= entry_price + params.tp1_rr * params.sl_mult * entry_atr:
                    partial_closed = True
                    # TP1 profit: 50% * tp1_rr * sl_mult ATR / (sl_mult ATR) = 50% * tp1_rr
                    # We track in R units: TP1 gives 50% * tp1_rr R
                    continue

                # TP2
                if partial_closed and highs[i] >= entry_price + params.tp2_rr * params.sl_mult * entry_atr:
                    # 25% more at tp2_rr R
                    pnl = 0.5 * params.tp1_rr + 0.25 * params.tp2_rr + 0.25 * params.tp3_rr
                    trades.append(pnl)
                    in_position = False
                    partial_closed = False
                    continue

                # Trailing stop after TP1
                if partial_closed:
                    trail_stop = entry_price  # breakeven
                    if lows[i] <= trail_stop:
                        pnl = 0.5 * params.tp1_rr  # TP1 profit, rest at breakeven
                        trades.append(pnl)
                        in_position = False
                        partial_closed = False
                        continue

            else:  # SHORT
                if highs[i] >= entry_price + params.sl_mult * entry_atr:
                    if partial_closed:
                        pnl = -0.5
                    else:
                        pnl = -1.0
                    trades.append(pnl)
                    in_position = False
                    continue

                if not partial_closed and lows[i] <= entry_price - params.tp1_rr * params.sl_mult * entry_atr:
                    partial_closed = True
                    continue

                if partial_closed and lows[i] <= entry_price - params.tp2_rr * params.sl_mult * entry_atr:
                    pnl = 0.5 * params.tp1_rr + 0.25 * params.tp2_rr + 0.25 * params.tp3_rr
                    trades.append(pnl)
                    in_position = False
                    partial_closed = False
                    continue

                if partial_closed:
                    trail_stop = entry_price
                    if highs[i] >= trail_stop:
                        pnl = 0.5 * params.tp1_rr
                        trades.append(pnl)
                        in_position = False
                        partial_closed = False
                        continue

        # Entry signals
        if not in_position and atr[i] > 0:
            score = 0.0

            # Trend
            if ema_fast[i] > ema_slow[i] * 1.001:
                score += 0.15
            elif ema_fast[i] < ema_slow[i] * 0.999:
                score -= 0.15

            # RSI
            if 30 <= rsi[i] <= 45:
                score += 0.20
            elif 55 <= rsi[i] <= 70:
                score -= 0.20

            # Bollinger
            if price <= bb_lower[i]:
                score += 0.15
            elif price >= bb_upper[i]:
                score -= 0.15

            # Volume
            if vol_ma[i] > 0:
                vol_ratio = volumes[i] / vol_ma[i]
                if vol_ratio >= 1.5:
                    score += 0.15
                elif vol_ratio < 0.5:
                    score -= 0.10

            # Momentum
            if i >= 3:
                bullish = sum(1 for j in range(3) if closes[i - j] > data[i - j][0])
                if bullish >= 2:
                    score += 0.10
                elif bullish <= 1:
                    score -= 0.10

            # HTF trend (simulated)
            if ema_fast[i] > ema_slow[i]:
                score += 0.25
            else:
                score -= 0.25

            # Trend regime detection
            regime_bull = ema_fast[i] > ema_slow[i] * 1.005
            regime_bear = ema_fast[i] < ema_slow[i] * 0.995

            # LONG signal (suppress in bear regime)
            if score >= params.min_score and rsi[i] < 55 and not regime_bear:
                in_position = True
                entry_price = price
                position_side = "LONG"
                entry_atr = atr[i]
                partial_closed = False

            # SHORT signal (suppress in bull regime)
            elif score <= -params.min_score and rsi[i] > 45 and not regime_bull:
                in_position = True
                entry_price = price
                position_side = "SHORT"
                entry_atr = atr[i]
                partial_closed = False

    if in_position:
        trades.append(0)

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0)

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    win_rate = wins / len(trades) if trades else 0

    avg_win = np.mean([t for t in trades if t > 0]) if wins > 0 else 0
    avg_loss = abs(np.mean([t for t in trades if t <= 0])) if losses > 0 else 1
    blended_rr = avg_win / avg_loss if avg_loss > 0 else 0

    expectancy = np.mean(trades)
    total_return = sum(trades) / 100 * 100  # in R units, as % of initial
    total_return_pct = (np.prod([1 + t / 10 for t in trades]) - 1) * 100

    # Max drawdown
    equity = [1.0]
    for t in trades:
        equity.append(equity[-1] * (1 + t / 10))
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    return BacktestResult(
        total_trades=len(trades),
        win_rate=win_rate,
        blended_rr=blended_rr,
        expectancy=expectancy,
        total_return=total_return_pct,
        max_drawdown=max_dd * 100,
        profit_factor=profit_factor,
    )


def optimize():
    """Run parameter sweep."""
    print("=" * 70)
    print("  BINANCE QUANT TRADER V3 — PARAMETER OPTIMIZATION")
    print("=" * 70)
    print()

    # Parameter grid
    scores = [0.55, 0.60, 0.65, 0.70]
    sl_mults = [0.8, 1.0, 1.2, 1.5]
    tp1_rrs = [1.0, 1.5, 2.0]
    tp2_rrs = [2.5, 3.0, 4.0]
    tp3_rrs = [5.0, 7.0, 8.0]

    # Generate market data (multiple regimes)
    regimes = {
        "Bull": generate_market_data(500, "bull", 42),
        "Bear": generate_market_data(500, "bear", 43),
        "Mixed": generate_market_data(500, "mixed", 44),
        "Volatile": generate_market_data(500, "volatile", 45),
    }

    total_combos = len(scores) * len(sl_mults) * len(tp1_rrs) * len(tp2_rrs) * len(tp3_rrs)
    print(f"Parameter grid: {total_combos} combinations")
    print(f"Markets: {list(regimes.keys())}")
    print()

    best_params = None
    best_score = -999
    results: List[Tuple[ParamSet, Dict[str, BacktestResult], float]] = []

    count = 0
    start = time.time()

    for score, sl, tp1, tp2, tp3 in itertools.product(
        scores, sl_mults, tp1_rrs, tp2_rrs, tp3_rrs
    ):
        params = ParamSet(
            min_score=score,
            sl_mult=sl,
            tp1_rr=tp1,
            tp2_rr=tp2,
            tp3_rr=tp3,
            trail_mult=2.0,
        )

        regime_results = {}
        total_expectancy = 0
        total_return = 0

        for regime_name, data in regimes.items():
            result = simulate_backtest(data, params)
            regime_results[regime_name] = result
            total_expectancy += result.expectancy
            total_return += result.total_return

        # Composite score: prioritize positive expectancy + high win rate + low drawdown
        avg_expectancy = total_expectancy / len(regimes)
        avg_return = total_return / len(regimes)
        avg_dd = np.mean([r.max_drawdown for r in regime_results.values()])
        avg_wr = np.mean([r.win_rate for r in regime_results.values()])

        composite = (
            avg_expectancy * 10
            + avg_wr * 2
            - avg_dd * 0.5
        )

        results.append((params, regime_results, composite))

        if composite > best_score:
            best_score = composite
            best_params = params

        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {count}/{total_combos} ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"\n  Completed {total_combos} combinations in {elapsed:.1f}s")

    # Sort by composite score
    results.sort(key=lambda x: x[2], reverse=True)

    # Print top 10
    print()
    print("=" * 70)
    print("  TOP 10 PARAMETER SETS")
    print("=" * 70)
    print()
    print(f"{'Rank':>4} {'Score':>6} {'MinScr':>6} {'SL':>4} {'TP1':>4} {'TP2':>4} {'TP3':>4} "
          f"{'AvgWR':>6} {'AvgRR':>6} {'ExpR':>6} {'Return':>8} {'MaxDD':>6}")
    print("-" * 85)

    for rank, (params, regime_results, composite) in enumerate(results[:10], 1):
        avg_wr = np.mean([r.win_rate for r in regime_results.values()]) * 100
        avg_rr = np.mean([r.blended_rr for r in regime_results.values()])
        avg_exp = np.mean([r.expectancy for r in regime_results.values()])
        avg_ret = np.mean([r.total_return for r in regime_results.values()])
        avg_dd = np.mean([r.max_drawdown for r in regime_results.values()])

        print(f"{rank:>4} {composite:>6.2f} {params.min_score:>6.2f} {params.sl_mult:>4.1f} "
              f"{params.tp1_rr:>4.1f} {params.tp2_rr:>4.1f} {params.tp3_rr:>4.1f} "
              f"{avg_wr:>5.1f}% {avg_rr:>5.2f} {avg_exp:>+5.2f}R {avg_ret:>+7.1f}% {avg_dd:>5.1f}%")

    # Detailed report for best params
    best = results[0]
    best_p, best_regimes, best_composite = best

    print()
    print("=" * 70)
    print("  BEST PARAMETER SET — DETAILED REPORT")
    print("=" * 70)
    print()
    print(f"  min_entry_score = {best_p.min_score}")
    print(f"  sl_mult         = {best_p.sl_mult}")
    print(f"  tp1_rr          = {best_p.tp1_rr}")
    print(f"  tp2_rr          = {best_p.tp2_rr}")
    print(f"  tp3_rr          = {best_p.tp3_rr}")
    print()

    for regime_name, result in best_regimes.items():
        print(f"  {regime_name:10s}: WR={result.win_rate*100:5.1f}%  R:R={result.blended_rr:.2f}:1  "
              f"E[R]={result.expectancy:+.2f}  Return={result.total_return:+.1f}%  "
              f"MaxDD={result.max_drawdown:.1f}%  PF={result.profit_factor:.2f}")

    avg_wr = np.mean([r.win_rate for r in best_regimes.values()]) * 100
    avg_rr = np.mean([r.blended_rr for r in best_regimes.values()])
    avg_exp = np.mean([r.expectancy for r in best_regimes.values()])
    avg_ret = np.mean([r.total_return for r in best_regimes.values()])
    avg_dd = np.mean([r.max_drawdown for r in best_regimes.values()])

    print()
    print(f"  AVERAGE: WR={avg_wr:.1f}%  R:R={avg_rr:.2f}:1  E[R]={avg_exp:+.2f}  "
          f"Return={avg_ret:+.1f}%  MaxDD={avg_dd:.1f}%")

    # Also print worst-case params for comparison
    worst = results[-1]
    worst_p = worst[0]
    print()
    print("=" * 70)
    print("  COMPARISON: Original vs Optimized")
    print("=" * 70)
    print()
    print(f"  {'Parameter':<20} {'Original':>10} {'Optimized':>10}")
    print(f"  {'-'*40}")
    print(f"  {'min_entry_score':<20} {'0.55':>10} {best_p.min_score:>10.2f}")
    print(f"  {'sl_mult':<20} {'1.0':>10} {best_p.sl_mult:>10.1f}")
    print(f"  {'tp1_rr':<20} {'2.0':>10} {best_p.tp1_rr:>10.1f}")
    print(f"  {'tp2_rr':<20} {'4.0':>10} {best_p.tp2_rr:>10.1f}")
    print(f"  {'tp3_rr':<20} {'8.0':>10} {best_p.tp3_rr:>10.1f}")

    # Save best params
    print()
    print("=" * 70)
    print("  RECOMMENDED config.py CHANGES")
    print("=" * 70)
    print()
    print(f"    min_entry_score = {best_p.min_score}")
    print(f"    sl_mult = {best_p.sl_mult}")
    print(f"    tp1_rr = {best_p.tp1_rr}")
    print(f"    tp2_rr = {best_p.tp2_rr}")
    print(f"    tp3_rr = {best_p.tp3_rr}")
    print()


if __name__ == "__main__":
    optimize()
