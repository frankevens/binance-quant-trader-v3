"""
Binance Quant Trader V2 - Backtest Engine
==========================================
Backtest the ATR strategy against historical Binance kline data.
Calculates win rate, profit factor, max drawdown, and Sharpe ratio.

Usage:
    python backtest.py                    # All symbols, default 30 days
    python backtest.py --symbol BTCUSDT   # Single symbol
    python backtest.py --days 90          # Last 90 days
    python backtest.py --sl 1.5 --tp 3.0  # Custom SL/TP multipliers
"""

import asyncio
import argparse
import time
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from binance import AsyncClient
from config import Config


@dataclass
class Trade:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: int
    exit_price: float = 0.0
    exit_time: int = 0
    sl_price: float = 0.0
    tp_price: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0


@dataclass
class BacktestResult:
    symbol: str
    trades: list = field(default_factory=list)
    initial_balance: float = 10000.0
    final_balance: float = 10000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct > 0)

    @property
    def losers(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct <= 0)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades * 100 if self.total_trades > 0 else 0

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [abs(t.pnl_pct) for t in self.trades if t.pnl_pct <= 0]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        """Profit Factor = Gross Profit / Gross Loss"""
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = sum(abs(t.pnl_pct) for t in self.trades if t.pnl_pct <= 0)
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def reward_risk_ratio(self) -> float:
        """Average R:R realized"""
        if self.avg_loss > 0:
            return self.avg_win / self.avg_loss
        return float('inf')

    @property
    def total_return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades:
            return 0.0
        balance = self.initial_balance
        peak = balance
        max_dd = 0.0
        for t in self.trades:
            balance *= (1 + t.pnl_pct / 100)
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl_pct for t in self.trades])
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r == 0:
            return 0.0
        return float(mean_r / std_r * np.sqrt(len(returns)))

    @property
    def exit_reasons(self) -> dict:
        reasons = {}
        for t in self.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        return reasons


def calculate_atr(klines: list, period: int = 14) -> list:
    """Calculate ATR for all klines, return list of ATR values."""
    atrs = [None] * period
    for i in range(period, len(klines)):
        highs = [float(klines[j][2]) for j in range(i - period, i + 1)]
        lows = [float(klines[j][3]) for j in range(i - period, i + 1)]
        closes = [float(klines[j][4]) for j in range(i - period, i + 1)]

        trs = []
        for k in range(1, len(highs)):
            tr = max(highs[k] - lows[k], abs(highs[k] - closes[k-1]), abs(lows[k] - closes[k-1]))
            trs.append(tr)
        atrs.append(np.mean(trs))
    return atrs


def calculate_rsi(closes: list, period: int = 14) -> list:
    """Calculate RSI series."""
    rsis = [None] * period
    for i in range(period, len(closes)):
        subset = closes[i - period:i + 1]
        deltas = [subset[j+1] - subset[j] for j in range(len(subset)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rsis.append(float(100.0 - (100.0 / (1.0 + avg_gain / avg_loss))))
    return rsis


def calculate_ema(closes: list, period: int) -> list:
    """Calculate EMA series."""
    emas = [None] * (period - 1)
    multiplier = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    emas.append(ema)
    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * multiplier + ema
        emas.append(ema)
    return emas


def calculate_bollinger(closes: list, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands."""
    uppers = [None] * (period - 1)
    mids = [None] * (period - 1)
    lowers = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        subset = closes[i - period + 1:i + 1]
        sma = np.mean(subset)
        std = np.std(subset)
        mids.append(sma)
        uppers.append(sma + std_dev * std)
        lowers.append(sma - std_dev * std)
    return uppers, mids, lowers


def calculate_volume_ratio(volumes: list, period: int = 20) -> list:
    """Volume ratio series."""
    ratios = [None] * period
    for i in range(period, len(volumes)):
        avg = np.mean(volumes[i - period:i])
        ratios.append(volumes[i] / avg if avg > 0 else 1.0)
    return ratios


async def backtest_symbol(client, symbol: str, days: int, sl_mult: float, tp_mult: float,
                           trail_mult: float, min_score: float) -> BacktestResult:
    """Run backtest for a single symbol."""
    print(f"\nFetching {symbol} data ({days} days)...")

    # Fetch 15m klines
    end_time = int(time.time() * 1000)
    start_time = int((time.time() - days * 86400) * 1000)

    all_klines = []
    current_start = start_time
    while current_start < end_time:
        klines = await client.futures_klines(
            symbol=symbol, interval="15m",
            startTime=current_start, endTime=end_time, limit=1500
        )
        if not klines:
            break
        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        await asyncio.sleep(0.1)

    if len(all_klines) < 100:
        print(f"  Insufficient data: {len(all_klines)} klines")
        return BacktestResult(symbol=symbol)

    print(f"  Loaded {len(all_klines)} klines")

    # Parse data
    closes = [float(k[4]) for k in all_klines]
    highs = [float(k[2]) for k in all_klines]
    lows = [float(k[3]) for k in all_klines]
    volumes = [float(k[5]) for k in all_klines]

    # Calculate indicators
    atrs = calculate_atr(all_klines, 14)
    rsis = calculate_rsi(closes, 14)
    ema10s = calculate_ema(closes, 10)
    ema30s = calculate_ema(closes, 30)
    bb_uppers, bb_mids, bb_lowers = calculate_bollinger(closes, 20)
    vol_ratios = calculate_volume_ratio(volumes, 20)

    # Run strategy
    result = BacktestResult(symbol=symbol)
    in_position = False
    current_trade = None
    COMMISSION_PCT = 0.05  # 0.05% taker fee (round trip = 0.1%)

    start_idx = 35  # Need enough history for indicators

    for i in range(start_idx, len(all_klines)):
        price = closes[i]
        atr = atrs[i]
        rsi = rsis[i]
        ema10 = ema10s[i]
        ema30 = ema30s[i]
        bb_upper = bb_uppers[i]
        bb_lower = bb_lowers[i]
        vol_ratio = vol_ratios[i]

        if any(v is None for v in [atr, rsi, ema10, ema30, bb_upper, bb_lower, vol_ratio]):
            continue

        # === Check exit if in position ===
        if in_position and current_trade:
            current_trade.bars_held += 1

            if current_trade.side == "LONG":
                # Stop loss
                if lows[i] <= current_trade.sl_price:
                    exit_price = current_trade.sl_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((exit_price - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "stop_loss"
                    result.trades.append(current_trade)
                    in_position = False
                    continue
                # Take profit
                if highs[i] >= current_trade.tp_price:
                    exit_price = current_trade.tp_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((exit_price - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "take_profit"
                    result.trades.append(current_trade)
                    in_position = False
                    continue
                # Trailing stop
                trail_price = price - trail_mult * atr
                if trail_price > current_trade.sl_price:
                    current_trade.sl_price = trail_price
                if lows[i] <= current_trade.sl_price and current_trade.bars_held > 3:
                    exit_price = current_trade.sl_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((exit_price - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "trailing_stop"
                    result.trades.append(current_trade)
                    in_position = False
                    continue

            elif current_trade.side == "SHORT":
                if highs[i] >= current_trade.sl_price:
                    exit_price = current_trade.sl_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((current_trade.entry_price - exit_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "stop_loss"
                    result.trades.append(current_trade)
                    in_position = False
                    continue
                if lows[i] <= current_trade.tp_price:
                    exit_price = current_trade.tp_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((current_trade.entry_price - exit_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "take_profit"
                    result.trades.append(current_trade)
                    in_position = False
                    continue
                trail_price = price + trail_mult * atr
                if trail_price < current_trade.sl_price:
                    current_trade.sl_price = trail_price
                if highs[i] >= current_trade.sl_price and current_trade.bars_held > 3:
                    exit_price = current_trade.sl_price
                    current_trade.exit_price = exit_price
                    current_trade.exit_time = int(all_klines[i][0])
                    current_trade.pnl_pct = ((current_trade.entry_price - exit_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                    current_trade.exit_reason = "trailing_stop"
                    result.trades.append(current_trade)
                    in_position = False
                    continue

        if in_position:
            continue

        # === Entry scoring (same as live strategy) ===
        long_score = 0.0
        short_score = 0.0

        # Trend (EMA)
        if price > ema30:
            long_score += 0.15
        else:
            short_score += 0.15
        if ema10 > ema30:
            long_score += 0.15
        elif ema10 < ema30:
            short_score += 0.15

        # RSI
        if 30 <= rsi <= 45:
            long_score += 0.20
        elif 45 < rsi <= 55:
            long_score += 0.10
            short_score += 0.10
        elif 55 <= rsi <= 70:
            short_score += 0.20
        elif rsi > 70:
            long_score -= 0.10
            short_score += 0.05
        elif rsi < 30:
            short_score -= 0.10
            long_score += 0.05

        # Bollinger
        if price <= bb_lower:
            long_score += 0.15
        elif price <= (bb_lower + bb_mids[i]) / 2 if bb_mids[i] else 0:
            long_score += 0.08
        if price >= bb_upper:
            short_score += 0.15
        elif bb_mids[i] and price >= (bb_upper + bb_mids[i]) / 2:
            short_score += 0.08

        # Volume
        if vol_ratio >= 1.5:
            long_score += 0.15
            short_score += 0.15
        elif vol_ratio >= 1.0:
            long_score += 0.08
            short_score += 0.08
        elif vol_ratio < 0.5:
            long_score -= 0.05
            short_score -= 0.05

        # Momentum
        if i >= 2:
            if closes[i] > closes[i-1] > closes[i-2]:
                long_score += 0.10
            elif closes[i] < closes[i-1] < closes[i-2]:
                short_score += 0.10
            elif closes[i] > closes[i-1]:
                long_score += 0.05
            elif closes[i] < closes[i-1]:
                short_score += 0.05

        # Entry decision
        if long_score >= min_score and long_score > short_score:
            current_trade = Trade(
                symbol=symbol, side="LONG",
                entry_price=price, entry_time=int(all_klines[i][0]),
                sl_price=price - sl_mult * atr,
                tp_price=price + tp_mult * atr,
            )
            in_position = True

        elif short_score >= min_score and short_score > long_score:
            current_trade = Trade(
                symbol=symbol, side="SHORT",
                entry_price=price, entry_time=int(all_klines[i][0]),
                sl_price=price + sl_mult * atr,
                tp_price=price - tp_mult * atr,
            )
            in_position = True

    # Close any open position at last price
    if in_position and current_trade:
        current_trade.exit_price = closes[-1]
        current_trade.exit_time = int(all_klines[-1][0])
        if current_trade.side == "LONG":
            current_trade.pnl_pct = ((closes[-1] - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
        else:
            current_trade.pnl_pct = ((current_trade.entry_price - closes[-1]) / current_trade.entry_price * 100) - COMMISSION_PCT
        current_trade.exit_reason = "end_of_data"
        result.trades.append(current_trade)

    # Calculate final balance
    balance = result.initial_balance
    for t in result.trades:
        balance *= (1 + t.pnl_pct / 100)
    result.final_balance = balance

    return result


def print_result(r: BacktestResult):
    """Print backtest results for a symbol."""
    print(f"\n{'='*60}")
    print(f"  {r.symbol} - BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total Trades:      {r.total_trades}")
    print(f"  Winners:           {r.winners}")
    print(f"  Losers:            {r.losers}")
    print(f"  Win Rate:          {r.win_rate:.1f}%")
    print(f"  Avg Win:           {r.avg_win:+.3f}%")
    print(f"  Avg Loss:          -{r.avg_loss:.3f}%")
    print(f"  Reward:Risk:       {r.reward_risk_ratio:.2f} : 1")
    print(f"  Profit Factor:     {r.profit_factor:.2f}")
    print(f"  Total Return:      {r.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:      {r.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:      {r.sharpe_ratio:.2f}")
    print(f"  Balance:           {r.initial_balance:.0f} → {r.final_balance:.0f} USDT")
    print(f"  Exit Reasons:      {r.exit_reasons}")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="ATR Strategy Backtest")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (default: all)")
    parser.add_argument("--days", type=int, default=30, help="Backtest period in days")
    parser.add_argument("--sl", type=float, default=1.5, help="Stop loss ATR multiplier")
    parser.add_argument("--tp", type=float, default=3.0, help="Take profit ATR multiplier")
    parser.add_argument("--trail", type=float, default=2.0, help="Trailing stop ATR multiplier")
    parser.add_argument("--score", type=float, default=0.65, help="Minimum entry score")
    args = parser.parse_args()

    config = Config()
    symbols = [args.symbol] if args.symbol else config.symbols

    print(f"Binance Quant Trader V2 - Backtest Engine")
    print(f"Period: {args.days} days | SL={args.sl}x ATR | TP={args.tp}x ATR | Trail={args.trail}x ATR")
    print(f"Min Score: {args.score} | Symbols: {', '.join(symbols)}")
    print(f"R:R Ratio: {args.tp/args.sl:.2f}:1 | Break-even Win Rate: {1/(1+args.tp/args.sl)*100:.1f}%")

    client = await AsyncClient.create()

    all_results = []
    for symbol in symbols:
        try:
            result = await backtest_symbol(
                client, symbol, args.days,
                args.sl, args.tp, args.trail, args.score
            )
            all_results.append(result)
            print_result(result)
        except Exception as e:
            print(f"\n  {symbol}: Error - {e}")

    await client.close_connection()

    # Aggregate results
    if all_results:
        total_trades = sum(r.total_trades for r in all_results)
        total_wins = sum(r.winners for r in all_results)
        all_pnls = []
        for r in all_results:
            all_pnls.extend([t.pnl_pct for t in r.trades])

        if total_trades > 0:
            print(f"\n{'='*60}")
            print(f"  AGGREGATE RESULTS (All Symbols)")
            print(f"{'='*60}")
            print(f"  Total Trades:      {total_trades}")
            print(f"  Win Rate:          {total_wins/total_trades*100:.1f}%")
            avg_win = np.mean([p for p in all_pnls if p > 0]) if any(p > 0 for p in all_pnls) else 0
            avg_loss = np.mean([abs(p) for p in all_pnls if p <= 0]) if any(p <= 0 for p in all_pnls) else 0
            print(f"  Avg Win:           {avg_win:+.3f}%")
            print(f"  Avg Loss:          -{avg_loss:.3f}%")
            if avg_loss > 0:
                print(f"  Reward:Risk:       {avg_win/avg_loss:.2f} : 1")
            gross_profit = sum(p for p in all_pnls if p > 0)
            gross_loss = sum(abs(p) for p in all_pnls if p <= 0)
            if gross_loss > 0:
                print(f"  Profit Factor:     {gross_profit/gross_loss:.2f}")
            print(f"  Total Return:      {sum(all_pnls):+.2f}%")
            if len(all_pnls) > 1:
                print(f"  Sharpe Ratio:      {np.mean(all_pnls)/np.std(all_pnls)*np.sqrt(len(all_pnls)):.2f}")
            print(f"{'='*60}")

            # Expectancy per trade
            wr = total_wins / total_trades
            expectancy = wr * avg_win - (1 - wr) * avg_loss
            print(f"\n  Expectancy/Trade:  {expectancy:+.3f}%")
            if expectancy > 0:
                print(f"  → Strategy is PROFITABLE (positive expectancy)")
            else:
                print(f"  → Strategy is UNPROFITABLE (negative expectancy)")


if __name__ == "__main__":
    asyncio.run(main())
