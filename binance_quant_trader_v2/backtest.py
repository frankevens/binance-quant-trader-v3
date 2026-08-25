"""
Binance Quant Trader V2 - Backtest Engine (V3)
================================================
Backtest with partial take profit simulation.
Measures: win rate, profit factor, max drawdown, Sharpe, blended R:R.

Usage:
    python backtest.py                          # All symbols, 30 days
    python backtest.py --symbol BTCUSDT --days 90
    python backtest.py --sl 1.0 --tp1 2 --tp2 4 --tp3 8
"""

import asyncio
import argparse
import time
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field

from binance import AsyncClient
from config import Config


@dataclass
class PartialTrade:
    """A single partial close within a position."""
    pct: float        # % of original position closed
    exit_price: float
    pnl_pct: float    # PnL % on this portion
    exit_reason: str
    bar_idx: int


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    entry_time: int
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    partials: list = field(default_factory=list)
    strategy_type: str = ""

    @property
    def is_winner(self) -> bool:
        """A trade wins if total PnL > 0."""
        return self.total_pnl_pct > 0

    @property
    def total_pnl_pct(self) -> float:
        """Weighted PnL across all partial closes."""
        if not self.partials:
            return 0.0
        return sum(p.pnl_pct * p.pct for p in self.partials)

    @property
    def total_r_multiple(self) -> float:
        """Total R-multiple (PnL / initial risk)."""
        risk = abs(self.entry_price - self.sl_price)
        if risk == 0:
            return 0.0
        if self.side == "LONG":
            total_gain = sum((p.exit_price - self.entry_price) * p.pct for p in self.partials)
        else:
            total_gain = sum((self.entry_price - p.exit_price) * p.pct for p in self.partials)
        return total_gain / risk

    @property
    def exit_reason_summary(self) -> str:
        if not self.partials:
            return "none"
        return "+".join(set(p.exit_reason for p in self.partials))


@dataclass
class BacktestResult:
    symbol: str
    trades: list = field(default_factory=list)
    initial_balance: float = 10000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def losers(self) -> int:
        return sum(1 for t in self.trades if not t.is_winner)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades * 100 if self.total_trades > 0 else 0

    @property
    def avg_r_multiple(self) -> float:
        rs = [t.total_r_multiple for t in self.trades]
        return np.mean(rs) if rs else 0

    @property
    def avg_win_r(self) -> float:
        wins = [t.total_r_multiple for t in self.trades if t.is_winner]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss_r(self) -> float:
        losses = [abs(t.total_r_multiple) for t in self.trades if not t.is_winner]
        return np.mean(losses) if losses else 0

    @property
    def blended_rr(self) -> float:
        if self.avg_loss_r > 0:
            return self.avg_win_r / self.avg_loss_r
        return float('inf') if self.avg_win_r > 0 else 0

    @property
    def profit_factor(self) -> float:
        gross_w = sum(t.total_r_multiple for t in self.trades if t.is_winner)
        gross_l = sum(abs(t.total_r_multiple) for t in self.trades if not t.is_winner)
        return gross_w / gross_l if gross_l > 0 else float('inf')

    @property
    def final_balance(self) -> float:
        bal = self.initial_balance
        for t in self.trades:
            # Each trade risks a fixed % of balance
            risk_pct = 2.0  # Risk 2% per trade
            pnl = t.total_r_multiple * risk_pct
            bal *= (1 + pnl / 100)
        return bal

    @property
    def total_return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    @property
    def max_drawdown_pct(self) -> float:
        bal = self.initial_balance
        peak = bal
        max_dd = 0.0
        for t in self.trades:
            risk_pct = 2.0
            pnl = t.total_r_multiple * risk_pct
            bal *= (1 + pnl / 100)
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.total_r_multiple for t in self.trades])
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(len(returns)))

    @property
    def tp1_hit_rate(self) -> float:
        """% of trades that hit TP1 (the key win rate driver)."""
        if not self.trades:
            return 0.0
        tp1_hits = sum(1 for t in self.trades if any(p.exit_reason == "tp1" for p in t.partials))
        return tp1_hits / len(self.trades) * 100

    @property
    def expectancy_per_trade(self) -> float:
        """Expected R per trade."""
        return np.mean([t.total_r_multiple for t in self.trades]) if self.trades else 0


def calculate_atr_series(klines: list, period: int = 14) -> list:
    atrs = [None] * period
    for i in range(period, len(klines)):
        trs = []
        for j in range(i - period + 1, i + 1):
            h, l, pc = float(klines[j][2]), float(klines[j][3]), float(klines[j-1][4])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atrs.append(np.mean(trs))
    return atrs


def calculate_rsi_series(closes: list, period: int = 14) -> list:
    rsis = [None] * period
    for i in range(period, len(closes)):
        deltas = [closes[i - period + j + 1] - closes[i - period + j] for j in range(period)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        ag, al = np.mean(gains), np.mean(losses)
        rsis.append(100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))
    return rsis


def calculate_ema_series(closes: list, period: int) -> list:
    emas = [None] * (period - 1)
    mult = 2.0 / (period + 1)
    ema = np.mean(closes[:period])
    emas.append(ema)
    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * mult + ema
        emas.append(ema)
    return emas


def calculate_bollinger_series(closes: list, period: int = 20, std_dev: float = 2.0):
    uppers, mids, lowers = [None]*(period-1), [None]*(period-1), [None]*(period-1)
    for i in range(period - 1, len(closes)):
        sub = closes[i - period + 1:i + 1]
        sma, std = np.mean(sub), np.std(sub)
        mids.append(sma)
        uppers.append(sma + std_dev * std)
        lowers.append(sma - std_dev * std)
    return uppers, mids, lowers


def calculate_volume_ratio_series(volumes: list, period: int = 20) -> list:
    ratios = [None] * period
    for i in range(period, len(volumes)):
        avg = np.mean(volumes[i - period:i])
        ratios.append(volumes[i] / avg if avg > 0 else 1.0)
    return ratios


async def backtest_symbol(client, symbol, days, sl_mult, tp1_rr, tp2_rr, tp3_rr,
                           trail_mult, min_score, tp1_pct, tp2_pct):
    print(f"\nFetching {symbol} ({days}d)...")

    end_time = int(time.time() * 1000)
    start_time = int((time.time() - days * 86400) * 1000)

    all_klines = []
    cur = start_time
    while cur < end_time:
        kl = await client.futures_klines(symbol=symbol, interval="15m",
                                          startTime=cur, endTime=end_time, limit=1500)
        if not kl:
            break
        all_klines.extend(kl)
        cur = kl[-1][0] + 1
        await asyncio.sleep(0.1)

    if len(all_klines) < 100:
        print(f"  Insufficient: {len(all_klines)} klines")
        return BacktestResult(symbol=symbol)

    print(f"  {len(all_klines)} klines loaded")

    closes = [float(k[4]) for k in all_klines]
    highs = [float(k[2]) for k in all_klines]
    lows = [float(k[3]) for k in all_klines]
    volumes = [float(k[5]) for k in all_klines]

    atrs = calculate_atr_series(all_klines, 14)
    rsis = calculate_rsi_series(closes, 14)
    ema10s = calculate_ema_series(closes, 10)
    ema30s = calculate_ema_series(closes, 30)
    bb_u, bb_m, bb_l = calculate_bollinger_series(closes, 20)
    vol_ratios = calculate_volume_ratio_series(volumes, 20)

    result = BacktestResult(symbol=symbol)
    in_position = False
    current_trade = None
    COMMISSION_PCT = 0.05

    for i in range(35, len(all_klines)):
        price = closes[i]
        atr = atrs[i]
        rsi = rsis[i]
        ema10 = ema10s[i]
        ema30 = ema30s[i]
        bb_upper = bb_u[i]
        bb_lower = bb_l[i]
        vol_ratio = vol_ratios[i]

        if any(v is None for v in [atr, rsi, ema10, ema30, bb_upper, bb_lower, vol_ratio]):
            continue

        sl_dist = sl_mult * atr

        # === Position Management with Partial TP ===
        if in_position and current_trade:
            if current_trade.side == "LONG":
                # Check TP1
                if not any(p.exit_reason == "tp1" for p in current_trade.partials):
                    if highs[i] >= current_trade.tp1_price:
                        pnl = ((current_trade.tp1_price - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                        current_trade.partials.append(PartialTrade(
                            pct=tp1_pct, exit_price=current_trade.tp1_price,
                            pnl_pct=pnl, exit_reason="tp1", bar_idx=i))
                        # Move SL to breakeven
                        current_trade.sl_price = current_trade.entry_price
                        continue

                # Check TP2
                if any(p.exit_reason == "tp1" for p in current_trade.partials) and \
                   not any(p.exit_reason == "tp2" for p in current_trade.partials):
                    if highs[i] >= current_trade.tp2_price:
                        pnl = ((current_trade.tp2_price - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                        current_trade.partials.append(PartialTrade(
                            pct=tp2_pct, exit_price=current_trade.tp2_price,
                            pnl_pct=pnl, exit_reason="tp2", bar_idx=i))
                        continue

                # Update trailing stop (only after TP1)
                if any(p.exit_reason == "tp1" for p in current_trade.partials):
                    new_trail = price - trail_mult * atr
                    if new_trail > current_trade.sl_price:
                        current_trade.sl_price = new_trail

                # Stop loss / trailing stop
                if lows[i] <= current_trade.sl_price:
                    remaining_pct = 1.0 - sum(p.pct for p in current_trade.partials)
                    if remaining_pct > 0.01:
                        exit_p = current_trade.sl_price
                        pnl = ((exit_p - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                        reason = "sl_be" if abs(exit_p - current_trade.entry_price) < atr * 0.1 else "trailing"
                        current_trade.partials.append(PartialTrade(
                            pct=remaining_pct, exit_price=exit_p,
                            pnl_pct=pnl, exit_reason=reason, bar_idx=i))
                    result.trades.append(current_trade)
                    in_position = False
                    continue

            elif current_trade.side == "SHORT":
                if not any(p.exit_reason == "tp1" for p in current_trade.partials):
                    if lows[i] <= current_trade.tp1_price:
                        pnl = ((current_trade.entry_price - current_trade.tp1_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                        current_trade.partials.append(PartialTrade(
                            pct=tp1_pct, exit_price=current_trade.tp1_price,
                            pnl_pct=pnl, exit_reason="tp1", bar_idx=i))
                        current_trade.sl_price = current_trade.entry_price
                        continue

                if any(p.exit_reason == "tp1" for p in current_trade.partials) and \
                   not any(p.exit_reason == "tp2" for p in current_trade.partials):
                    if lows[i] <= current_trade.tp2_price:
                        pnl = ((current_trade.entry_price - current_trade.tp2_price) / current_trade.entry_price * 100) - COMMISSION_PCT
                        current_trade.partials.append(PartialTrade(
                            pct=tp2_pct, exit_price=current_trade.tp2_price,
                            pnl_pct=pnl, exit_reason="tp2", bar_idx=i))
                        continue

                if any(p.exit_reason == "tp1" for p in current_trade.partials):
                    new_trail = price + trail_mult * atr
                    if new_trail < current_trade.sl_price:
                        current_trade.sl_price = new_trail

                if highs[i] >= current_trade.sl_price:
                    remaining_pct = 1.0 - sum(p.pct for p in current_trade.partials)
                    if remaining_pct > 0.01:
                        exit_p = current_trade.sl_price
                        pnl = ((current_trade.entry_price - exit_p) / current_trade.entry_price * 100) - COMMISSION_PCT
                        reason = "sl_be" if abs(exit_p - current_trade.entry_price) < atr * 0.1 else "trailing"
                        current_trade.partials.append(PartialTrade(
                            pct=remaining_pct, exit_price=exit_p,
                            pnl_pct=pnl, exit_reason=reason, bar_idx=i))
                    result.trades.append(current_trade)
                    in_position = False
                    continue

        if in_position:
            continue

        # === Entry Scoring (V3 Multi-Strategy) ===
        long_score = 0.0
        short_score = 0.0

        # Trend
        if ema10 > ema30 * 1.001:
            long_score += 0.20
        elif ema10 < ema30 * 0.999:
            short_score += 0.20
        if price > ema30:
            long_score += 0.10
        else:
            short_score += 0.10

        # RSI
        if 30 <= rsi <= 45:
            long_score += 0.25
        elif 45 < rsi <= 55:
            long_score += 0.10
            short_score += 0.10
        elif 55 <= rsi <= 70:
            short_score += 0.25
        elif rsi > 70:
            short_score += 0.05
        elif rsi < 30:
            long_score += 0.05

        # Bollinger
        if price <= bb_lower:
            long_score += 0.15
        if price >= bb_upper:
            short_score += 0.15

        # Volume
        if vol_ratio >= 1.3:
            long_score += 0.15
            short_score += 0.15
        elif vol_ratio >= 1.0:
            long_score += 0.08
            short_score += 0.08

        # Momentum
        if i >= 2:
            if closes[i] > closes[i-1] > closes[i-2]:
                long_score += 0.10
            elif closes[i] < closes[i-1] < closes[i-2]:
                short_score += 0.10

        # Entry
        if long_score >= min_score and long_score > short_score:
            current_trade = Trade(
                symbol=symbol, side="LONG", entry_price=price,
                entry_time=int(all_klines[i][0]),
                sl_price=price - sl_dist,
                tp1_price=price + tp1_rr * sl_dist,
                tp2_price=price + tp2_rr * sl_dist,
                tp3_price=price + tp3_rr * sl_dist,
                strategy_type="fusion",
            )
            in_position = True
        elif short_score >= min_score and short_score > long_score:
            current_trade = Trade(
                symbol=symbol, side="SHORT", entry_price=price,
                entry_time=int(all_klines[i][0]),
                sl_price=price + sl_dist,
                tp1_price=price - tp1_rr * sl_dist,
                tp2_price=price - tp2_rr * sl_dist,
                tp3_price=price - tp3_rr * sl_dist,
                strategy_type="fusion",
            )
            in_position = True

    # Close open position
    if in_position and current_trade:
        remaining_pct = 1.0 - sum(p.pct for p in current_trade.partials)
        if remaining_pct > 0.01:
            if current_trade.side == "LONG":
                pnl = ((closes[-1] - current_trade.entry_price) / current_trade.entry_price * 100) - COMMISSION_PCT
            else:
                pnl = ((current_trade.entry_price - closes[-1]) / current_trade.entry_price * 100) - COMMISSION_PCT
            current_trade.partials.append(PartialTrade(
                pct=remaining_pct, exit_price=closes[-1],
                pnl_pct=pnl, exit_reason="end", bar_idx=len(all_klines)-1))
        result.trades.append(current_trade)

    return result


def print_result(r: BacktestResult):
    print(f"\n{'='*65}")
    print(f"  {r.symbol} — V3 BACKTEST (Partial TP)")
    print(f"{'='*65}")
    print(f"  Total Trades:      {r.total_trades}")
    print(f"  Winners:           {r.winners}")
    print(f"  Losers:            {r.losers}")
    print(f"  Win Rate:          {r.win_rate:.1f}%")
    print(f"  TP1 Hit Rate:      {r.tp1_hit_rate:.1f}%")
    print(f"  Avg Win (R):       {r.avg_win_r:+.2f}R")
    print(f"  Avg Loss (R):      -{r.avg_loss_r:.2f}R")
    print(f"  Blended R:R:       {r.blended_rr:.2f} : 1")
    print(f"  Avg R/Trade:       {r.expectancy_per_trade:+.2f}R")
    print(f"  Profit Factor:     {r.profit_factor:.2f}")
    print(f"  Total Return:      {r.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:      {r.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:      {r.sharpe_ratio:.2f}")
    print(f"  Balance:           {r.initial_balance:.0f} → {r.final_balance:.0f} USDT")
    print(f"{'='*65}")


async def main():
    parser = argparse.ArgumentParser(description="ATR V3 Backtest")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--sl", type=float, default=1.0, help="SL ATR multiplier")
    parser.add_argument("--tp1", type=float, default=2.0, help="TP1 R-multiple")
    parser.add_argument("--tp2", type=float, default=4.0, help="TP2 R-multiple")
    parser.add_argument("--tp3", type=float, default=8.0, help="TP3 R-multiple")
    parser.add_argument("--trail", type=float, default=1.5, help="Trailing ATR mult")
    parser.add_argument("--score", type=float, default=0.55, help="Min entry score")
    parser.add_argument("--tp1-pct", type=float, default=0.50, help="TP1 close %%")
    parser.add_argument("--tp2-pct", type=float, default=0.25, help="TP2 close %%")
    args = parser.parse_args()

    config = Config()
    symbols = [args.symbol] if args.symbol else config.symbols

    print("Binance Quant Trader V3 — Backtest Engine")
    print(f"Period: {args.days}d | SL={args.sl}x | TP1={args.tp1}R({args.tp1_pct*100:.0f}%) "
          f"TP2={args.tp2}R({args.tp2_pct*100:.0f}%) TP3={args.tp3}R(trail)")
    print(f"Trail={args.trail}x | MinScore={args.score} | Symbols: {', '.join(symbols)}")

    client = await AsyncClient.create()

    all_results = []
    for symbol in symbols:
        try:
            r = await backtest_symbol(client, symbol, args.days, args.sl,
                                       args.tp1, args.tp2, args.tp3, args.trail,
                                       args.score, args.tp1_pct, args.tp2_pct)
            all_results.append(r)
            print_result(r)
        except Exception as e:
            print(f"\n  {symbol}: Error - {e}")

    await client.close_connection()

    # Aggregate
    if all_results:
        total_trades = sum(r.total_trades for r in all_results)
        total_wins = sum(r.winners for r in all_results)
        all_r = [t.total_r_multiple for r in all_results for t in r.trades]

        if total_trades > 0:
            print(f"\n{'='*65}")
            print(f"  AGGREGATE (All Symbols)")
            print(f"{'='*65}")
            print(f"  Total Trades:      {total_trades}")
            print(f"  Win Rate:          {total_wins/total_trades*100:.1f}%")
            avg_r = np.mean(all_r) if all_r else 0
            avg_win = np.mean([r for r in all_r if r > 0]) if any(r > 0 for r in all_r) else 0
            avg_loss = np.mean([abs(r) for r in all_r if r <= 0]) if any(r <= 0 for r in all_r) else 0
            print(f"  Avg Win:           {avg_win:+.2f}R")
            print(f"  Avg Loss:          -{avg_loss:.2f}R")
            if avg_loss > 0:
                print(f"  Blended R:R:       {avg_win/avg_loss:.2f} : 1")
            print(f"  Avg R/Trade:       {avg_r:+.2f}R")
            gw = sum(r for r in all_r if r > 0)
            gl = sum(abs(r) for r in all_r if r <= 0)
            if gl > 0:
                print(f"  Profit Factor:     {gw/gl:.2f}")
            total_ret = sum(all_r) * 2.0 / total_trades * 100  # Approx with 2% risk
            print(f"  Approx Return:     {total_ret:+.1f}%")
            if len(all_r) > 1:
                print(f"  Sharpe:            {np.mean(all_r)/np.std(all_r)*np.sqrt(len(all_r)):.2f}")

            tp1_total = sum(1 for r in all_results for t in r.trades if any(p.exit_reason == "tp1" for p in t.partials))
            print(f"  TP1 Hit Rate:      {tp1_total/total_trades*100:.1f}%")
            print(f"{'='*65}")

            if avg_r > 0:
                print(f"\n  ✓ Strategy is PROFITABLE (expectancy = {avg_r:+.2f}R/trade)")
            else:
                print(f"\n  ✗ Strategy is UNPROFITABLE (expectancy = {avg_r:+.2f}R/trade)")
                print(f"  Consider adjusting parameters or market conditions.")


if __name__ == "__main__":
    asyncio.run(main())
