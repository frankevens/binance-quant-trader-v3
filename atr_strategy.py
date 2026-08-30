"""
Binance Quant Trader V3 - ATR Strategy (V3 Ultimate)
=====================================================
Multi-strategy fusion with partial take profit system.

Architecture:
  1. Trend Following (EMA + HTF alignment) — catches big moves
  2. Mean Reversion (RSI + Bollinger) — catches overextensions
  3. Breakout (Volume + ATR expansion) — catches momentum shifts
  4. Partial TP: 50% at 2R, 25% at 4R, 25% trailing to 8-10R

This maximizes win rate (via partial TP1) while keeping upside (via trailing).
Realistic targets: 55-65% win rate, 3-4:1 blended R:R.
"""

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("trader.atr")


@dataclass
class PartialTP:
    """Partial take profit levels."""
    tp1_pct: float = 0.50       # Close 50% at TP1
    tp2_pct: float = 0.25       # Close 25% at TP2
    tp3_pct: float = 0.25       # Remaining 25% trails to TP3
    tp1_rr: float = 2.0         # TP1 at 2R (locks in win)
    tp2_rr: float = 4.0         # TP2 at 4R
    tp3_rr: float = 8.0         # TP3 target at 8R (trailing)
    move_sl_to_be: bool = True  # Move SL to breakeven after TP1


@dataclass
class ATRSignal:
    symbol: str
    signal: str  # "LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT", "PARTIAL_CLOSE", "HOLD"
    entry_price: float
    atr_value: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    confidence: float
    reason: str
    # V3: partial TP levels
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    tp1_close_pct: float = 0.0
    tp2_close_pct: float = 0.0
    strategy_type: str = ""  # "trend", "mean_reversion", "breakout"


class ATRCalculator:
    """
    V3 Multi-strategy fusion with partial take profit.
    """

    def __init__(self, config):
        self.config = config
        self._kline_cache: dict[str, list] = {}
        self._kline_htf_cache: dict[str, list] = {}
        self._last_signal: dict[str, ATRSignal] = {}
        self._mark_prices: dict[str, float] = {}
        self._partial_tp_state: dict[str, dict] = {}  # Track partial TP progress

    def update_mark_price(self, symbol: str, price: float):
        self._mark_prices[symbol] = price

    def update_klines(self, symbol: str, klines: list):
        self._kline_cache[symbol] = klines

    def update_klines_htf(self, symbol: str, klines: list):
        self._kline_htf_cache[symbol] = klines

    # === Indicator Calculations ===

    def calculate_atr(self, symbol: str) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < self.config.atr.atr_period + 1:
            return None
        period = self.config.atr.atr_period
        highs = np.array([float(k[2]) for k in klines[-(period + 1):]])
        lows = np.array([float(k[3]) for k in klines[-(period + 1):]])
        closes = np.array([float(k[4]) for k in klines[-(period + 1):]])
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        if len(tr_list) < period:
            return None
        return float(np.mean(tr_list[-period:]))

    def calculate_atr_ratio(self, symbol: str) -> Optional[float]:
        """Current ATR / Average ATR — detects volatility expansion/contraction."""
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < 50:
            return None
        period = self.config.atr.atr_period
        # Current ATR
        current_atr = self.calculate_atr(symbol)
        if current_atr is None:
            return None
        # ATR from 20 bars ago (average volatility baseline)
        old_klines = klines[-(period + 21):-(period + 1)]
        if len(old_klines) < period + 1:
            return 1.0
        old_highs = np.array([float(k[2]) for k in old_klines])
        old_lows = np.array([float(k[3]) for k in old_klines])
        old_closes = np.array([float(k[4]) for k in old_klines])
        old_trs = []
        for i in range(1, len(old_highs)):
            tr = max(old_highs[i] - old_lows[i], abs(old_highs[i] - old_closes[i-1]), abs(old_lows[i] - old_closes[i-1]))
            old_trs.append(tr)
        old_atr = float(np.mean(old_trs)) if old_trs else current_atr
        return current_atr / old_atr if old_atr > 0 else 1.0

    def calculate_ema(self, closes: np.ndarray, period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = float(closes[0])
        for price in closes[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < period + 1:
            return None
        closes = np.array([float(k[4]) for k in klines[-(period + 2):]])
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        return float(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))

    def calculate_volume_ratio(self, symbol: str, period: int = 20) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < period + 1:
            return None
        volumes = np.array([float(k[5]) for k in klines[-period:]])
        avg_vol = np.mean(volumes[:-1])
        if avg_vol == 0:
            return 1.0
        return float(volumes[-1] / avg_vol)

    def calculate_bollinger(self, closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
        if len(closes) < period:
            return None, None, None
        subset = closes[-period:]
        sma = float(np.mean(subset))
        std = float(np.std(subset))
        return sma + std_dev * std, sma, sma - std_dev * std

    def _get_closes(self, symbol: str) -> Optional[np.ndarray]:
        klines = self._kline_cache.get(symbol)
        if not klines:
            return None
        return np.array([float(k[4]) for k in klines])

    def _get_swing_low(self, symbol: str, lookback: int = 10) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < lookback:
            return None
        lows = [float(k[3]) for k in klines[-lookback:]]
        return min(lows)

    def _get_swing_high(self, symbol: str, lookback: int = 10) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < lookback:
            return None
        highs = [float(k[2]) for k in klines[-lookback:]]
        return max(highs)

    def detect_trend_regime(self, symbol: str) -> str:
        """Detect market regime using HTF data.
        Returns: 'strong_bull', 'bull', 'range', 'bear', 'strong_bear'
        """
        htf_klines = self._kline_htf_cache.get(symbol)
        if not htf_klines or len(htf_klines) < 50:
            return "range"

        htf_closes = np.array([float(k[4]) for k in htf_klines])
        htf_ema20 = self.calculate_ema(htf_closes, 20)
        htf_ema50 = self.calculate_ema(htf_closes, 50)
        if htf_ema20 is None or htf_ema50 is None:
            return "range"

        current_price = float(htf_closes[-1])
        ema_spread = (htf_ema20 - htf_ema50) / htf_ema50

        # Strong bull: price > EMA20 > EMA50, spread > 2%
        if current_price > htf_ema20 > htf_ema50 and ema_spread > 0.02:
            return "strong_bull"
        # Bull: EMA20 > EMA50, price > EMA50
        elif htf_ema20 > htf_ema50 and current_price > htf_ema50:
            return "bull"
        # Strong bear: price < EMA20 < EMA50, spread < -2%
        elif current_price < htf_ema20 < htf_ema50 and ema_spread < -0.02:
            return "strong_bear"
        # Bear: EMA20 < EMA50, price < EMA50
        elif htf_ema20 < htf_ema50 and current_price < htf_ema50:
            return "bear"
        return "range"

    # === Strategy Scorers ===

    def _score_trend(self, symbol: str, mark_price: float, closes: np.ndarray,
                     ema10: float, ema30: float, rsi: float, vol_ratio: float) -> tuple:
        """Trend following: EMA alignment + HTF trend + volume."""
        long_s, short_s = 0.0, 0.0
        reasons_l, reasons_s = [], []

        # HTF trend (strongest signal)
        htf_klines = self._kline_htf_cache.get(symbol)
        if htf_klines and len(htf_klines) >= 50:
            htf_closes = np.array([float(k[4]) for k in htf_klines])
            htf_ema20 = self.calculate_ema(htf_closes, 20)
            htf_ema50 = self.calculate_ema(htf_closes, 50)
            if htf_ema20 and htf_ema50:
                if htf_ema20 > htf_ema50 and mark_price > htf_ema20:
                    long_s += 0.30
                    reasons_l.append("htf_strong_uptrend")
                elif htf_ema20 < htf_ema50 and mark_price < htf_ema20:
                    short_s += 0.30
                    reasons_s.append("htf_strong_downtrend")
                elif htf_ema20 > htf_ema50:
                    long_s += 0.15
                    reasons_l.append("htf_weak_uptrend")
                elif htf_ema20 < htf_ema50:
                    short_s += 0.15
                    reasons_s.append("htf_weak_downtrend")

        # LTF EMA alignment
        if ema10 > ema30 * 1.001:
            long_s += 0.20
            reasons_l.append("ema_bullish_aligned")
        elif ema10 < ema30 * 0.999:
            short_s += 0.20
            reasons_s.append("ema_bearish_aligned")

        # Price above/below EMA30
        if mark_price > ema30:
            long_s += 0.10
        else:
            short_s += 0.10

        # RSI in trend direction
        if 45 <= rsi <= 65:
            long_s += 0.10 if ema10 > ema30 else 0
            short_s += 0.10 if ema10 < ema30 else 0
            reasons_l.append(f"rsi_trend_ok({rsi:.0f})")

        # Volume confirmation
        if vol_ratio >= 1.3:
            long_s += 0.15
            short_s += 0.15
            reasons_l.append(f"vol_confirm({vol_ratio:.1f}x)")

        return long_s, short_s, reasons_l, reasons_s

    def _score_mean_reversion(self, symbol: str, mark_price: float, closes: np.ndarray,
                               bb_upper: float, bb_mid: float, bb_lower: float,
                               rsi: float, vol_ratio: float) -> tuple:
        """Mean reversion: overextended RSI + Bollinger extremes."""
        long_s, short_s = 0.0, 0.0
        reasons_l, reasons_s = [], []

        # RSI extremes (primary signal)
        if rsi <= 25:
            long_s += 0.35
            reasons_l.append(f"rsi_deep_oversold({rsi:.0f})")
        elif rsi <= 35:
            long_s += 0.25
            reasons_l.append(f"rsi_oversold({rsi:.0f})")
        elif rsi >= 75:
            short_s += 0.35
            reasons_s.append(f"rsi_deep_overbought({rsi:.0f})")
        elif rsi >= 65:
            short_s += 0.25
            reasons_s.append(f"rsi_overbought({rsi:.0f})")

        # Bollinger extremes
        bb_width = bb_upper - bb_lower if bb_upper and bb_lower else 0
        if bb_lower and mark_price <= bb_lower:
            long_s += 0.25
            reasons_l.append("at_lower_bb")
        if bb_upper and mark_price >= bb_upper:
            short_s += 0.25
            reasons_s.append("at_upper_bb")

        # Volume spike on reversal (capitulation)
        if vol_ratio >= 2.0:
            if rsi < 40:
                long_s += 0.20
                reasons_l.append(f"capitulation_vol({vol_ratio:.1f}x)")
            elif rsi > 60:
                short_s += 0.20
                reasons_s.append(f"climax_vol({vol_ratio:.1f}x)")

        # Candle reversal pattern
        klines = self._kline_cache.get(symbol, [])
        if len(klines) >= 3:
            body_curr = float(klines[-1][4]) - float(klines[-1][1])
            body_prev = float(klines[-2][4]) - float(klines[-2][1])
            if body_prev < 0 and body_curr > 0:
                long_s += 0.15
                reasons_l.append("bullish_engulfing")
            elif body_prev > 0 and body_curr < 0:
                short_s += 0.15
                reasons_s.append("bearish_engulfing")

        return long_s, short_s, reasons_l, reasons_s

    def _score_breakout(self, symbol: str, mark_price: float, atr: float,
                         vol_ratio: float, atr_ratio: float) -> tuple:
        """Breakout: ATR expansion + volume surge + price at extremes."""
        long_s, short_s = 0.0, 0.0
        reasons_l, reasons_s = [], []

        # ATR expansion (volatility breakout)
        if atr_ratio and atr_ratio >= 1.5:
            long_s += 0.20
            short_s += 0.20
            reasons_l.append(f"vol_expansion({atr_ratio:.1f}x)")

        # Volume surge
        if vol_ratio >= 2.0:
            long_s += 0.25
            short_s += 0.25
            reasons_l.append(f"vol_surge({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.5:
            long_s += 0.15
            short_s += 0.15

        # Price near swing high/low
        swing_high = self._get_swing_high(symbol, 20)
        swing_low = self._get_swing_low(symbol, 20)
        if swing_high and mark_price >= swing_high * 0.998:
            long_s += 0.25
            reasons_l.append("near_swing_high_breakout")
        if swing_low and mark_price <= swing_low * 1.002:
            short_s += 0.25
            reasons_s.append("near_swing_low_breakdown")

        # Strong momentum candle
        klines = self._kline_cache.get(symbol, [])
        if len(klines) >= 2:
            range_curr = float(klines[-1][2]) - float(klines[-1][3])
            avg_range = atr if atr else range_curr
            if range_curr > avg_range * 1.5:
                body = abs(float(klines[-1][4]) - float(klines[-1][1]))
                if body > range_curr * 0.6:
                    if float(klines[-1][4]) > float(klines[-1][1]):
                        long_s += 0.20
                        reasons_l.append("strong_bull_breakout_candle")
                    else:
                        short_s += 0.20
                        reasons_s.append("strong_bear_breakdown_candle")

        return long_s, short_s, reasons_l, reasons_s

    # === Main Signal Generation ===

    def generate_signal(self, symbol: str, current_position_amt: float = 0) -> Optional[ATRSignal]:
        atr = self.calculate_atr(symbol)
        if atr is None or atr <= 0:
            return None

        mark_price = self._mark_prices.get(symbol)
        if mark_price is None:
            return None

        closes = self._get_closes(symbol)
        if closes is None or len(closes) < 35:
            return None

        # Calculate all indicators
        ema10 = self.calculate_ema(closes, 10)
        ema30 = self.calculate_ema(closes, 30)
        rsi = self.calculate_rsi(symbol)
        vol_ratio = self.calculate_volume_ratio(symbol)
        atr_ratio = self.calculate_atr_ratio(symbol)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger(closes)

        if any(v is None for v in [ema10, ema30, rsi, vol_ratio, bb_upper]):
            return None

        cfg = self.config.atr
        sl_mult = cfg.atr_sl_multiplier
        trail_mult = cfg.atr_trailing_multiplier
        ptp = cfg.partial_tp

        # === Position Management ===
        last = self._last_signal.get(symbol)
        tp_state = self._partial_tp_state.get(symbol, {"tp1_hit": False, "tp2_hit": False})

        if current_position_amt > 0 and last:
            # Check TP1 (partial close)
            if not tp_state.get("tp1_hit") and last.tp1_price > 0 and mark_price >= last.tp1_price:
                tp_state["tp1_hit"] = True
                self._partial_tp_state[symbol] = tp_state
                if ptp.move_sl_to_be:
                    last.stop_loss = last.entry_price  # Move SL to breakeven
                return ATRSignal(symbol=symbol, signal="PARTIAL_CLOSE", entry_price=mark_price,
                    atr_value=atr, stop_loss=last.stop_loss, take_profit=last.take_profit,
                    trailing_stop=last.trailing_stop, confidence=1.0,
                    reason=f"TP1 hit at {last.tp1_price:.2f}, close {ptp.tp1_pct*100}%",
                    tp1_price=last.tp1_price, tp2_price=last.tp2_price, tp3_price=last.tp3_price,
                    tp1_close_pct=ptp.tp1_pct, strategy_type=last.strategy_type)

            # Check TP2 (partial close)
            if tp_state.get("tp1_hit") and not tp_state.get("tp2_hit") and last.tp2_price > 0 and mark_price >= last.tp2_price:
                tp_state["tp2_hit"] = True
                self._partial_tp_state[symbol] = tp_state
                # Tighten trailing stop
                last.trailing_stop = max(last.trailing_stop, last.tp2_price - trail_mult * atr)
                return ATRSignal(symbol=symbol, signal="PARTIAL_CLOSE", entry_price=mark_price,
                    atr_value=atr, stop_loss=last.stop_loss, take_profit=last.take_profit,
                    trailing_stop=last.trailing_stop, confidence=1.0,
                    reason=f"TP2 hit at {last.tp2_price:.2f}, close {ptp.tp2_pct*100}%",
                    tp1_price=last.tp1_price, tp2_price=last.tp2_price, tp3_price=last.tp3_price,
                    tp2_close_pct=ptp.tp2_pct, strategy_type=last.strategy_type)

            # Stop loss
            if mark_price <= last.stop_loss:
                self._partial_tp_state.pop(symbol, None)
                return ATRSignal(symbol=symbol, signal="CLOSE_LONG", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="Stop loss hit" if not tp_state.get("tp1_hit") else "SL hit (after TP1, likely breakeven)")

            # Trailing stop for remaining position
            new_trailing = mark_price - trail_mult * atr
            if new_trailing > last.trailing_stop:
                last.trailing_stop = new_trailing
            if mark_price <= last.trailing_stop and tp_state.get("tp1_hit"):
                self._partial_tp_state.pop(symbol, None)
                return ATRSignal(symbol=symbol, signal="CLOSE_LONG", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=last.trailing_stop,
                    confidence=0.95, reason="Trailing stop (post-TP1)")

        elif current_position_amt < 0 and last:
            if not tp_state.get("tp1_hit") and last.tp1_price > 0 and mark_price <= last.tp1_price:
                tp_state["tp1_hit"] = True
                self._partial_tp_state[symbol] = tp_state
                if ptp.move_sl_to_be:
                    last.stop_loss = last.entry_price
                return ATRSignal(symbol=symbol, signal="PARTIAL_CLOSE", entry_price=mark_price,
                    atr_value=atr, stop_loss=last.stop_loss, take_profit=last.take_profit,
                    trailing_stop=last.trailing_stop, confidence=1.0,
                    reason=f"TP1 hit at {last.tp1_price:.2f}, close {ptp.tp1_pct*100}%",
                    tp1_price=last.tp1_price, tp2_price=last.tp2_price, tp3_price=last.tp3_price,
                    tp1_close_pct=ptp.tp1_pct, strategy_type=last.strategy_type)

            if tp_state.get("tp1_hit") and not tp_state.get("tp2_hit") and last.tp2_price > 0 and mark_price <= last.tp2_price:
                tp_state["tp2_hit"] = True
                self._partial_tp_state[symbol] = tp_state
                last.trailing_stop = min(last.trailing_stop, last.tp2_price + trail_mult * atr)
                return ATRSignal(symbol=symbol, signal="PARTIAL_CLOSE", entry_price=mark_price,
                    atr_value=atr, stop_loss=last.stop_loss, take_profit=last.take_profit,
                    trailing_stop=last.trailing_stop, confidence=1.0,
                    reason=f"TP2 hit at {last.tp2_price:.2f}, close {ptp.tp2_pct*100}%",
                    tp1_price=last.tp1_price, tp2_price=last.tp2_price, tp3_price=last.tp3_price,
                    tp2_close_pct=ptp.tp2_pct, strategy_type=last.strategy_type)

            if mark_price >= last.stop_loss:
                self._partial_tp_state.pop(symbol, None)
                return ATRSignal(symbol=symbol, signal="CLOSE_SHORT", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="Stop loss hit" if not tp_state.get("tp1_hit") else "SL hit (after TP1)")

            new_trailing = mark_price + trail_mult * atr
            if new_trailing < last.trailing_stop:
                last.trailing_stop = new_trailing
            if mark_price >= last.trailing_stop and tp_state.get("tp1_hit"):
                self._partial_tp_state.pop(symbol, None)
                return ATRSignal(symbol=symbol, signal="CLOSE_SHORT", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=last.trailing_stop,
                    confidence=0.95, reason="Trailing stop (post-TP1)")

        if current_position_amt != 0:
            return ATRSignal(symbol=symbol, signal="HOLD", entry_price=mark_price,
                atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                confidence=0, reason="Position open")

        # === Multi-Strategy Fusion ===
        # Run all three strategies
        t_l, t_s, t_rl, t_rs = self._score_trend(symbol, mark_price, closes, ema10, ema30, rsi, vol_ratio)
        mr_l, mr_s, mr_rl, mr_rs = self._score_mean_reversion(symbol, mark_price, closes, bb_upper, bb_mid, bb_lower, rsi, vol_ratio)
        bo_l, bo_s, bo_rl, bo_rs = self._score_breakout(symbol, mark_price, atr, vol_ratio, atr_ratio)

        # Fuse scores (weighted by strategy reliability)
        # Trend: most reliable, highest weight
        # Mean reversion: good in ranging markets
        # Breakout: high R:R but lower win rate
        long_score = t_l * 0.40 + mr_l * 0.30 + bo_l * 0.30
        short_score = t_s * 0.40 + mr_s * 0.30 + bo_s * 0.30

        # Determine dominant strategy type
        strategy_type = "trend"
        if mr_l > t_l and mr_l > bo_l:
            strategy_type = "mean_reversion"
        elif bo_l > t_l and bo_l > mr_l:
            strategy_type = "breakout"
        if short_score > long_score:
            if mr_s > t_s and mr_s > bo_s:
                strategy_type = "mean_reversion"
            elif bo_s > t_s and bo_s > mr_s:
                strategy_type = "breakout"

        # Anti-correlation penalty: if both strategies conflict, reduce confidence
        if long_score > 0.3 and short_score > 0.3:
            conflict_penalty = min(long_score, short_score) * 0.3
            long_score -= conflict_penalty
            short_score -= conflict_penalty

        # === Trend Regime Filter ===
        # In strong bull: suppress shorts entirely; in strong bear: suppress longs
        regime = self.detect_trend_regime(symbol)
        if regime in ("strong_bull", "bull"):
            short_score *= 0.1  # Heavily penalize counter-trend shorts
        elif regime in ("strong_bear", "bear"):
            long_score *= 0.1   # Heavily penalize counter-trend longs

        # Minimum score threshold
        MIN_SCORE = getattr(cfg, 'min_entry_score', 0.55)

        signal = "HOLD"
        confidence = 0.0
        reasons = []
        sl_price = 0.0
        tp1_price = 0.0
        tp2_price = 0.0
        tp3_price = 0.0

        sl_dist = sl_mult * atr

        if long_score >= MIN_SCORE and long_score > short_score:
            signal = "LONG"
            confidence = min(long_score, 1.0)
            reasons = t_rl + mr_rl + bo_rl
            sl_price = mark_price - sl_dist
            tp1_price = mark_price + ptp.tp1_rr * sl_dist  # 2R
            tp2_price = mark_price + ptp.tp2_rr * sl_dist  # 4R
            tp3_price = mark_price + ptp.tp3_rr * sl_dist  # 8R
        elif short_score >= MIN_SCORE and short_score > long_score:
            signal = "SHORT"
            confidence = min(short_score, 1.0)
            reasons = t_rs + mr_rs + bo_rs
            sl_price = mark_price + sl_dist
            tp1_price = mark_price - ptp.tp1_rr * sl_dist
            tp2_price = mark_price - ptp.tp2_rr * sl_dist
            tp3_price = mark_price - ptp.tp3_rr * sl_dist

        if signal != "HOLD":
            # Reset partial TP state
            self._partial_tp_state[symbol] = {"tp1_hit": False, "tp2_hit": False}

            result = ATRSignal(
                symbol=symbol, signal=signal, entry_price=mark_price, atr_value=atr,
                stop_loss=sl_price, take_profit=tp3_price, trailing_stop=sl_price,
                confidence=confidence,
                reason=f"[{strategy_type}] " + " | ".join(reasons[:5]) + f" | RSI={rsi:.0f} Vol={vol_ratio:.1f}x",
                tp1_price=tp1_price, tp2_price=tp2_price, tp3_price=tp3_price,
                tp1_close_pct=ptp.tp1_pct, tp2_close_pct=ptp.tp2_pct,
                strategy_type=strategy_type,
            )
            self._last_signal[symbol] = result
            logger.info(f"[{symbol}] {signal} [{strategy_type}] score={confidence:.2f}")
            return result

        return ATRSignal(symbol=symbol, signal="HOLD", entry_price=mark_price,
            atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
            confidence=0, reason=f"No signal | L={long_score:.2f} S={short_score:.2f} RSI={rsi:.0f}")

    def get_atr_value(self, symbol: str) -> Optional[float]:
        return self.calculate_atr(symbol)

    def get_mark_price(self, symbol: str) -> Optional[float]:
        return self._mark_prices.get(symbol)
