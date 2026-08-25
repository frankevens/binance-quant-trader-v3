"""
Binance Quant Trader V2 - ATR Strategy (V2 Optimized)
======================================================
Multi-factor ATR strategy with RSI, volume, multi-timeframe trend filter.
Optimized for higher win rate while maintaining 2:1 reward/risk.
"""

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("trader.atr")


@dataclass
class ATRSignal:
    symbol: str
    signal: str  # "LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"
    entry_price: float
    atr_value: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    confidence: float
    reason: str


class ATRCalculator:
    """
    Multi-factor ATR strategy.

    Entry logic (score-based, 0-1.0):
      Trend filter (HTF EMA):     0.25
      RSI zone:                   0.20
      Bollinger position:         0.15
      LTF EMA alignment:          0.15
      Volume confirmation:        0.15
      Candle momentum:            0.10

    Minimum score to enter: 0.65 (configurable)
    """

    def __init__(self, config):
        self.config = config
        self._kline_cache: dict[str, list] = {}
        self._kline_htf_cache: dict[str, list] = {}  # Higher timeframe
        self._last_signal: dict[str, ATRSignal] = {}
        self._mark_prices: dict[str, float] = {}

    def update_mark_price(self, symbol: str, price: float):
        self._mark_prices[symbol] = price

    def update_klines(self, symbol: str, klines: list):
        self._kline_cache[symbol] = klines

    def update_klines_htf(self, symbol: str, klines: list):
        """Update higher timeframe klines (e.g. 1h for 15m strategy)."""
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
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        return float(np.mean(tr_list[-period:]))

    def calculate_ema(self, closes: np.ndarray, period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = float(closes[0])
        for price in closes[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
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
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def calculate_volume_ratio(self, symbol: str, period: int = 20) -> Optional[float]:
        """Current volume / average volume ratio."""
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < period + 1:
            return None

        volumes = np.array([float(k[5]) for k in klines[-period:]])
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1])

        if avg_vol == 0:
            return 1.0
        return float(current_vol / avg_vol)

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

    # === Signal Generation ===

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
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger(closes)

        if any(v is None for v in [ema10, ema30, rsi, vol_ratio, bb_upper]):
            return None

        cfg = self.config.atr
        stop_loss_dist = cfg.atr_sl_multiplier * atr
        take_profit_dist = cfg.atr_tp_multiplier * atr
        trailing_dist = cfg.atr_trailing_multiplier * atr

        # === Position Management (unchanged logic) ===
        last = self._last_signal.get(symbol)

        if current_position_amt > 0 and last:
            if mark_price <= last.stop_loss:
                return ATRSignal(symbol=symbol, signal="CLOSE_LONG", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="ATR stop loss hit")
            if mark_price >= last.take_profit:
                return ATRSignal(symbol=symbol, signal="CLOSE_LONG", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="ATR take profit hit")
            new_trailing = mark_price - trailing_dist
            if new_trailing > last.trailing_stop:
                last.trailing_stop = new_trailing
            if mark_price <= last.trailing_stop:
                return ATRSignal(symbol=symbol, signal="CLOSE_LONG", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=last.trailing_stop,
                    confidence=0.9, reason="ATR trailing stop hit")

        elif current_position_amt < 0 and last:
            if mark_price >= last.stop_loss:
                return ATRSignal(symbol=symbol, signal="CLOSE_SHORT", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="ATR stop loss hit")
            if mark_price <= last.take_profit:
                return ATRSignal(symbol=symbol, signal="CLOSE_SHORT", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                    confidence=1.0, reason="ATR take profit hit")
            new_trailing = mark_price + trailing_dist
            if new_trailing < last.trailing_stop:
                last.trailing_stop = new_trailing
            if mark_price >= last.trailing_stop:
                return ATRSignal(symbol=symbol, signal="CLOSE_SHORT", entry_price=mark_price,
                    atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=last.trailing_stop,
                    confidence=0.9, reason="ATR trailing stop hit")

        if current_position_amt != 0:
            return ATRSignal(symbol=symbol, signal="HOLD", entry_price=mark_price,
                atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
                confidence=0, reason="Position already open")

        # === Multi-Factor Entry Scoring ===

        # --- LONG scoring ---
        long_score = 0.0
        long_reasons = []

        # 1. HTF Trend filter (25% weight) - use 1h EMA if available, else 30 EMA as proxy
        htf_klines = self._kline_htf_cache.get(symbol)
        if htf_klines and len(htf_klines) >= 30:
            htf_closes = np.array([float(k[4]) for k in htf_klines])
            htf_ema20 = self.calculate_ema(htf_closes, 20)
            htf_ema50 = self.calculate_ema(htf_closes, 50)
            if htf_ema20 and htf_ema50:
                if htf_ema20 > htf_ema50 and mark_price > htf_ema20:
                    long_score += 0.25
                    long_reasons.append("htf_uptrend")
                elif htf_ema20 < htf_ema50 and mark_price < htf_ema20:
                    long_score -= 0.15  # Penalty: counter-trend
                    long_reasons.append("htf_downtrend_penalty")
        else:
            # Fallback: use LTF EMA30 as trend proxy
            if ema30 and mark_price > ema30:
                long_score += 0.15
                long_reasons.append("above_ema30")

        # 2. RSI zone (20% weight) - best entry: RSI 30-45 (oversold bounce)
        if 30 <= rsi <= 45:
            long_score += 0.20
            long_reasons.append(f"rsi_oversold_zone({rsi:.0f})")
        elif 45 < rsi <= 55:
            long_score += 0.10
            long_reasons.append(f"rsi_neutral({rsi:.0f})")
        elif rsi < 30:
            long_score += 0.05  # Too oversold = strong downtrend, risky
            long_reasons.append(f"rsi_deep_oversold({rsi:.0f})")
        elif rsi > 70:
            long_score -= 0.10  # Penalty: overbought
            long_reasons.append(f"rsi_overbought_penalty({rsi:.0f})")

        # 3. Bollinger position (15% weight)
        if mark_price <= bb_lower:
            long_score += 0.15
            long_reasons.append("below_lower_bb")
        elif mark_price <= bb_mid:
            long_score += 0.08
            long_reasons.append("below_mid_bb")

        # 4. LTF EMA alignment (15% weight)
        if ema10 > ema30:
            long_score += 0.15
            long_reasons.append("ema_bullish_cross")
        elif ema10 > ema30 * 0.998:
            long_score += 0.05
            long_reasons.append("ema_near_cross")

        # 5. Volume confirmation (15% weight)
        if vol_ratio >= 1.5:
            long_score += 0.15
            long_reasons.append(f"high_volume({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.0:
            long_score += 0.08
            long_reasons.append(f"normal_volume({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            long_score -= 0.05
            long_reasons.append(f"low_volume_penalty({vol_ratio:.1f}x)")

        # 6. Candle momentum (10% weight)
        klines = self._kline_cache.get(symbol, [])
        if len(klines) >= 3:
            c1 = float(klines[-1][4])
            c2 = float(klines[-2][4])
            c3 = float(klines[-3][4])
            if c1 > c2 > c3:
                long_score += 0.10
                long_reasons.append("strong_bullish_momentum")
            elif c1 > c2:
                long_score += 0.05
                long_reasons.append("bullish_candle")

        # --- SHORT scoring ---
        short_score = 0.0
        short_reasons = []

        # 1. HTF Trend
        if htf_klines and len(htf_klines) >= 30:
            htf_closes = np.array([float(k[4]) for k in htf_klines])
            htf_ema20 = self.calculate_ema(htf_closes, 20)
            htf_ema50 = self.calculate_ema(htf_closes, 50)
            if htf_ema20 and htf_ema50:
                if htf_ema20 < htf_ema50 and mark_price < htf_ema20:
                    short_score += 0.25
                    short_reasons.append("htf_downtrend")
                elif htf_ema20 > htf_ema50 and mark_price > htf_ema20:
                    short_score -= 0.15
                    short_reasons.append("htf_uptrend_penalty")
        else:
            if ema30 and mark_price < ema30:
                short_score += 0.15
                short_reasons.append("below_ema30")

        # 2. RSI zone
        if 55 <= rsi <= 70:
            short_score += 0.20
            short_reasons.append(f"rsi_overbought_zone({rsi:.0f})")
        elif 45 < rsi < 55:
            short_score += 0.10
            short_reasons.append(f"rsi_neutral({rsi:.0f})")
        elif rsi > 70:
            short_score += 0.05
            short_reasons.append(f"rsi_deep_overbought({rsi:.0f})")
        elif rsi < 30:
            short_score -= 0.10
            short_reasons.append(f"rsi_oversold_penalty({rsi:.0f})")

        # 3. Bollinger
        if mark_price >= bb_upper:
            short_score += 0.15
            short_reasons.append("above_upper_bb")
        elif mark_price >= bb_mid:
            short_score += 0.08
            short_reasons.append("above_mid_bb")

        # 4. LTF EMA
        if ema10 < ema30:
            short_score += 0.15
            short_reasons.append("ema_bearish_cross")
        elif ema10 < ema30 * 1.002:
            short_score += 0.05
            short_reasons.append("ema_near_cross")

        # 5. Volume
        if vol_ratio >= 1.5:
            short_score += 0.15
            short_reasons.append(f"high_volume({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.0:
            short_score += 0.08
            short_reasons.append(f"normal_volume({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            short_score -= 0.05
            short_reasons.append(f"low_volume_penalty({vol_ratio:.1f}x)")

        # 6. Candle momentum
        if len(klines) >= 3:
            c1 = float(klines[-1][4])
            c2 = float(klines[-2][4])
            c3 = float(klines[-3][4])
            if c1 < c2 < c3:
                short_score += 0.10
                short_reasons.append("strong_bearish_momentum")
            elif c1 < c2:
                short_score += 0.05
                short_reasons.append("bearish_candle")

        # === Decision ===
        MIN_SCORE = getattr(cfg, 'min_entry_score', 0.65)

        signal = "HOLD"
        confidence = 0.0
        reasons = []
        sl_price = 0.0
        tp_price = 0.0

        if long_score >= MIN_SCORE and long_score > short_score:
            signal = "LONG"
            confidence = min(long_score, 1.0)
            reasons = long_reasons
            sl_price = mark_price - stop_loss_dist
            tp_price = mark_price + take_profit_dist
        elif short_score >= MIN_SCORE and short_score > long_score:
            signal = "SHORT"
            confidence = min(short_score, 1.0)
            reasons = short_reasons
            sl_price = mark_price + stop_loss_dist
            tp_price = mark_price - take_profit_dist

        if signal != "HOLD":
            result = ATRSignal(
                symbol=symbol, signal=signal, entry_price=mark_price, atr_value=atr,
                stop_loss=sl_price, take_profit=tp_price, trailing_stop=sl_price,
                confidence=confidence,
                reason=" | ".join(reasons) + f" | RSI={rsi:.0f} Vol={vol_ratio:.1f}x"
            )
            self._last_signal[symbol] = result
            logger.info(f"[{symbol}] {signal} score={confidence:.2f} reasons={reasons}")
            return result

        return ATRSignal(symbol=symbol, signal="HOLD", entry_price=mark_price,
            atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
            confidence=0, reason=f"No signal | L={long_score:.2f} S={short_score:.2f} RSI={rsi:.0f}")

    def get_atr_value(self, symbol: str) -> Optional[float]:
        return self.calculate_atr(symbol)

    def get_mark_price(self, symbol: str) -> Optional[float]:
        return self._mark_prices.get(symbol)
