"""
Binance Quant Trader V2 - ATR Strategy
========================================
ATR (Average True Range) based strategy for entry signals,
stop loss, take profit, and trailing stop calculations.
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
    """Calculates ATR and generates trading signals."""

    def __init__(self, config):
        self.config = config
        self._kline_cache: dict[str, list] = {}
        self._last_signal: dict[str, ATRSignal] = {}
        self._mark_prices: dict[str, float] = {}

    def update_mark_price(self, symbol: str, price: float):
        self._mark_prices[symbol] = price

    def update_klines(self, symbol: str, klines: list):
        self._kline_cache[symbol] = klines

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

    def calculate_ema(self, symbol: str, period: int = 20) -> Optional[float]:
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < period:
            return None

        closes = np.array([float(k[4]) for k in klines[-period * 2:]])
        if len(closes) < period:
            return None

        multiplier = 2.0 / (period + 1)
        ema = float(closes[0])
        for price in closes[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def calculate_bollinger(self, symbol: str, period: int = 20, std_dev: float = 2.0):
        klines = self._kline_cache.get(symbol)
        if not klines or len(klines) < period:
            return None, None, None

        closes = np.array([float(k[4]) for k in klines[-period:]])
        sma = float(np.mean(closes))
        std = float(np.std(closes))
        return sma + std_dev * std, sma, sma - std_dev * std

    def generate_signal(self, symbol: str, current_position_amt: float = 0) -> Optional[ATRSignal]:
        atr = self.calculate_atr(symbol)
        if atr is None or atr <= 0:
            return None

        mark_price = self._mark_prices.get(symbol)
        if mark_price is None:
            return None

        ema_short = self.calculate_ema(symbol, 10)
        ema_long = self.calculate_ema(symbol, 30)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger(symbol)

        if ema_short is None or ema_long is None or bb_upper is None:
            return None

        cfg = self.config.atr
        stop_loss_dist = cfg.atr_sl_multiplier * atr
        take_profit_dist = cfg.atr_tp_multiplier * atr
        trailing_dist = cfg.atr_trailing_multiplier * atr

        last = self._last_signal.get(symbol)

        # Existing position management
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

        # Entry signals
        klines = self._kline_cache.get(symbol, [])

        long_confidence = 0.0
        long_reasons = []
        if mark_price <= bb_lower:
            long_confidence += 0.4
            long_reasons.append("price_below_lower_bb")
        if ema_short > ema_long:
            long_confidence += 0.3
            long_reasons.append("ema_uptrend")
        if len(klines) >= 2:
            if float(klines[-1][4]) > float(klines[-2][4]):
                long_confidence += 0.2
                long_reasons.append("bullish_candle")
        if mark_price < ema_short:
            long_confidence += 0.1
            long_reasons.append("below_short_ema")

        short_confidence = 0.0
        short_reasons = []
        if mark_price >= bb_upper:
            short_confidence += 0.4
            short_reasons.append("price_above_upper_bb")
        if ema_short < ema_long:
            short_confidence += 0.3
            short_reasons.append("ema_downtrend")
        if len(klines) >= 2:
            if float(klines[-1][4]) < float(klines[-2][4]):
                short_confidence += 0.2
                short_reasons.append("bearish_candle")
        if mark_price > ema_short:
            short_confidence += 0.1
            short_reasons.append("above_short_ema")

        signal = "HOLD"
        confidence = 0.0
        reasons = []
        sl_price = 0.0
        tp_price = 0.0
        MIN_CONFIDENCE = 0.6

        if long_confidence >= MIN_CONFIDENCE and long_confidence > short_confidence:
            signal = "LONG"
            confidence = long_confidence
            reasons = long_reasons
            sl_price = mark_price - stop_loss_dist
            tp_price = mark_price + take_profit_dist
        elif short_confidence >= MIN_CONFIDENCE and short_confidence > long_confidence:
            signal = "SHORT"
            confidence = short_confidence
            reasons = short_reasons
            sl_price = mark_price + stop_loss_dist
            tp_price = mark_price - take_profit_dist

        if signal != "HOLD":
            result = ATRSignal(
                symbol=symbol, signal=signal, entry_price=mark_price, atr_value=atr,
                stop_loss=sl_price, take_profit=tp_price, trailing_stop=sl_price,
                confidence=confidence, reason=" | ".join(reasons)
            )
            self._last_signal[symbol] = result
            return result

        return ATRSignal(symbol=symbol, signal="HOLD", entry_price=mark_price,
            atr_value=atr, stop_loss=0, take_profit=0, trailing_stop=0,
            confidence=0, reason="No signal")

    def get_atr_value(self, symbol: str) -> Optional[float]:
        return self.calculate_atr(symbol)

    def get_mark_price(self, symbol: str) -> Optional[float]:
        return self._mark_prices.get(symbol)
