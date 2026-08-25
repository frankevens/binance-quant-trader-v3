"""
Binance Quant Trader V2 - Risk Manager
========================================
Risk control: position sizing, exposure limits, daily loss caps,
correlation checks, and trading halt triggers.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger("trader.risk")


class RiskManager:
    """Enforces risk rules before and during trade execution."""

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self._daily_start_balance: Optional[float] = None
        self._current_balance: float = 0.0
        self._trading_halted = False
        self._halt_reason = ""
        self._open_positions: dict[str, float] = {}
        self._last_daily_reset: str = ""

    def update_balance(self, balance: float):
        self._current_balance = balance
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._last_daily_reset:
            self._daily_start_balance = balance
            self._last_daily_reset = today
            self._trading_halted = False
            self._halt_reason = ""
            logger.info(f"Daily reset: starting balance = {balance:.2f} USDT")

    def update_position(self, symbol: str, amount: float):
        if amount == 0:
            self._open_positions.pop(symbol, None)
        else:
            self._open_positions[symbol] = amount

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def check_pre_trade(self, symbol: str, side: str, quantity_usdt: float) -> tuple:
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"

        if self._current_balance <= 0:
            return False, "Account balance is zero or negative"

        max_position_usdt = self._current_balance * self.config.risk.max_position_pct
        if quantity_usdt > max_position_usdt:
            return False, (
                f"Position size {quantity_usdt:.2f} USDT exceeds limit "
                f"{max_position_usdt:.2f} USDT ({self.config.risk.max_position_pct*100}%)"
            )

        if quantity_usdt < self.config.min_notional_usdt:
            return False, f"Order {quantity_usdt:.2f} USDT below min notional {self.config.min_notional_usdt} USDT"

        total_exposure = sum(abs(amt) for amt in self._open_positions.values())
        total_exposure += quantity_usdt
        max_exposure = self._current_balance * self.config.risk.max_total_exposure_pct
        if total_exposure > max_exposure:
            return False, (
                f"Total exposure {total_exposure:.2f} USDT would exceed limit "
                f"{max_exposure:.2f} USDT ({self.config.risk.max_total_exposure_pct*100}%)"
            )

        today_count = self.db.get_today_trade_count()
        if today_count >= self.config.risk.max_open_orders * 10:
            return False, f"Today's trade count ({today_count}) too high"

        if side in ("BUY", "SELL"):
            long_count = sum(1 for amt in self._open_positions.values() if amt > 0)
            short_count = sum(1 for amt in self._open_positions.values() if amt < 0)
            if side == "BUY" and long_count >= self.config.risk.max_correlated_positions:
                return False, f"Max correlated long positions ({self.config.risk.max_correlated_positions}) reached"
            if side == "SELL" and short_count >= self.config.risk.max_correlated_positions:
                return False, f"Max correlated short positions ({self.config.risk.max_correlated_positions}) reached"

        if self._daily_start_balance and self._daily_start_balance > 0:
            daily_pnl = self.db.get_daily_pnl()
            if daily_pnl:
                daily_loss = abs(min(0, daily_pnl["realized_pnl"]))
                max_loss = self._daily_start_balance * self.config.risk.max_daily_loss_pct
                if daily_loss >= max_loss:
                    self._trading_halted = True
                    self._halt_reason = f"Daily loss limit hit: {daily_loss:.2f} >= {max_loss:.2f} USDT"
                    logger.critical(self._halt_reason)
                    return False, self._halt_reason

        return True, "OK"

    def calculate_position_size(self, symbol: str, atr_value: float,
                                 entry_price: float, risk_per_trade: float = None) -> float:
        if risk_per_trade is None:
            risk_per_trade = self._current_balance * self.config.risk.max_position_pct * 0.5

        max_usdt = self._current_balance * self.config.risk.max_position_pct

        if atr_value > 0:
            atr_risk_qty = risk_per_trade / (self.config.atr.atr_sl_multiplier * atr_value)
            atr_risk_usdt = atr_risk_qty * entry_price
        else:
            atr_risk_usdt = max_usdt

        final_usdt = min(atr_risk_usdt, max_usdt)

        if final_usdt < self.config.min_notional_usdt:
            return 0.0

        return round(final_usdt, 2)

    def get_risk_summary(self) -> dict:
        daily_pnl = self.db.get_daily_pnl()
        return {
            "balance": self._current_balance,
            "daily_start_balance": self._daily_start_balance,
            "daily_realized_pnl": daily_pnl["realized_pnl"] if daily_pnl else 0,
            "daily_commission": daily_pnl["commission"] if daily_pnl else 0,
            "open_positions": len(self._open_positions),
            "total_exposure": sum(abs(amt) for amt in self._open_positions.values()),
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "max_position_pct": self.config.risk.max_position_pct,
            "max_daily_loss_pct": self.config.risk.max_daily_loss_pct,
        }

    def force_halt(self, reason: str):
        self._trading_halted = True
        self._halt_reason = reason
        logger.critical(f"FORCED TRADING HALT: {reason}")

    def resume(self):
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
