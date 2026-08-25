"""
Binance Quant Trader V2 - Trading Engine
==========================================
Core trading engine that orchestrates strategy, risk management,
order execution, and position management.
"""

import asyncio
import logging
import time
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from binance import AsyncClient
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT,
    FUTURES_ORDER_TYPE_MARKET,
)

logger = logging.getLogger("trader.engine")


class TradingEngine:
    """Core trading engine coordinating strategy, risk, and execution."""

    def __init__(self, client: AsyncClient, config, db, atr_strategy, risk_manager, position_sync):
        self.client = client
        self.config = config
        self.db = db
        self.atr = atr_strategy
        self.risk = risk_manager
        self.pos_sync = position_sync
        self._running = False
        self._symbol_info: dict = {}
        self._tick_sizes: dict[str, float] = {}
        self._step_sizes: dict[str, float] = {}
        self._last_strategy_run: dict[str, float] = {}
        self._strategy_interval = 15

    async def initialize(self):
        logger.info("Initializing trading engine...")

        exchange_info = await self.client.futures_exchange_info()
        for s in exchange_info["symbols"]:
            if s["symbol"] in self.config.symbols:
                self._symbol_info[s["symbol"]] = s
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        self._tick_sizes[s["symbol"]] = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        self._step_sizes[s["symbol"]] = float(f["stepSize"])

        logger.info(f"Loaded symbol info for {len(self._symbol_info)} symbols")

        for symbol in self.config.symbols:
            await self.pos_sync.setup_leverage_and_margin(symbol)
            await asyncio.sleep(0.1)

        account = await self.client.futures_account_balance()
        for asset in account:
            if asset["asset"] == "USDT":
                balance = float(asset["balance"])
                self.risk.update_balance(balance)
                logger.info(f"Account balance: {balance:.2f} USDT")
                break

        for symbol in self.config.symbols:
            await self._fetch_initial_klines(symbol)
            await self._fetch_initial_klines_htf(symbol)
            await asyncio.sleep(0.1)

        logger.info("Trading engine initialized")

    async def _fetch_initial_klines(self, symbol: str):
        try:
            klines = await self.client.futures_klines(
                symbol=symbol,
                interval=self.config.atr.kline_interval,
                limit=self.config.atr.kline_limit
            )
            self.atr.update_klines(symbol, klines)
            logger.info(f"{symbol}: loaded {len(klines)} klines ({self.config.atr.kline_interval}), ATR={self.atr.calculate_atr(symbol):.4f}")
        except Exception as e:
            logger.error(f"{symbol}: failed to fetch klines: {e}")

    async def _fetch_initial_klines_htf(self, symbol: str):
        try:
            klines = await self.client.futures_klines(
                symbol=symbol,
                interval=self.config.atr.htf_interval,
                limit=self.config.atr.htf_kline_limit
            )
            self.atr.update_klines_htf(symbol, klines)
            logger.info(f"{symbol}: loaded {len(klines)} HTF klines ({self.config.atr.htf_interval})")
        except Exception as e:
            logger.error(f"{symbol}: failed to fetch HTF klines: {e}")

    async def start(self):
        self._running = True
        logger.info("Trading engine started")

        while self._running:
            try:
                await self._update_balance()

                for symbol in self.config.symbols:
                    if not self._running:
                        break

                    sym_config = self.config.symbol_configs.get(symbol)
                    if not sym_config or not sym_config.trade_enabled:
                        continue

                    now = time.time()
                    last_run = self._last_strategy_run.get(symbol, 0)
                    if now - last_run < self._strategy_interval:
                        continue

                    self._last_strategy_run[symbol] = now
                    await self._run_strategy(symbol)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        self._running = False
        logger.info("Trading engine stopped")

    async def _update_balance(self):
        try:
            account = await self.client.futures_account_balance()
            for asset in account:
                if asset["asset"] == "USDT":
                    self.risk.update_balance(float(asset["balance"]))
                    break
        except Exception as e:
            logger.warning(f"Balance update failed: {e}")

    async def _run_strategy(self, symbol: str):
        try:
            # Refresh klines before signal generation
            try:
                klines = await self.client.futures_klines(
                    symbol=symbol,
                    interval=self.config.atr.kline_interval,
                    limit=self.config.atr.kline_limit
                )
                self.atr.update_klines(symbol, klines)
            except Exception:
                pass

            # Refresh HTF klines every 5th cycle (~75s)
            if int(time.time()) % 75 < 2:
                try:
                    htf_klines = await self.client.futures_klines(
                        symbol=symbol,
                        interval=self.config.atr.htf_interval,
                        limit=self.config.atr.htf_kline_limit
                    )
                    self.atr.update_klines_htf(symbol, htf_klines)
                except Exception:
                    pass

            position_amt = self.pos_sync.get_position_amt(symbol)
            signal = self.atr.generate_signal(symbol, position_amt)

            if signal is None or signal.signal == "HOLD":
                return

            logger.info(
                f"[{symbol}] Signal: {signal.signal} | confidence={signal.confidence:.2f} | "
                f"ATR={signal.atr_value:.4f} | reason={signal.reason}"
            )

            self.db.log_event(
                event_type="SIGNAL", symbol=symbol,
                message=f"{signal.signal} conf={signal.confidence:.2f}",
                data={
                    "signal": signal.signal, "confidence": signal.confidence,
                    "atr": signal.atr_value, "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss, "take_profit": signal.take_profit,
                    "reason": signal.reason,
                }
            )

            if signal.signal in ("CLOSE_LONG", "CLOSE_SHORT"):
                await self._close_position(symbol, signal)
                return

            if signal.signal == "PARTIAL_CLOSE":
                await self._partial_close_position(symbol, signal)
                return

            if signal.signal in ("LONG", "SHORT"):
                await self._open_position(symbol, signal)

        except Exception as e:
            logger.error(f"[{symbol}] Strategy error: {e}")

    async def _open_position(self, symbol: str, signal):
        side = SIDE_BUY if signal.signal == "LONG" else SIDE_SELL

        position_usdt = self.risk.calculate_position_size(
            symbol, signal.atr_value, signal.entry_price
        )
        if position_usdt <= 0:
            logger.warning(f"[{symbol}] Position size calculation returned 0")
            return

        allowed, reason = self.risk.check_pre_trade(symbol, side, position_usdt)
        if not allowed:
            logger.warning(f"[{symbol}] Risk check failed: {reason}")
            self.db.log_event("RISK_BLOCK", symbol, reason)
            return

        quantity = self._calculate_quantity(symbol, position_usdt, signal.entry_price)
        if quantity <= 0:
            logger.warning(f"[{symbol}] Calculated quantity is 0")
            return

        try:
            order = await self._place_order(symbol, side, quantity)
            if order:
                self.pos_sync.update_position(symbol, quantity if side == SIDE_BUY else -quantity)
                self.risk.update_position(symbol, quantity if side == SIDE_BUY else -quantity)

                self.db.log_trade(
                    symbol=symbol, side=side, position_side="BOTH",
                    order_type=self.config.order_type, quantity=quantity,
                    price=signal.entry_price, status="SUBMITTED",
                    order_id=order.get("orderId"),
                    leverage=self.config.symbol_configs[symbol].leverage,
                    margin_type=self.config.symbol_configs[symbol].margin_type,
                    entry_price=signal.entry_price, stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.take_profit, atr_value=signal.atr_value,
                    strategy_signal=signal.signal, notes=signal.reason,
                    raw_response=order,
                )

                logger.info(
                    f"[{symbol}] OPENED {signal.signal}: qty={quantity} "
                    f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}"
                )

        except Exception as e:
            logger.error(f"[{symbol}] Order execution failed: {e}")
            self.db.log_event("ORDER_ERROR", symbol, str(e))

    async def _close_position(self, symbol: str, signal):
        position = self.pos_sync.get_position(symbol)
        if not position:
            logger.warning(f"[{symbol}] No position to close")
            return

        position_amt = position["position_amt"]
        if position_amt == 0:
            return

        side = SIDE_SELL if position_amt > 0 else SIDE_BUY
        quantity = abs(position_amt)

        try:
            order = await self._place_order(symbol, side, quantity, reduce_only=True)
            if order:
                self.pos_sync.update_position(symbol, 0)
                self.risk.update_position(symbol, 0)

                self.db.log_trade(
                    symbol=symbol, side=side, position_side="BOTH",
                    order_type="MARKET", quantity=quantity,
                    price=signal.entry_price, status="CLOSED",
                    order_id=order.get("orderId"),
                    strategy_signal=signal.signal, notes=signal.reason,
                    raw_response=order,
                )

                logger.info(f"[{symbol}] CLOSED {signal.signal}: qty={quantity} reason={signal.reason}")

        except Exception as e:
            logger.error(f"[{symbol}] Close order failed: {e}")
            self.db.log_event("CLOSE_ERROR", symbol, str(e))

    async def _partial_close_position(self, symbol: str, signal):
        """Partial close: close a percentage of position at TP1/TP2."""
        position = self.pos_sync.get_position(symbol)
        if not position:
            return

        position_amt = position["position_amt"]
        if position_amt == 0:
            return

        # Determine close percentage
        close_pct = signal.tp1_close_pct if signal.tp1_close_pct > 0 else signal.tp2_close_pct
        if close_pct <= 0:
            close_pct = 0.5  # Default 50%

        close_qty = abs(position_amt) * close_pct
        close_qty = self._calculate_quantity(symbol, close_qty * signal.entry_price, signal.entry_price)
        if close_qty <= 0:
            return

        side = SIDE_SELL if position_amt > 0 else SIDE_BUY

        try:
            order = await self._place_order(symbol, side, close_qty, reduce_only=True)
            if order:
                remaining = abs(position_amt) - close_qty
                new_amt = remaining if position_amt > 0 else -remaining
                self.pos_sync.update_position(symbol, new_amt)
                self.risk.update_position(symbol, new_amt)

                self.db.log_trade(
                    symbol=symbol, side=side, position_side="BOTH",
                    order_type="MARKET", quantity=close_qty,
                    price=signal.entry_price, status="PARTIAL_CLOSE",
                    order_id=order.get("orderId"),
                    strategy_signal="PARTIAL_CLOSE",
                    notes=f"{close_pct*100:.0f}% closed | {signal.reason}",
                    raw_response=order,
                )

                logger.info(
                    f"[{symbol}] PARTIAL CLOSE: {close_pct*100:.0f}% ({close_qty}) | "
                    f"remaining={remaining:.6f} | {signal.reason}"
                )

        except Exception as e:
            logger.error(f"[{symbol}] Partial close failed: {e}")
            self.db.log_event("PARTIAL_CLOSE_ERROR", symbol, str(e))

    async def _place_order(self, symbol: str, side: str, quantity: float,
                            reduce_only: bool = False) -> Optional[dict]:
        try:
            if self.config.order_type == "MARKET":
                order = await self.client.futures_create_order(
                    symbol=symbol, side=side,
                    type=FUTURES_ORDER_TYPE_MARKET,
                    quantity=quantity, reduceOnly=reduce_only,
                )
            else:
                mark_price = self.atr.get_mark_price(symbol)
                if mark_price is None:
                    logger.error(f"[{symbol}] No mark price for limit order")
                    return None

                offset = mark_price * self.config.limit_price_offset_pct / 100
                if side in (SIDE_BUY, "BUY"):
                    price = mark_price + offset
                else:
                    price = mark_price - offset

                price = self._round_price(symbol, price)

                order = await self.client.futures_create_order(
                    symbol=symbol, side=side, type=ORDER_TYPE_LIMIT,
                    price=price, quantity=quantity, timeInForce="GTC",
                    reduceOnly=reduce_only,
                )

            return order

        except Exception as e:
            logger.error(f"[{symbol}] Order placement error: {e}")
            self.db.log_event("ORDER_ERROR", symbol, str(e))
            return None

    def _calculate_quantity(self, symbol: str, usdt_amount: float, price: float) -> float:
        if price <= 0:
            return 0.0
        raw_qty = usdt_amount / price
        step_size = self._step_sizes.get(symbol, 0.001)
        qty = Decimal(str(raw_qty))
        step = Decimal(str(step_size))
        return float(qty.quantize(step, rounding=ROUND_DOWN))

    def _round_price(self, symbol: str, price: float) -> float:
        tick_size = self._tick_sizes.get(symbol, 0.01)
        p = Decimal(str(price))
        t = Decimal(str(tick_size))
        return float(p.quantize(t, rounding=ROUND_DOWN))

    async def get_status(self) -> dict:
        positions = self.pos_sync.get_all_positions()
        return {
            "running": self._running,
            "symbols": self.config.symbols,
            "positions": {
                s: {"amt": p["position_amt"], "entry": p["entry_price"], "uPnL": p["unrealized_pnl"]}
                for s, p in positions.items()
            },
            "total_unrealized_pnl": self.pos_sync.get_total_unrealized_pnl(),
            "risk": self.risk.get_risk_summary(),
        }
