"""
Binance Quant Trader V3 - Position Sync
========================================
Synchronizes local position state with Binance exchange state.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("trader.pos_sync")


class PositionSync:
    """Keeps local position state in sync with Binance."""

    def __init__(self, client, config, db):
        self.client = client
        self.config = config
        self.db = db
        self._positions: dict[str, dict] = {}
        self._reconcile_interval = 300
        self._running = False

    async def start(self):
        self._running = True
        await self.full_reconcile()
        asyncio.create_task(self._reconcile_loop())

    async def stop(self):
        self._running = False

    async def _reconcile_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self._reconcile_interval)
                await self.full_reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
                await asyncio.sleep(30)

    async def full_reconcile(self):
        try:
            positions = await self.client.futures_position_information()
            exchange_positions = {}

            for pos in positions:
                symbol = pos["symbol"]
                if symbol not in self.config.symbols:
                    continue

                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue

                exchange_positions[symbol] = {
                    "position_side": pos.get("positionSide", "BOTH"),
                    "position_amt": amt,
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                    "margin_type": pos.get("marginType", "ISOLATED"),
                    "liquidation_price": float(pos.get("liquidationPrice", 0)),
                }
                self._positions[symbol] = exchange_positions[symbol]
                self.db.upsert_position(symbol, **exchange_positions[symbol])

            for symbol in list(self._positions.keys()):
                if symbol not in exchange_positions:
                    self._positions.pop(symbol, None)
                    self.db.clear_position(symbol)

            logger.info(
                f"Position reconcile: {len(exchange_positions)} open | "
                + " | ".join([
                    f"{s}: {p['position_amt']:+.6f} @ {p['entry_price']:.2f}"
                    for s, p in exchange_positions.items()
                ])
            )
        except Exception as e:
            logger.error(f"Full reconcile failed: {e}")

    async def process_user_data_update(self, msg: dict):
        event_type = msg.get("e")
        if event_type == "ACCOUNT_UPDATE":
            await self._handle_account_update(msg)
        elif event_type == "ORDER_TRADE_UPDATE":
            await self._handle_order_update(msg)

    async def _handle_account_update(self, msg: dict):
        account = msg.get("a", {})
        positions = account.get("P", [])

        for pos in positions:
            symbol = pos.get("s")
            if symbol not in self.config.symbols:
                continue

            amt = float(pos.get("pa", 0))
            entry_price = float(pos.get("ep", 0))
            unrealized_pnl = float(pos.get("up", 0))
            position_side = pos.get("ps", "BOTH")

            if amt == 0:
                self._positions.pop(symbol, None)
                self.db.clear_position(symbol)
                logger.info(f"Position closed: {symbol}")
            else:
                self._positions[symbol] = {
                    "position_side": position_side,
                    "position_amt": amt,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized_pnl,
                    "leverage": self._positions.get(symbol, {}).get("leverage", 10),
                    "margin_type": self._positions.get(symbol, {}).get("margin_type", "ISOLATED"),
                    "liquidation_price": 0,
                }
                self.db.upsert_position(symbol, **self._positions[symbol])
                logger.info(
                    f"Position update: {symbol} amt={amt:+.6f} entry={entry_price:.2f} uPnL={unrealized_pnl:.4f}"
                )

    async def _handle_order_update(self, msg: dict):
        order = msg.get("o", {})
        symbol = order.get("s")
        if symbol not in self.config.symbols:
            return

        status = order.get("X")
        side = order.get("S")
        order_type = order.get("o")
        executed_qty = float(order.get("z", 0))
        avg_price = float(order.get("ap", 0))
        realized_pnl = float(order.get("rp", 0))
        commission = float(order.get("n", 0))
        commission_asset = order.get("N", "")
        order_id = int(order.get("i", 0))

        if status in ("FILLED", "PARTIALLY_FILLED"):
            self.db.log_trade(
                symbol=symbol, side=side, position_side=order.get("ps", "BOTH"),
                order_type=order_type, quantity=executed_qty, price=avg_price,
                avg_price=avg_price, status=status, order_id=order_id,
                realized_pnl=realized_pnl, commission=commission,
                commission_asset=commission_asset,
            )
            if realized_pnl != 0:
                self.db.update_daily_pnl(realized_pnl, commission)

            logger.info(
                f"Order {status}: {symbol} {side} qty={executed_qty} "
                f"avg_price={avg_price:.2f} rPnL={realized_pnl:.4f} comm={commission:.4f}"
            )

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)

    def get_position_amt(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        return pos["position_amt"] if pos else 0.0

    def get_all_positions(self) -> dict:
        return dict(self._positions)

    def get_total_unrealized_pnl(self) -> float:
        return sum(p.get("unrealized_pnl", 0) for p in self._positions.values())

    async def setup_leverage_and_margin(self, symbol: str):
        sym_config = self.config.symbol_configs.get(symbol)
        if not sym_config:
            return

        try:
            await self.client.futures_change_margin_type(
                symbol=symbol, marginType=sym_config.margin_type
            )
            logger.info(f"{symbol}: margin type set to {sym_config.margin_type}")
        except Exception as e:
            if "No need to change" not in str(e):
                logger.warning(f"{symbol}: margin type change failed: {e}")

        try:
            await self.client.futures_change_leverage(
                symbol=symbol, leverage=sym_config.leverage
            )
            logger.info(f"{symbol}: leverage set to {sym_config.leverage}x")
        except Exception as e:
            logger.warning(f"{symbol}: leverage change failed: {e}")
