"""
Binance Quant Trader V2 - WebSocket Manager
============================================
Manages Binance Futures WebSocket streams with automatic reconnection,
heartbeat monitoring, and user data stream keepalive.
"""

import asyncio
import time
import logging
import json
from typing import Callable, Optional

import aiohttp
from binance import AsyncClient, BinanceSocketManager
from binance.enums import FuturesType

logger = logging.getLogger("trader.ws")


class WebSocketManager:
    """
    Manages multiple WebSocket streams with:
    - Exponential backoff reconnection
    - Heartbeat monitoring
    - User data stream keepalive
    - Graceful shutdown
    """

    def __init__(self, client: AsyncClient, config):
        self.client = client
        self.config = config
        self.bsm = BinanceSocketManager(client)
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._last_message_time: dict[str, float] = {}
        self._reconnect_count: dict[str, int] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._listen_key: Optional[str] = None
        self._listen_key_expiry: float = 0

    def register_callback(self, stream_name: str, callback: Callable):
        if stream_name not in self._callbacks:
            self._callbacks[stream_name] = []
        self._callbacks[stream_name].append(callback)

    async def start(self):
        """Start all WebSocket streams."""
        self._running = True
        logger.info("Starting WebSocket manager...")

        # Start user data stream
        await self._start_user_data_stream()

        # Start kline streams for ATR calculation
        for symbol in self.config.symbols:
            await self._start_kline_stream(symbol)

        # Start mark price streams
        await self._start_mark_price_stream()

        # Start heartbeat monitor
        self._tasks["heartbeat"] = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"WebSocket manager started with {len(self._tasks)} streams")

    async def stop(self):
        """Gracefully stop all streams."""
        self._running = False
        logger.info("Stopping WebSocket manager...")

        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close user data stream
        if self._listen_key:
            try:
                await self.client.futures_cancel_listen_key(self._listen_key)
            except Exception as e:
                logger.warning(f"Failed to cancel listen key: {e}")

        self._tasks.clear()
        logger.info("WebSocket manager stopped")

    async def _start_user_data_stream(self):
        """Start the user data stream for order/position updates."""
        async def _run():
            while self._running:
                try:
                    # Get fresh listen key
                    self._listen_key = await self.client.futures_create_listen_key()
                    self._listen_key_expiry = time.time() + 3600
                    logger.info(f"User data stream started, listen_key={self._listen_key[:8]}...")

                    socket = self.bsm.futures_user_socket()
                    async with socket as stream:
                        while self._running:
                            msg = await asyncio.wait_for(stream.recv(), timeout=300)
                            self._last_message_time["user_data"] = time.time()
                            await self._dispatch("user_data", msg)

                except asyncio.TimeoutError:
                    logger.warning("User data stream timeout, reconnecting...")
                    await self._reconnect_delay("user_data")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"User data stream error: {e}")
                    await self._reconnect_delay("user_data")

        self._tasks["user_data"] = asyncio.create_task(_run())

    async def _start_kline_stream(self, symbol: str):
        """Start kline stream for a specific symbol."""
        stream_name = f"kline_{symbol}"

        async def _run():
            while self._running:
                try:
                    socket = self.bsm.futures_kline_socket(
                        symbol=symbol,
                        interval=self.config.atr.kline_interval
                    )
                    async with socket as stream:
                        while self._running:
                            msg = await asyncio.wait_for(stream.recv(), timeout=300)
                            self._last_message_time[stream_name] = time.time()
                            await self._dispatch(stream_name, msg)

                except asyncio.TimeoutError:
                    logger.warning(f"{stream_name} timeout, reconnecting...")
                    await self._reconnect_delay(stream_name)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"{stream_name} error: {e}")
                    await self._reconnect_delay(stream_name)

        self._tasks[stream_name] = asyncio.create_task(_run())

    async def _start_mark_price_stream(self):
        """Start mark price stream for all symbols."""
        stream_name = "mark_price"

        async def _run():
            while self._running:
                try:
                    socket = self.bsm.futures_mark_price_socket()
                    async with socket as stream:
                        while self._running:
                            msg = await asyncio.wait_for(stream.recv(), timeout=300)
                            self._last_message_time[stream_name] = time.time()
                            await self._dispatch(stream_name, msg)

                except asyncio.TimeoutError:
                    logger.warning(f"{stream_name} timeout, reconnecting...")
                    await self._reconnect_delay(stream_name)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"{stream_name} error: {e}")
                    await self._reconnect_delay(stream_name)

        self._tasks[stream_name] = asyncio.create_task(_run())

    async def _dispatch(self, stream_name: str, msg):
        """Dispatch message to registered callbacks."""
        callbacks = self._callbacks.get(stream_name, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(msg)
                else:
                    cb(msg)
            except Exception as e:
                logger.error(f"Callback error in {stream_name}: {e}")

    async def _reconnect_delay(self, stream_name: str):
        """Exponential backoff before reconnection."""
        count = self._reconnect_count.get(stream_name, 0) + 1
        self._reconnect_count[stream_name] = count

        if count > self.config.ws.reconnect_max_attempts:
            logger.critical(f"{stream_name}: max reconnection attempts ({count}) exceeded")
            self._running = False
            return

        delay = min(
            self.config.ws.reconnect_delay_base * (2 ** (count - 1)),
            self.config.ws.reconnect_delay_max
        )
        logger.info(f"{stream_name}: reconnecting in {delay:.1f}s (attempt {count})")
        await asyncio.sleep(delay)

    async def _heartbeat_loop(self):
        """Monitor stream health and keepalive user data stream."""
        while self._running:
            try:
                await asyncio.sleep(self.config.ws.heartbeat_interval)

                # Check stream health
                now = time.time()
                for name, last_time in self._last_message_time.items():
                    gap = now - last_time
                    if gap > self.config.ws.heartbeat_interval * 2:
                        logger.warning(f"Stream {name} silent for {gap:.0f}s")

                # Keepalive listen key
                if self._listen_key and time.time() > self._listen_key_expiry - 600:
                    try:
                        await self.client.futures_keepalive_listen_key(self._listen_key)
                        self._listen_key_expiry = time.time() + 3600
                        logger.debug("Listen key keepalive sent")
                    except Exception as e:
                        logger.warning(f"Listen key keepalive failed: {e}")
                        # Try to get new listen key
                        try:
                            self._listen_key = await self.client.futures_create_listen_key()
                            self._listen_key_expiry = time.time() + 3600
                            logger.info("New listen key obtained")
                        except Exception as e2:
                            logger.error(f"Failed to get new listen key: {e2}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_status(self) -> dict:
        now = time.time()
        return {
            name: {
                "running": not task.done(),
                "last_msg_ago": now - self._last_message_time.get(name, 0),
                "reconnects": self._reconnect_count.get(name, 0)
            }
            for name, task in self._tasks.items()
        }
