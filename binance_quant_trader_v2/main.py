"""
Binance Quant Trader V2 - Main Entry Point
============================================
Real trading bot for Binance USDT-M Perpetual Futures.

Symbols: BTC/ETH/BNB/SOL/XRP/DOGE/AVAX/LINK
Leverage: Up to 10x, Isolated Margin
Strategy: ATR-based entry/exit with trailing stop
Risk: Position sizing, daily loss cap, exposure limits

Usage:
    1. Set environment variables:
       export BINANCE_API_KEY="your_api_key"
       export BINANCE_API_SECRET="your_api_secret"

    2. Or edit config.py directly (not recommended for production)

    3. Run:
       python main.py

    4. With custom config:
       python main.py --config my_config.py

WARNING: This is a REAL TRADING bot. It uses real funds.
         Make sure you understand the risks before running.
"""

import asyncio
import signal
import sys
import os
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

from binance import AsyncClient

from config import Config, config as default_config
from db_logger import TradeDB
from ws_manager import WebSocketManager
from atr_strategy import ATRCalculator
from risk_manager import RiskManager
from position_sync import PositionSync
from trading_engine import TradingEngine


def setup_logging(config: Config):
    """Configure logging to both file and console."""
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.log_file, encoding="utf-8"),
        ]
    )

    # Suppress noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def print_banner(config: Config):
    """Print startup banner."""
    print("=" * 70)
    print("  BINANCE QUANT TRADER V2 - REAL TRADING MODE")
    print("  USDT-M Perpetual Futures")
    print("=" * 70)
    print(f"  Symbols:    {', '.join(config.symbols)}")
    print(f"  Leverage:   Up to {max(s.leverage for s in config.symbol_configs.values())}x ISOLATED")
    print(f"  Strategy:   ATR({config.atr.atr_period}) SL={config.atr.atr_sl_multiplier}x "
          f"TP={config.atr.atr_tp_multiplier}x Trail={config.atr.atr_trailing_multiplier}x")
    print(f"  Risk:       Max pos {config.risk.max_position_pct*100}% | "
          f"Max exposure {config.risk.max_total_exposure_pct*100}% | "
          f"Daily loss {config.risk.max_daily_loss_pct*100}%")
    print(f"  Database:   {config.db_path}")
    print(f"  Log:        {config.log_file}")
    print(f"  Testnet:    {config.testnet}")
    print(f"  API Key:    {config.api_key[:8]}...{config.api_key[-4:]}" if len(config.api_key) > 12 else "  API Key:    NOT SET")
    print("=" * 70)
    print("  *** THIS IS REAL TRADING. REAL MONEY IS AT RISK. ***")
    print("=" * 70)
    print()


class TraderApp:
    """Main application container."""

    def __init__(self, config: Config):
        self.config = config
        self.db = TradeDB(config.db_path)
        self.client: AsyncClient = None
        self.ws_manager: WebSocketManager = None
        self.atr_strategy: ATRCalculator = None
        self.risk_manager: RiskManager = None
        self.position_sync: PositionSync = None
        self.engine: TradingEngine = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Initialize and start all components."""
        logger = logging.getLogger("trader.main")

        # Validate config
        errors = self.config.validate()
        if errors:
            for e in errors:
                logger.critical(f"Config error: {e}")
            raise RuntimeError(f"Configuration invalid: {errors}")

        print_banner(self.config)

        # Connect to Binance
        logger.info("Connecting to Binance...")
        if self.config.testnet:
            self.client = await AsyncClient.create(
                self.config.api_key,
                self.config.api_secret,
                testnet=True
            )
        else:
            self.client = await AsyncClient.create(
                self.config.api_key,
                self.config.api_secret
            )

        # Verify connection
        server_time = await self.client.futures_ping()
        logger.info(f"Connected to Binance (testnet={self.config.testnet})")

        # Initialize components
        self.atr_strategy = ATRCalculator(self.config)
        self.risk_manager = RiskManager(self.config, self.db)
        self.position_sync = PositionSync(self.client, self.config, self.db)
        self.engine = TradingEngine(
            self.client, self.config, self.db,
            self.atr_strategy, self.risk_manager, self.position_sync
        )
        self.ws_manager = WebSocketManager(self.client, self.config)

        # Register WebSocket callbacks
        self.ws_manager.register_callback("user_data", self._on_user_data)
        self.ws_manager.register_callback("mark_price", self._on_mark_price)

        for symbol in self.config.symbols:
            self.ws_manager.register_callback(
                f"kline_{symbol}",
                lambda msg, s=symbol: self._on_kline(s, msg)
            )

        # Initialize engine (fetch exchange info, setup leverage)
        await self.engine.initialize()

        # Start position sync
        await self.position_sync.start()

        # Start WebSocket streams
        await self.ws_manager.start()

        # Log startup event
        self.db.log_event("SYSTEM", message="Trader V2 started",
                         data={"symbols": self.config.symbols})
        self.db.set_state("last_start", datetime.now(timezone.utc).isoformat())

        logger.info("All components started. Trading loop beginning...")

        # Start trading engine
        try:
            await self.engine.start()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Graceful shutdown."""
        logger = logging.getLogger("trader.main")
        logger.info("Initiating graceful shutdown...")

        # Stop trading engine first
        if self.engine:
            await self.engine.stop()

        # Stop WebSocket
        if self.ws_manager:
            await self.ws_manager.stop()

        # Stop position sync
        if self.position_sync:
            await self.position_sync.stop()

        # Close client
        if self.client:
            await self.client.close_connection()

        # Log shutdown
        self.db.log_event("SYSTEM", message="Trader V2 stopped")
        self.db.close()

        logger.info("Shutdown complete")

    async def _on_user_data(self, msg):
        """Handle user data stream messages."""
        if self.position_sync:
            await self.position_sync.process_user_data_update(msg)

    async def _on_mark_price(self, msg):
        """Handle mark price updates."""
        if isinstance(msg, list):
            for item in msg:
                symbol = item.get("s")
                price = float(item.get("p", 0))
                if symbol and self.atr_strategy:
                    self.atr_strategy.update_mark_price(symbol, price)
        elif isinstance(msg, dict):
            symbol = msg.get("s")
            price = float(msg.get("p", 0))
            if symbol and self.atr_strategy:
                self.atr_strategy.update_mark_price(symbol, price)

    async def _on_kline(self, symbol: str, msg):
        """Handle kline updates for ATR calculation."""
        if not self.atr_strategy:
            return

        kline = msg.get("k", {})
        if not kline:
            return

        # Update kline cache with latest data
        # For real-time ATR, we update the last candle
        try:
            klines = await self.client.futures_klines(
                symbol=symbol,
                interval=self.config.atr.kline_interval,
                limit=self.config.atr.kline_limit
            )
            self.atr_strategy.update_klines(symbol, klines)
        except Exception as e:
            logging.getLogger("trader.main").debug(f"Kline refresh error for {symbol}: {e}")


async def main():
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="Binance Quant Trader V2")
    parser.add_argument("--config", type=str, help="Path to custom config module")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    args = parser.parse_args()

    # Load config
    cfg = default_config

    # Override from environment if set
    if os.getenv("BINANCE_API_KEY"):
        cfg.api_key = os.getenv("BINANCE_API_KEY")
    if os.getenv("BINANCE_API_SECRET"):
        cfg.api_secret = os.getenv("BINANCE_API_SECRET")

    # Setup logging
    setup_logging(cfg)
    logger = logging.getLogger("trader.main")

    # Dry run mode
    if args.dry_run:
        errors = cfg.validate()
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("Configuration OK")
            print_banner(cfg)
            sys.exit(0)

    # Create and run app
    app = TraderApp(cfg)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
