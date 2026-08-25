"""
Binance Quant Trader V2 - Configuration
========================================
Real trading configuration for Binance USDT-M Perpetual Futures.
API Key/Secret must be filled by the user before running.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SymbolConfig:
    """Per-symbol trading configuration."""
    symbol: str
    leverage: int = 10
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED
    trade_enabled: bool = True


@dataclass
class RiskConfig:
    """Global risk management parameters."""
    max_position_pct: float = 0.1        # Max 10% of balance per position
    max_total_exposure_pct: float = 0.5  # Max 50% total exposure
    max_daily_loss_pct: float = 0.05     # Max 5% daily loss -> halt trading
    max_open_orders: int = 20            # Max concurrent open orders
    max_correlated_positions: int = 4    # Max positions in same direction


@dataclass
class ATRConfig:
    """ATR-based stop loss / take profit parameters."""
    atr_period: int = 14
    atr_sl_multiplier: float = 2.0       # Stop loss = entry +/- 2.0 * ATR
    atr_tp_multiplier: float = 3.0       # Take profit = entry +/- 3.0 * ATR
    atr_trailing_multiplier: float = 1.5 # Trailing stop = 1.5 * ATR
    kline_interval: str = "15m"          # Kline interval for ATR calculation
    kline_limit: int = 100               # Number of klines to fetch


@dataclass
class WebSocketConfig:
    """WebSocket connection parameters."""
    reconnect_delay_base: float = 1.0    # Base delay in seconds
    reconnect_delay_max: float = 60.0    # Max delay in seconds
    reconnect_max_attempts: int = 50     # Max reconnection attempts before halt
    heartbeat_interval: int = 180        # Seconds between heartbeat checks
    user_data_stream_keepalive: int = 1800  # Listen key keepalive interval


@dataclass
class Config:
    """Main configuration container."""
    # === API Credentials (FILL THESE IN) ===
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")

    # === Trading Mode ===
    testnet: bool = False  # MUST be False for real trading

    # === Symbols ===
    symbols: list = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"
    ])

    # === Sub-configs ===
    symbol_configs: Dict[str, SymbolConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    atr: ATRConfig = field(default_factory=ATRConfig)
    ws: WebSocketConfig = field(default_factory=WebSocketConfig)

    # === Database ===
    db_path: str = "trader_data/trades.db"

    # === Logging ===
    log_level: str = "INFO"
    log_file: str = "trader_data/trader.log"

    # === Order Execution ===
    order_type: str = "MARKET"           # MARKET or LIMIT
    limit_price_offset_pct: float = 0.01 # Offset for limit orders (0.01%)
    default_quantity_usdt: float = 100.0 # Default order size in USDT
    min_notional_usdt: float = 5.0       # Binance minimum notional

    def __post_init__(self):
        if not self.symbol_configs:
            self.symbol_configs = {
                s: SymbolConfig(symbol=s, leverage=10, margin_type="ISOLATED")
                for s in self.symbols
            }

    def validate(self) -> list:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.api_key:
            errors.append("BINANCE_API_KEY is not set")
        if not self.api_secret:
            errors.append("BINANCE_API_SECRET is not set")
        if self.testnet:
            errors.append("testnet=True detected - this is REAL trading mode, set testnet=False")
        if not self.symbols:
            errors.append("No symbols configured")
        for sym, cfg in self.symbol_configs.items():
            if cfg.leverage > 10:
                errors.append(f"{sym}: leverage {cfg.leverage} exceeds max 10x")
            if cfg.margin_type not in ("ISOLATED", "CROSSED"):
                errors.append(f"{sym}: invalid margin_type {cfg.margin_type}")
        if self.risk.max_position_pct > 0.25:
            errors.append("max_position_pct > 25% is too aggressive")
        return errors


# Global config instance
config = Config()
