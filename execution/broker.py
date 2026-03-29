"""
QuantSwarm v3 — Slippage-Aware Execution Layer
Market impact model, iceberg splitting, retry logic.
Paper mode first — always validate before going live.
"""
from __future__ import annotations
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger("quantswarm.execution")


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    ticker: str
    side: OrderSide
    size_pct: float           # % of portfolio
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    order_id: str = ""
    filled_price: float = 0.0
    filled_qty: float = 0.0
    status: str = "pending"   # pending | filled | partial | rejected | cancelled
    slippage_bps: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SlippageModel:
    """
    Market impact slippage model.
    Larger orders relative to volume = more slippage.
    """
    base_bps: float = 5.0              # 0.05% base slippage
    market_impact_coeff: float = 0.1   # impact coefficient
    max_adv_fraction: float = 0.05     # max 5% of avg daily volume

    def estimate(
        self,
        order_value_usd: float,
        avg_daily_volume_usd: float,
        is_crypto: bool = False,
    ) -> float:
        """Returns expected slippage in bps."""
        if avg_daily_volume_usd <= 0:
            return self.base_bps * 5  # high slippage for illiquid

        adv_fraction = order_value_usd / avg_daily_volume_usd
        # Linear market impact: impact = base + coeff * sqrt(participation_rate)
        impact_bps = self.base_bps + self.market_impact_coeff * (adv_fraction ** 0.5) * 100
        # Crypto gets 1.5x slippage due to wider spreads
        if is_crypto:
            impact_bps *= 1.5
        return float(impact_bps)

    def adjusted_price(self, price: float, side: OrderSide, slippage_bps: float) -> float:
        """Apply slippage to execution price."""
        factor = slippage_bps / 10000
        if side == OrderSide.BUY:
            return price * (1 + factor)   # pay more when buying
        else:
            return price * (1 - factor)   # receive less when selling


class PaperBroker:
    """
    Simulated paper trading broker.
    Applies realistic slippage and fills.
    """

    def __init__(self, slippage_model: SlippageModel):
        self.slippage_model = slippage_model
        self.orders = {}

    async def submit(self, order: Order, current_price: float, adv_usd: float) -> Order:
        """Simulate order execution with slippage."""
        is_crypto = "-USD" in order.ticker
        order_value = order.filled_qty * current_price if order.filled_qty else order.size_pct * 100000

        slippage_bps = self.slippage_model.estimate(order_value, adv_usd, is_crypto)
        filled_price = self.slippage_model.adjusted_price(current_price, order.side, slippage_bps)

        order.filled_price = round(filled_price, 6)
        order.filled_qty = order.size_pct  # as fraction
        order.slippage_bps = round(slippage_bps, 2)
        order.status = "filled"
        order.order_id = f"PAPER-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        logger.info(
            f"Paper fill: {order.ticker} {order.side.value} @ {order.filled_price:.4f} "
            f"(slippage {order.slippage_bps:.1f}bps)"
        )
        return order


class AlpacaBroker:
    """Live/paper trading via Alpaca API (stocks + crypto)."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self._client = None
        self._try_connect()

    def _try_connect(self):
        try:
            import alpaca_trade_api as tradeapi
            self._client = tradeapi.REST(
                self.api_key, self.secret_key, self.base_url, api_version="v2"
            )
            logger.info("Alpaca connected")
        except ImportError:
            logger.warning("alpaca_trade_api not installed — use: pip install alpaca-trade-api")
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")

    async def submit(self, order: Order, qty: float) -> Order:
        if not self._client:
            order.status = "rejected"
            order.order_id = "NO_CLIENT"
            return order

        for attempt in range(3):
            try:
                alpaca_order = self._client.submit_order(
                    symbol=order.ticker.replace("-USD", ""),
                    qty=qty,
                    side=order.side.value,
                    type=order.order_type.value,
                    time_in_force="day",
                    limit_price=order.limit_price,
                )
                order.order_id = alpaca_order.id
                order.status = alpaca_order.status
                order.filled_price = float(alpaca_order.filled_avg_price or 0)
                return order
            except Exception as e:
                logger.warning(f"Alpaca submit attempt {attempt+1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        order.status = "rejected"
        return order


class IcebergSplitter:
    """
    Splits large orders into smaller child orders to reduce market impact.
    Activates when order > threshold.
    """

    def __init__(self, threshold_usd: float = 10000, n_slices: int = 5):
        self.threshold = threshold_usd
        self.n_slices = n_slices

    def should_split(self, order_value_usd: float) -> bool:
        return order_value_usd > self.threshold

    def split(self, order: Order, n_slices: int = None) -> list:
        """Split order into equal child orders."""
        n = n_slices or self.n_slices
        child_size = order.size_pct / n
        children = []
        for i in range(n):
            child = Order(
                ticker=order.ticker,
                side=order.side,
                size_pct=child_size,
                order_type=OrderType.LIMIT,  # use limits for iceberg
                limit_price=order.limit_price,
            )
            children.append(child)
        return children


class ExecutionManager:
    """
    Master execution controller.
    Routes orders to paper or live broker, applies iceberg splitting,
    handles retries, and logs all fills.
    """

    def __init__(self, config: dict, portfolio_value: float):
        self.config = config
        self.portfolio_value = portfolio_value
        self.mode = config.get("mode", "paper")
        self.slippage_model = SlippageModel(
            base_bps=config.get("base_slippage_bps", 5),
            market_impact_coeff=config.get("market_impact_coefficient", 0.1),
            max_adv_fraction=config.get("max_order_pct_adv", 0.05),
        )
        self.iceberg = IcebergSplitter(
            threshold_usd=config.get("iceberg_threshold_usd", 10000),
        )

        if self.mode == "paper":
            self.broker = PaperBroker(self.slippage_model)
        else:
            import os
            self.broker = AlpacaBroker(
                api_key=os.getenv("ALPACA_API_KEY", ""),
                secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            )

        self.fill_log = []
        logger.info(f"ExecutionManager: mode={self.mode}")

    async def execute(
        self,
        ticker: str,
        direction: str,
        size_pct: float,
        current_price: float,
        adv_usd: float = 1_000_000,
    ) -> Order:
        """Execute a trade with slippage modeling and optional iceberg splitting."""
        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL
        order_value = size_pct * self.portfolio_value
        stop_loss_pct = self.config.get("stop_loss_pct", 0.05)

        # Calculate stop-loss and take-profit prices
        if direction == "LONG":
            stop_price = current_price * (1 - stop_loss_pct)
            tp_price = current_price * (1 + stop_loss_pct * 2.5)  # 2.5:1 R/R
        else:
            stop_price = current_price * (1 + stop_loss_pct)
            tp_price = current_price * (1 - stop_loss_pct * 2.5)

        order = Order(
            ticker=ticker,
            side=side,
            size_pct=size_pct,
            order_type=OrderType.LIMIT,
            limit_price=current_price * (1.001 if side == OrderSide.BUY else 0.999),
            stop_price=stop_price,
        )

        # Check if iceberg split needed
        if self.iceberg.should_split(order_value):
            children = self.iceberg.split(order)
            last_fill = None
            for child in children:
                last_fill = await self.broker.submit(child, current_price, adv_usd)
                await asyncio.sleep(0.5)  # small delay between child orders
            filled_order = last_fill or order
        else:
            filled_order = await self.broker.submit(order, current_price, adv_usd)

        self.fill_log.append(filled_order)
        return filled_order

    def get_fill_summary(self) -> dict:
        """Summary statistics of all fills."""
        if not self.fill_log:
            return {"total_fills": 0}
        avg_slippage = sum(f.slippage_bps for f in self.fill_log) / len(self.fill_log)
        return {
            "total_fills": len(self.fill_log),
            "avg_slippage_bps": round(avg_slippage, 2),
            "rejected": sum(1 for f in self.fill_log if f.status == "rejected"),
        }
