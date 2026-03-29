"""
QuantSwarm v3 — Risk Management Layer
Quarter-Kelly, max drawdown circuit breaker, correlation caps,
stop-losses, black swan protection — loss prevention first.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("quantswarm.risk")


class RiskStatus(Enum):
    NORMAL = "normal"
    CAUTION = "caution"           # approaching limits
    PAUSE = "pause"               # 3-loss streak or moderate drawdown
    LOCKDOWN = "lockdown"         # circuit breaker fired
    BLACK_SWAN = "black_swan"     # VIX > 40 or 3-sigma move


@dataclass
class Position:
    ticker: str
    direction: str          # LONG | SHORT
    size_pct: float         # % of portfolio
    entry_price: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: float
    current_price: float = 0.0
    pnl_pct: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        if self.direction == "LONG":
            self.pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - price) / self.entry_price

    def should_stop_loss(self) -> bool:
        if self.direction == "LONG":
            return self.current_price <= self.stop_loss_price
        else:
            return self.current_price >= self.stop_loss_price

    def should_take_profit(self) -> bool:
        if self.direction == "LONG":
            return self.current_price >= self.take_profit_price
        else:
            return self.current_price <= self.take_profit_price


@dataclass
class RiskState:
    status: RiskStatus = RiskStatus.NORMAL
    portfolio_value: float = 0.0
    peak_value: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    circuit_breaker_fired: bool = False
    circuit_breaker_reason: str = ""
    paused_until: Optional[datetime] = None


class KellySizer:
    """
    Quarter-Kelly position sizing.
    Prevents ruin while maximizing long-run growth.
    """

    def __init__(self, fraction: float = 0.25, max_pct: float = 0.10):
        self.fraction = fraction          # quarter Kelly = 0.25
        self.max_pct = max_pct           # absolute cap per position

    def size(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float,
        portfolio_value: float,
    ) -> float:
        """
        Returns fraction of portfolio to allocate (0 to max_pct).
        win_prob: probability of winning trade (0..1)
        win_return: expected return if win (e.g. 0.03 = 3%)
        loss_return: expected loss if lose (e.g. 0.02 = 2%, positive number)
        """
        if win_prob <= 0 or win_return <= 0 or loss_return <= 0:
            return 0.0

        loss_prob = 1 - win_prob
        # Full Kelly formula: f* = (p/l - q/w) = (p*w - q*l) / (w*l)
        # where p=win_prob, q=loss_prob, w=win_return, l=loss_return
        kelly = (win_prob * win_return - loss_prob * loss_return) / (win_return * loss_return)
        kelly = max(0.0, kelly)

        # Apply fraction
        fractional_kelly = kelly * self.fraction

        # Hard cap
        return float(min(fractional_kelly, self.max_pct))


class CorrelationGuard:
    """
    Prevents holding highly correlated positions simultaneously.
    Protects against correlated drawdowns during market stress.
    """

    # Pre-defined correlation clusters (simplified)
    CLUSTERS = {
        "mega_tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA"],
        "semis": ["AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "LRCX", "KLAC"],
        "financials": ["JPM", "GS", "MS", "BAC", "WFC", "V", "MA"],
        "crypto_major": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
        "crypto_defi": ["UNI-USD", "AAVE-USD", "GMX-USD", "DYDX-USD"],
        "energy": ["CVX", "XOM", "DVN", "COP", "HAL", "SLB"],
    }
    MAX_CLUSTER_EXPOSURE = 0.20  # max 20% in any one cluster

    def check(
        self,
        new_ticker: str,
        new_size_pct: float,
        existing_positions: Dict[str, Position],
    ) -> Tuple[bool, str]:
        """
        Returns (approved, reason).
        Returns False if adding position would exceed cluster exposure.
        """
        # Find which cluster new_ticker belongs to
        new_cluster = None
        for cluster_name, members in self.CLUSTERS.items():
            if new_ticker in members:
                new_cluster = cluster_name
                break

        if new_cluster is None:
            return True, "No cluster restriction"

        # Sum existing exposure in same cluster
        cluster_exposure = sum(
            pos.size_pct for ticker, pos in existing_positions.items()
            if ticker in self.CLUSTERS[new_cluster]
        )

        if cluster_exposure + new_size_pct > self.MAX_CLUSTER_EXPOSURE:
            return False, f"Cluster {new_cluster} already at {cluster_exposure:.1%} exposure (max {self.MAX_CLUSTER_EXPOSURE:.0%})"

        return True, "OK"


class DrawdownMonitor:
    """
    Monitors drawdown in real time and fires circuit breakers.
    The most important loss prevention component.
    """

    def __init__(self, config: dict):
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.15)
        self.caution_pct = self.max_drawdown_pct * 0.6    # 9% if max is 15%
        self.loss_streak_limit = config.get("loss_streak_limit", 3)
        self.vix_threshold = config.get("black_swan_vix_threshold", 40)
        self.sigma_threshold = config.get("sigma_threshold", 3.0)

    def assess(self, state: RiskState, current_value: float, market_data: dict = None) -> RiskStatus:
        """Assess risk status and update state."""
        market_data = market_data or {}

        # Update peak
        if current_value > state.peak_value:
            state.peak_value = current_value

        # Current drawdown
        if state.peak_value > 0:
            state.current_drawdown = (state.peak_value - current_value) / state.peak_value
        if state.current_drawdown > state.max_drawdown:
            state.max_drawdown = state.current_drawdown

        state.portfolio_value = current_value
        state.equity_curve.append(current_value)

        # Black swan check
        vix = market_data.get("vix", 0)
        price_shock = abs(market_data.get("market_return_1d", 0))
        if vix > self.vix_threshold or price_shock > self._sigma_threshold_return(state):
            state.status = RiskStatus.BLACK_SWAN
            state.circuit_breaker_fired = True
            state.circuit_breaker_reason = f"Black swan: VIX={vix:.1f}, shock={price_shock:.2%}"
            logger.critical(f"BLACK SWAN DETECTED: {state.circuit_breaker_reason}")
            return RiskStatus.BLACK_SWAN

        # Hard circuit breaker
        if state.current_drawdown >= self.max_drawdown_pct:
            state.status = RiskStatus.LOCKDOWN
            state.circuit_breaker_fired = True
            state.circuit_breaker_reason = f"Max drawdown {state.current_drawdown:.2%} >= {self.max_drawdown_pct:.2%}"
            logger.error(f"CIRCUIT BREAKER FIRED: {state.circuit_breaker_reason}")
            return RiskStatus.LOCKDOWN

        # Loss streak pause
        if state.consecutive_losses >= self.loss_streak_limit:
            state.status = RiskStatus.PAUSE
            state.paused_until = datetime.utcnow() + timedelta(hours=24)
            logger.warning(f"LOSS STREAK PAUSE: {state.consecutive_losses} consecutive losses")
            return RiskStatus.PAUSE

        # Caution zone
        if state.current_drawdown >= self.caution_pct:
            state.status = RiskStatus.CAUTION
            return RiskStatus.CAUTION

        state.status = RiskStatus.NORMAL
        return RiskStatus.NORMAL

    def _sigma_threshold_return(self, state: RiskState) -> float:
        """Compute 3-sigma daily return threshold from equity curve."""
        if len(state.equity_curve) < 20:
            return 0.08  # default 8%
        window = state.equity_curve[-61:]
        returns = np.diff(window) / np.array(window[:-1])
        if len(returns) < 5:
            return 0.08
        return float(np.mean(np.abs(returns)) + self.sigma_threshold * np.std(returns))

    def record_trade(self, state: RiskState, pnl_pct: float):
        """Update loss streak counter."""
        state.total_trades += 1
        if pnl_pct > 0:
            state.winning_trades += 1
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1

        state.trade_returns.append(pnl_pct)
        if len(state.trade_returns) > 1000:
            state.trade_returns = state.trade_returns[-500:]


class OvernightProtection:
    """Reduce exposure before market close to limit overnight gap risk."""

    def __init__(self, reduction_factor: float = 0.5):
        self.reduction_factor = reduction_factor

    def should_reduce(self) -> bool:
        """Returns True in the 30 minutes before market close (4pm ET)."""
        now = datetime.utcnow()
        # NYSE closes at 20:00 UTC (4pm ET)
        close_utc_hour = 20
        minutes_to_close = (close_utc_hour - now.hour) * 60 - now.minute
        return 0 < minutes_to_close <= 30

    def adjusted_size(self, original_size: float) -> float:
        if self.should_reduce():
            return original_size * self.reduction_factor
        return original_size


class RiskManager:
    """
    Master risk orchestrator — combines all risk components.
    The gatekeeper for every trade.
    """

    def __init__(self, config: dict, initial_capital: float):
        self.config = config
        self.state = RiskState(
            portfolio_value=initial_capital,
            peak_value=initial_capital,
        )
        self.kelly = KellySizer(
            fraction=config.get("kelly_fraction", 0.25),
            max_pct=config.get("max_position_pct", 0.10),
        )
        self.correlation_guard = CorrelationGuard()
        self.drawdown_monitor = DrawdownMonitor(config)
        self.overnight = OvernightProtection(
            config.get("overnight_size_reduction", 0.5)
        )
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.05)
        logger.info(f"RiskManager initialized: capital=${initial_capital:,.0f}, max_dd={config.get('max_drawdown_pct', 0.15):.0%}")

    def approve_trade(
        self,
        ticker: str,
        direction: str,
        confidence: float,
        win_prob: float,
        entry_price: float,
        market_data: dict = None,
    ) -> Tuple[bool, float, str]:
        """
        Returns (approved, position_size_pct, reason).
        The single entry point for ALL trade decisions.
        """
        market_data = market_data or {}

        # 1. Check current risk status
        status = self.drawdown_monitor.assess(
            self.state, self.state.portfolio_value, market_data
        )

        if status in (RiskStatus.LOCKDOWN, RiskStatus.BLACK_SWAN):
            return False, 0.0, f"BLOCKED: {status.value} — {self.state.circuit_breaker_reason}"

        if status == RiskStatus.PAUSE:
            if self.state.paused_until and datetime.utcnow() < self.state.paused_until:
                return False, 0.0, "BLOCKED: Loss streak pause active"

        # 2. Earnings blackout window check
        earnings_blackout = market_data.get("minutes_to_earnings", 9999)
        if earnings_blackout < self.config.get("blackout_pre_earnings_min", 30):
            return False, 0.0, f"BLOCKED: Earnings blackout ({earnings_blackout:.0f}min to event)"

        # 3. Low liquidity time block
        minutes_to_open = market_data.get("minutes_to_market_open", 999)
        minutes_to_close = market_data.get("minutes_to_market_close", 999)
        blackout_min = self.config.get("no_trade_open_close_min", 15)
        if minutes_to_open < blackout_min or minutes_to_close < blackout_min:
            return False, 0.0, "BLOCKED: Market open/close blackout window"

        # 4. Minimum confidence threshold
        if confidence < self.config.get("min_confidence", 0.60):
            return False, 0.0, f"BLOCKED: Confidence {confidence:.2f} below threshold"

        # 5. Kelly sizing
        win_return = 0.03  # expected win ~3%
        loss_return = self.stop_loss_pct
        size = self.kelly.size(win_prob, win_return, loss_return, self.state.portfolio_value)

        # 6. Regime-conditional position scaling
        # Drift tier is passed from the orchestrator's drift detector output:
        #   tier 0 = no drift    → full size
        #   tier 1 = KS p<0.05  → reduce to 75% (caution)
        #   tier 2 = KS p<0.01  → reduce to 50% (drift confirmed, retrain queued)
        #   tier 3 = KS p<0.001 → reduce to 25% (lockdown, human alert raised)
        drift_tier: int = market_data.get("drift_tier", 0)
        if drift_tier == 1:
            size *= 0.75
            logger.info(f"Regime scale Tier-1 drift: size reduced to 75%")
        elif drift_tier == 2:
            size *= 0.50
            logger.warning(f"Regime scale Tier-2 drift: size reduced to 50%")
        elif drift_tier >= 3:
            size *= 0.25
            logger.warning(f"Regime scale Tier-3 drift LOCKDOWN: size reduced to 25%")

        # Also halve in CAUTION status (consecutive-loss streak or moderate drawdown)
        if status == RiskStatus.CAUTION:
            size *= 0.5

        # 7. Overnight reduction
        size = self.overnight.adjusted_size(size)

        # 8. Absolute minimum size
        if size < 0.005:
            return False, 0.0, "BLOCKED: Position size too small to be meaningful"

        # 9. Correlation guard
        corr_ok, corr_reason = self.correlation_guard.check(
            ticker, size, self.state.positions
        )
        if not corr_ok:
            return False, 0.0, f"BLOCKED: {corr_reason}"

        # 10. Leverage check
        total_exposure = sum(p.size_pct for p in self.state.positions.values()) + size
        if total_exposure > self.config.get("max_leverage", 1.5):
            size = max(0, self.config.get("max_leverage", 1.5) - total_exposure + size)
            if size < 0.005:
                return False, 0.0, "BLOCKED: Max leverage reached"

        # All checks passed
        reason = f"APPROVED: size={size:.2%}, conf={confidence:.2f}, status={status.value}"
        logger.info(f"Trade approved: {ticker} {direction} {size:.2%}")
        return True, round(size, 4), reason

    def register_trade(self, position: Position):
        """Register an opened position."""
        self.state.positions[position.ticker] = position

    def close_trade(self, ticker: str, exit_price: float) -> float:
        """Close a position and return P&L."""
        pos = self.state.positions.pop(ticker, None)
        if not pos:
            return 0.0
        pos.update_price(exit_price)
        self.drawdown_monitor.record_trade(self.state, pos.pnl_pct)
        pnl_dollar = pos.pnl_pct * pos.size_pct * self.state.portfolio_value
        self.state.portfolio_value += pnl_dollar
        logger.info(f"Trade closed: {ticker}, P&L={pos.pnl_pct:.2%}, portfolio=${self.state.portfolio_value:,.0f}")
        return pos.pnl_pct

    def check_stops(self, prices: Dict[str, float]) -> List[str]:
        """Check all open positions for stop-loss or take-profit triggers."""
        to_close = []
        for ticker, pos in list(self.state.positions.items()):
            if ticker in prices:
                pos.update_price(prices[ticker])
                if pos.should_stop_loss():
                    logger.warning(f"STOP LOSS HIT: {ticker} at {prices[ticker]:.4f}")
                    to_close.append(ticker)
                elif pos.should_take_profit():
                    logger.info(f"TAKE PROFIT HIT: {ticker} at {prices[ticker]:.4f}")
                    to_close.append(ticker)
        return to_close

    def get_summary(self) -> dict:
        """Return current risk summary for dashboard."""
        win_rate = self.state.winning_trades / max(self.state.total_trades, 1)
        returns = self.state.trade_returns
        sharpe = 0.0
        if len(returns) > 20:
            r = np.array(returns)
            if np.std(r) > 0:
                sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252))
        return {
            "status": self.state.status.value,
            "portfolio_value": round(self.state.portfolio_value, 2),
            "current_drawdown": round(self.state.current_drawdown, 4),
            "max_drawdown": round(self.state.max_drawdown, 4),
            "consecutive_losses": self.state.consecutive_losses,
            "total_trades": self.state.total_trades,
            "win_rate": round(win_rate, 3),
            "sharpe_ratio": round(sharpe, 3),
            "open_positions": len(self.state.positions),
            "circuit_breaker_fired": self.state.circuit_breaker_fired,
        }
