"""
QuantSwarm v3 — Main Orchestrator
FRAMEWORM-AGENT ReAct loop: Observe -> Think -> Act
The central controller that connects all layers.
"""
from __future__ import annotations
import asyncio
import logging
import os
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from ingestion.sources import IngestionManager
from nlp.pipeline import NLPPipeline
from drift.detector import DriftDetector, DriftTier, MarketRegime
from prediction.engine import PredictionEngine
from mirofish.swarm import SwarmOrchestrator
from risk.manager import RiskManager, Position
from execution.broker import ExecutionManager
from explainability.shap_engine import SHAPExplainer

logger = logging.getLogger("quantswarm.agent")


def load_config(path: str = "config/base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class AlertManager:
    """Send alerts via Telegram/Slack."""

    def __init__(self, config: dict):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = config.get("telegram", True) and bool(self.telegram_token)

    async def send(self, message: str, level: str = "info"):
        prefix = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "success": "✅"}.get(level, "ℹ️")
        full_msg = f"{prefix} QuantSwarm v3\n{message}\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

        if self.enabled:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                        json={"chat_id": self.telegram_chat_id, "text": full_msg},
                        timeout=5.0,
                    )
            except Exception as e:
                logger.debug(f"Alert send error: {e}")

        # Always log
        log_fn = {"critical": logger.critical, "warning": logger.warning}.get(level, logger.info)
        log_fn(f"ALERT [{level.upper()}]: {message}")


class QuantSwarmAgent:
    """
    The FRAMEWORM-AGENT ReAct loop.

    OBSERVE:  Pull signals from all sources
    THINK:    NLP -> drift check -> predict -> risk assess
    ACT:      Execute / alert / pause based on outputs
    """

    def __init__(self, config: dict, initial_capital: float = 100_000):
        self.config = config
        self.initial_capital = initial_capital
        self.tickers = (
            config["instruments"]["stocks"] +
            config["instruments"]["crypto"]
        )
        logger.info(f"QuantSwarm: {len(self.tickers)} instruments loaded")

        # Initialize all layers
        self.ingestion = IngestionManager(self.tickers, config.get("ingestion", {}))
        self.nlp = NLPPipeline(self.tickers, config.get("nlp", {}))
        self.drift = DriftDetector(config.get("drift", {}))
        self.prediction = PredictionEngine(config.get("prediction", {}))
        self.mirofish = SwarmOrchestrator(config.get("mirofish", {}), self.tickers)
        self.risk = RiskManager(config.get("risk", {}), initial_capital)
        self.execution = ExecutionManager(config.get("execution", {}), initial_capital)
        self.explainer = SHAPExplainer()
        self.alert = AlertManager(config.get("alerts", {}))

        self._cycle_count = 0
        self._running = False

    async def observe(self) -> dict:
        """Layer 1: Ingest all signals."""
        raw_signals = await self.ingestion.fetch_all(lookback_hours=2)
        processed = self.nlp.process(raw_signals)
        ticker_sentiments = self.nlp.aggregate_by_ticker(processed)
        logger.info(f"Observe: {len(raw_signals)} raw -> {len(processed)} processed for {len(ticker_sentiments)} tickers")
        return {
            "raw_count": len(raw_signals),
            "processed_count": len(processed),
            "ticker_sentiments": ticker_sentiments,
        }

    async def think(self, observations: dict) -> dict:
        """Layer 2: Drift detection + prediction + swarm."""
        ticker_sentiments = observations.get("ticker_sentiments", {})

        # Collect sentiment scores as array for drift detection
        if ticker_sentiments:
            sentiment_scores = np.array([
                v["sentiment"] for v in ticker_sentiments.values()
            ])
            labels = [
                "positive" if v["sentiment"] > 0.1 else "negative" if v["sentiment"] < -0.1 else "neutral"
                for v in ticker_sentiments.values()
            ]
        else:
            sentiment_scores = np.zeros(10)
            labels = ["neutral"] * 10

        # Drift detection
        drift_result = self.drift.detect(sentiment_scores, labels)

        # MiroFish swarm (if update needed)
        mirofish_scores = {}
        if self.mirofish.needs_update():
            signal_data = {
                ticker: {
                    "market": ticker_sentiments.get(ticker, {}).get("sentiment", 0),
                    "sentiment": ticker_sentiments.get(ticker, {}).get("sentiment", 0),
                    "price_return": 0.0,
                }
                for ticker in self.tickers
            }
            swarm_results = self.mirofish.run_all(signal_data)
            mirofish_scores = {t: r.consensus_signal for t, r in swarm_results.items()}

        # Build predictions
        ticker_features = {
            ticker: {
                "sentiment_weighted": ticker_sentiments.get(ticker, {}).get("sentiment", 0),
                "sentiment_momentum": 0.0,
                "sec_insider_signal": 0.0,
                "news_count_24h": len(ticker_sentiments.get(ticker, {}).get("signals", [])),
                "reddit_sentiment": 0.0,
                "price_return_1h": 0.0,
                "price_return_24h": 0.0,
                "price_return_7d": 0.0,
                "volume_ratio": 1.0,
                "volatility_7d": 0.15,
                "funding_rate": 0.0,
                "drift_score": drift_result.confidence,
                "mirofish_consensus": mirofish_scores.get(ticker, 0.0),
            }
            for ticker in self.tickers
        }

        predictions = self.prediction.predict_all(
            ticker_features, ticker_sentiments, mirofish_scores
        )

        # Filter to actionable predictions
        actionable = [
            p for p in predictions
            if p.confidence >= self.config.get("prediction", {}).get("min_confidence", 0.60)
            and p.direction != "NEUTRAL"
        ]
        logger.info(f"Think: drift={drift_result.tier.value}, {len(actionable)}/{len(predictions)} actionable predictions")

        return {
            "drift_result": drift_result,
            "predictions": predictions,
            "actionable": actionable,
            "mirofish_scores": mirofish_scores,
            "ticker_features": ticker_features,
        }

    async def act(self, thoughts: dict, observations: dict) -> dict:
        """Layer 3: Risk check -> execute -> alert."""
        drift = thoughts["drift_result"]
        actionable = thoughts["actionable"]
        ticker_features = thoughts.get("ticker_features", {})

        actions_taken = []

        # Check if trading is blocked
        if not self.drift.should_trade():
            msg = f"Trading PAUSED: {drift.tier.value} — {drift.description}"
            await self.alert.send(msg, "warning")
            logger.warning(msg)
            return {"actions": [], "paused": True, "reason": msg}

        # Fetch real-time prices for ALL open positions + actionable tickers
        tickers_needed = (
            list(self.risk.state.positions.keys())
            + [p.ticker for p in actionable[:5]]
        )
        current_prices: Dict[str, float] = {}
        if tickers_needed:
            try:
                import yfinance as yf
                unique_tickers = list(set(tickers_needed))
                data = yf.download(unique_tickers, period="1d", interval="1m",
                                   group_by="ticker", progress=False, threads=True)
                for t in unique_tickers:
                    try:
                        if len(unique_tickers) == 1:
                            df = data
                        else:
                            df = data[t]
                        price = float(df["Close"].dropna().iloc[-1])
                        current_prices[t] = price
                    except Exception:
                        # Fall back to last known feature price
                        current_prices[t] = ticker_features.get(t, {}).get("current_price", 0.0)
            except Exception as e:
                logger.warning(f"Live price fetch failed: {e}. Using feature prices.")
                for t in tickers_needed:
                    current_prices[t] = ticker_features.get(t, {}).get("current_price", 0.0)

        # Check stop-losses on open positions (always runs)
        stops_hit = self.risk.check_stops(current_prices)
        for ticker in stops_hit:
            price = current_prices.get(ticker, 0)
            pnl = self.risk.close_trade(ticker, price)
            actions_taken.append({"type": "stop_loss", "ticker": ticker, "pnl": pnl})
            await self.alert.send(f"Stop-loss executed: {ticker}, P&L={pnl:.2%}", "warning")

        # Execute actionable predictions
        for pred in actionable[:5]:  # max 5 new trades per cycle
            features = ticker_features.get(pred.ticker, {})
            win_prob = pred.confidence
            # Use live price; fall back to feature price, then last close from market data
            current_price = (
                current_prices.get(pred.ticker)
                or features.get("current_price")
                or features.get("price")
                or 1.0
            )

            # Risk approval
            approved, size_pct, reason = self.risk.approve_trade(
                ticker=pred.ticker,
                direction=pred.direction,
                confidence=pred.confidence,
                win_prob=win_prob,
                entry_price=current_price,
            )

            if not approved:
                logger.debug(f"Trade rejected: {pred.ticker} — {reason}")
                continue

            # Generate SHAP explanation
            explanation = self.explainer.explain(pred, features)
            logger.info(f"\n{explanation.human_readable}")

            # Execute
            fill = await self.execution.execute(
                ticker=pred.ticker,
                direction=pred.direction,
                size_pct=size_pct,
                current_price=current_price,
            )

            if fill.status == "filled":
                stop_price = current_price * (1 - self.config["risk"]["stop_loss_pct"])
                tp_price = current_price * (1 + self.config["risk"]["stop_loss_pct"] * 2.5)
                position = Position(
                    ticker=pred.ticker,
                    direction=pred.direction,
                    size_pct=size_pct,
                    entry_price=fill.filled_price,
                    entry_time=datetime.utcnow(),
                    stop_loss_price=stop_price,
                    take_profit_price=tp_price,
                    current_price=fill.filled_price,
                )
                self.risk.register_trade(position)
                actions_taken.append({
                    "type": "open",
                    "ticker": pred.ticker,
                    "direction": pred.direction,
                    "size": size_pct,
                    "price": fill.filled_price,
                    "slippage_bps": fill.slippage_bps,
                    "confidence": pred.confidence,
                })
                # Send signal alert
                await self.alert.send(
                    f"NEW SIGNAL: {pred.direction} {pred.ticker}\n"
                    f"Confidence: {pred.confidence:.1%}\n"
                    f"Expected: {pred.magnitude*100:+.1f}% ({pred.horizon})\n"
                    f"Size: {size_pct:.1%} of portfolio",
                    "success"
                )

        return {"actions": actions_taken, "paused": False}

    async def run_cycle(self):
        """One full ReAct cycle: Observe -> Think -> Act."""
        self._cycle_count += 1
        logger.info(f"=== Cycle {self._cycle_count} === {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        observations = await self.observe()
        thoughts = await self.think(observations)
        result = await self.act(thoughts, observations)

        # Risk summary
        risk_summary = self.risk.get_summary()
        logger.info(f"Risk: {risk_summary}")

        return {
            "cycle": self._cycle_count,
            "observations": observations,
            "thoughts": {
                "drift": thoughts["drift_result"].description,
                "regime": thoughts["drift_result"].regime.value,
                "actionable_predictions": len(thoughts["actionable"]),
            },
            "result": result,
            "risk": risk_summary,
        }

    async def run(self, interval_sec: int = 900):
        """Main loop — runs every 15 minutes by default."""
        self._running = True
        await self.alert.send("QuantSwarm v4 started", "info")

        while self._running:
            try:
                await self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Stopping QuantSwarm...")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                await self.alert.send(f"Cycle error: {str(e)[:200]}", "warning")

            logger.info(f"Sleeping {interval_sec}s until next cycle...")
            await asyncio.sleep(interval_sec)

    def stop(self):
        self._running = False


async def main():
    """Entry point for paper trading."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    config = load_config("config/base.yaml")
    capital = float(os.getenv("INITIAL_CAPITAL", "100000"))
    agent = QuantSwarmAgent(config, initial_capital=capital)
    await agent.run(interval_sec=config["ingestion"]["refresh_interval_sec"])


if __name__ == "__main__":
    asyncio.run(main())
