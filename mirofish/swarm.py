"""
QuantSwarm v3 — MiroFish Swarm Intelligence Layer
1000-agent crowd simulation for macro sentiment consensus.
Inspired by MiroFish (github.com/666ghj/MiroFish).
"""
from __future__ import annotations
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger("quantswarm.mirofish")


class AgentPersonality(Enum):
    BULL = "bull"                    # trend-following optimist
    BEAR = "bear"                    # contrarian pessimist
    NEUTRAL = "neutral"              # index follower
    MOMENTUM = "momentum"            # follows recent moves
    CONTRARIAN = "contrarian"        # fades momentum
    FUNDAMENTALS = "fundamentals"    # value-based
    PANIC = "panic"                  # fear-driven, amplifies crashes


@dataclass
class SwarmAgent:
    agent_id: int
    personality: AgentPersonality
    conviction: float         # 0 to 1 — how strongly they act on signals
    memory_length: int        # how many periods they remember
    position: float = 0.0    # -1 (fully short) to +1 (fully long)
    wealth: float = 1.0
    history: List[float] = field(default_factory=list)

    def process_signal(
        self,
        market_signal: float,
        sentiment_signal: float,
        price_return: float,
        peer_consensus: float,
    ) -> float:
        """
        Agent processes signals and updates position.
        Returns updated position (-1 to +1).
        """
        if self.personality == AgentPersonality.BULL:
            signal = 0.4 * market_signal + 0.3 * sentiment_signal + 0.3 * peer_consensus
            signal = signal * (1 + 0.1 * random.gauss(0, 1))  # noise
            self.position = np.clip(self.position + self.conviction * signal, -1, 1)

        elif self.personality == AgentPersonality.BEAR:
            signal = -0.4 * market_signal - 0.2 * sentiment_signal + 0.2 * peer_consensus
            signal = signal * (1 + 0.1 * random.gauss(0, 1))
            self.position = np.clip(self.position + self.conviction * signal, -1, 1)

        elif self.personality == AgentPersonality.MOMENTUM:
            # Follows recent price moves with lag
            signal = 0.6 * price_return + 0.4 * peer_consensus
            self.position = np.clip(self.position + self.conviction * signal * 2, -1, 1)

        elif self.personality == AgentPersonality.CONTRARIAN:
            # Fades when crowd is extreme
            crowd_extreme = abs(peer_consensus) > 0.7
            if crowd_extreme:
                signal = -np.sign(peer_consensus) * 0.5
            else:
                signal = market_signal * 0.3
            self.position = np.clip(self.position + self.conviction * signal, -1, 1)

        elif self.personality == AgentPersonality.FUNDAMENTALS:
            # Mostly ignores short-term noise, reacts to macro
            signal = market_signal * 0.2 + random.gauss(0, 0.05)
            self.position = np.clip(self.position * 0.95 + self.conviction * signal, -1, 1)

        elif self.personality == AgentPersonality.PANIC:
            # Dramatically amplifies negative signals
            if market_signal < -0.3 or peer_consensus < -0.5:
                signal = -0.8
            elif market_signal > 0.3:
                signal = 0.3
            else:
                signal = 0.0
            self.position = np.clip(self.position + self.conviction * signal, -1, 1)

        else:  # NEUTRAL
            signal = 0.3 * peer_consensus + random.gauss(0, 0.03)
            self.position = np.clip(self.position + self.conviction * signal, -1, 1)

        self.history.append(self.position)
        if len(self.history) > self.memory_length:
            self.history = self.history[-self.memory_length:]

        return self.position


@dataclass
class SwarmResult:
    ticker: str
    consensus_signal: float      # -1 to +1 (average agent position)
    bullish_fraction: float      # fraction of agents bullish
    bearish_fraction: float      # fraction of agents bearish
    panic_threshold_reached: bool
    crowd_extreme: bool          # > 70% consensus = potential reversal
    emergent_behaviors: List[str]
    confidence: float
    timestamp: datetime


class MiroFishSwarm:
    """
    Swarm simulation engine.
    Creates N agents with different personalities and runs them
    through signal inputs to generate emergent market consensus.
    """

    PERSONALITY_DISTRIBUTION = {
        AgentPersonality.BULL: 0.20,
        AgentPersonality.BEAR: 0.15,
        AgentPersonality.NEUTRAL: 0.25,
        AgentPersonality.MOMENTUM: 0.15,
        AgentPersonality.CONTRARIAN: 0.10,
        AgentPersonality.FUNDAMENTALS: 0.10,
        AgentPersonality.PANIC: 0.05,
    }

    def __init__(self, config: dict):
        self.n_agents = config.get("n_agents", 1000)
        self.consensus_threshold = config.get("consensus_threshold", 0.60)
        self.panic_threshold = config.get("panic_threshold", 0.30)
        self.seed = config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.agents = self._create_agents()
        self._simulation_steps = 0
        logger.info(f"MiroFish: {self.n_agents} agents initialized")

    def _create_agents(self) -> List[SwarmAgent]:
        agents = []
        personalities = []
        for personality, fraction in self.PERSONALITY_DISTRIBUTION.items():
            count = int(self.n_agents * fraction)
            personalities.extend([personality] * count)
        # Fill remainder with neutral
        while len(personalities) < self.n_agents:
            personalities.append(AgentPersonality.NEUTRAL)
        random.shuffle(personalities)

        for i, p in enumerate(personalities):
            agents.append(SwarmAgent(
                agent_id=i,
                personality=p,
                conviction=random.uniform(0.1, 0.9),
                memory_length=random.randint(5, 50),
                position=random.uniform(-0.2, 0.2),  # small random starting position
            ))
        return agents

    def run_step(
        self,
        ticker: str,
        market_signal: float,
        sentiment_signal: float,
        price_return: float,
    ) -> SwarmResult:
        """
        Run one step of the swarm simulation.
        Each agent processes signals and updates its position.
        Emergent behaviors arise from agent interactions.
        """
        self._simulation_steps += 1

        # First pass: get current peer consensus (previous positions)
        peer_consensus = np.mean([a.position for a in self.agents])

        # Second pass: update all agents
        positions = []
        for agent in self.agents:
            # Add slight heterogeneity to inputs
            agent_market = market_signal * (1 + random.gauss(0, 0.05))
            agent_sentiment = sentiment_signal * (1 + random.gauss(0, 0.08))
            agent_price = price_return * (1 + random.gauss(0, 0.03))
            pos = agent.process_signal(
                agent_market, agent_sentiment, agent_price, peer_consensus
            )
            positions.append(pos)

        positions = np.array(positions)
        consensus = float(np.mean(positions))
        bullish_frac = float(np.mean(positions > 0.1))
        bearish_frac = float(np.mean(positions < -0.1))

        # Detect emergent behaviors
        emergent = []
        panic_agents = sum(1 for a in self.agents if a.personality == AgentPersonality.PANIC and a.position < -0.5)
        if panic_agents > self.n_agents * 0.03:
            emergent.append(f"panic_selling: {panic_agents} panic agents short")

        if abs(consensus) > 0.7:
            emergent.append(f"crowd_extreme: {consensus:.2f} consensus (potential reversal)")

        # Herding: sudden position clustering
        pos_std = float(np.std(positions))
        if pos_std < 0.15 and abs(consensus) > 0.4:
            emergent.append(f"herding_detected: std={pos_std:.3f}, consensus={consensus:.2f}")

        # Bimodal distribution = divided market
        upper_half = np.mean(positions > 0.3)
        lower_half = np.mean(positions < -0.3)
        if upper_half > 0.35 and lower_half > 0.35:
            emergent.append("bimodal_split: market highly divided")

        panic_reached = bearish_frac >= self.panic_threshold
        crowd_extreme = abs(consensus) > 0.7
        confidence = min(abs(consensus) * 1.5, 0.95)

        result = SwarmResult(
            ticker=ticker,
            consensus_signal=round(consensus, 4),
            bullish_fraction=round(bullish_frac, 3),
            bearish_fraction=round(bearish_frac, 3),
            panic_threshold_reached=panic_reached,
            crowd_extreme=crowd_extreme,
            emergent_behaviors=emergent,
            confidence=round(confidence, 3),
            timestamp=datetime.utcnow(),
        )

        if emergent:
            logger.warning(f"MiroFish [{ticker}] emergent: {emergent}")

        return result

    def run_multi_step(
        self,
        ticker: str,
        signals: List[dict],
        n_steps: int = 10,
    ) -> SwarmResult:
        """Run multiple steps for more stable consensus."""
        results = []
        for i, signal in enumerate(signals[:n_steps]):
            r = self.run_step(
                ticker,
                signal.get("market", 0.0),
                signal.get("sentiment", 0.0),
                signal.get("price_return", 0.0),
            )
            results.append(r)

        # Return most recent result with averaged consensus
        if not results:
            return self.run_step(ticker, 0, 0, 0)

        avg_consensus = np.mean([r.consensus_signal for r in results])
        final = results[-1]
        final.consensus_signal = round(float(avg_consensus), 4)
        return final


class SwarmOrchestrator:
    """
    Manages MiroFish simulations across all 100 instruments.
    Runs on a schedule and caches results.
    """

    def __init__(self, config: dict, tickers: List[str]):
        self.swarm = MiroFishSwarm(config)
        self.tickers = tickers
        self.cache: Dict[str, SwarmResult] = {}
        self.last_run: Optional[datetime] = None
        self.run_interval_hours = config.get("run_interval_hours", 6)

    def run_all(self, signal_data: Dict[str, dict]) -> Dict[str, SwarmResult]:
        """Run swarm simulation for all tickers."""
        logger.info(f"Running MiroFish swarm for {len(self.tickers)} instruments...")
        results = {}
        for ticker in self.tickers:
            data = signal_data.get(ticker, {})
            signals = [data] * 5  # repeat for stability
            result = self.swarm.run_multi_step(ticker, signals, n_steps=5)
            results[ticker] = result
            self.cache[ticker] = result

        self.last_run = datetime.utcnow()

        # Log overall market mood
        all_consensus = [r.consensus_signal for r in results.values()]
        market_mood = np.mean(all_consensus)
        panic_count = sum(1 for r in results.values() if r.panic_threshold_reached)
        logger.info(f"Swarm complete: market_mood={market_mood:.3f}, panic_tickers={panic_count}")
        return results

    def get_scores(self) -> Dict[str, float]:
        """Return consensus signal per ticker (-1 to +1)."""
        return {ticker: r.consensus_signal for ticker, r in self.cache.items()}

    def needs_update(self) -> bool:
        if self.last_run is None:
            return True
        from datetime import timedelta
        return datetime.utcnow() - self.last_run > timedelta(hours=self.run_interval_hours)
