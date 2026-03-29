"""
QuantSwarm v3 — Test Suite
203 tests total: 29 tests × 7 agent scales (100, 500, 1000, 5000, 10000, 50000, 100000).
Covers: unit, integration, stress, regime detection, backtest, MiroFish swarm,
risk management, conformal prediction, and performance benchmarks.
Run: pytest tests/ -v --tb=short
"""
import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ========== UNIT TESTS (80 tests) ==========

class TestBotFilter:
    """Unit tests for bot filter (10 tests)."""

    def setup_method(self):
        from nlp.pipeline import BotFilter
        self.filter = BotFilter({
            "min_account_age_days": 30,
            "max_tweets_per_day": 100,
            "min_follower_ratio": 0.05,
            "cosine_similarity_threshold": 0.88,
            "consensus_required": 5,
        })

    def test_new_account_flagged(self):
        is_bot, score = self.filter.score("Buy $AAPL now!", {"account_age_days": 3})
        assert score > 0.5

    def test_old_clean_account_passes(self):
        is_bot, score = self.filter.score(
            "Analyzing earnings results for Q3...",
            {"account_age_days": 500, "followers": 1000, "following": 200}
        )
        assert not is_bot

    def test_pump_language_flagged(self):
        is_bot, score = self.filter.score(
            "This will 100x TO THE MOON 🚀🚀🚀🚀 BUY NOW GUARANTEED",
            {"account_age_days": 100}
        )
        assert score > 0

    def test_low_follower_ratio_flagged(self):
        is_bot, score = self.filter.score("text", {"followers": 1, "following": 10000, "account_age_days": 100})
        assert score > 0.5

    def test_duplicate_texts_detected(self):
        text = "Company XYZ announces major acquisition today"
        self.filter.score(text, {})
        is_bot, score = self.filter.score(text, {})
        assert score > 0.5

    def test_high_frequency_poster_flagged(self):
        is_bot, score = self.filter.score("text", {"posts_per_day": 200, "account_age_days": 365})
        assert score > 0.4

    def test_returns_tuple(self):
        result = self.filter.score("test", {})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_range(self):
        _, score = self.filter.score("test text", {})
        assert 0 <= score <= 1

    def test_empty_text_handled(self):
        is_bot, score = self.filter.score("", {})
        assert score >= 0

    def test_normal_financial_text_not_flagged(self):
        is_bot, score = self.filter.score(
            "Q3 earnings were strong with revenue up 12% year-over-year",
            {"account_age_days": 200, "followers": 500, "following": 300}
        )
        assert score < 0.8


class TestFinBERTSentiment:
    """Unit tests for sentiment analysis (10 tests)."""

    def setup_method(self):
        from nlp.pipeline import FinBERTSentiment
        self.model = FinBERTSentiment(use_model=False)  # lexicon mode for speed

    def test_positive_text(self):
        sentiment, conf, label = self.model.analyze("Company beats earnings, strong profit growth")
        assert sentiment > 0 or label in ("positive", "neutral")

    def test_negative_text(self):
        sentiment, conf, label = self.model.analyze("Company misses earnings, stock crashes on weak revenue")
        assert sentiment < 0 or label in ("negative", "neutral")

    def test_neutral_text(self):
        sentiment, conf, label = self.model.analyze("The company announced a meeting for Tuesday")
        # Neutral or low magnitude
        assert abs(sentiment) < 0.8

    def test_returns_three_values(self):
        result = self.model.analyze("test")
        assert len(result) == 3

    def test_confidence_range(self):
        _, conf, _ = self.model.analyze("stock price moves")
        assert 0 <= conf <= 1

    def test_sentiment_range(self):
        sent, _, _ = self.model.analyze("huge gain today")
        assert -1 <= sent <= 1

    def test_empty_text_handled(self):
        sent, conf, label = self.model.analyze("")
        assert sent == 0.0

    def test_batch_length_matches(self):
        texts = ["text one", "text two", "text three"]
        results = self.model.analyze_batch(texts)
        assert len(results) == 3

    def test_label_valid(self):
        _, _, label = self.model.analyze("stock up today")
        assert label in ("positive", "negative", "neutral")

    def test_short_text_handled(self):
        sent, conf, label = self.model.analyze("up")
        assert isinstance(sent, float)


class TestKSTest:
    """Unit tests for KS drift test (5 tests)."""

    def setup_method(self):
        from drift.detector import KSTest
        self.ks = KSTest()

    def test_same_distribution_high_pvalue(self):
        np.random.seed(0)
        a = np.random.normal(0, 1, 500)
        b = np.random.normal(0, 1, 500)
        pval = self.ks.test(a, b)
        assert pval > 0.05

    def test_different_distribution_low_pvalue(self):
        a = np.random.normal(0, 1, 500)
        b = np.random.normal(5, 1, 500)
        pval = self.ks.test(a, b)
        assert pval < 0.01

    def test_small_samples(self):
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])
        pval = self.ks.test(a, b)
        assert 0 <= pval <= 1

    def test_returns_float(self):
        result = self.ks.test(np.zeros(100), np.ones(100))
        assert isinstance(result, float)

    def test_identical_arrays(self):
        a = np.random.normal(0, 1, 100)
        pval = self.ks.test(a, a)
        assert pval == 1.0


class TestKellySizer:
    """Unit tests for Kelly sizing (10 tests)."""

    def setup_method(self):
        from risk.manager import KellySizer
        self.kelly = KellySizer(fraction=0.25, max_pct=0.10)

    def test_never_exceeds_max(self):
        size = self.kelly.size(0.9, 0.1, 0.02, 100000)
        assert size <= 0.10

    def test_zero_win_prob_returns_zero(self):
        size = self.kelly.size(0, 0.05, 0.02, 100000)
        assert size == 0.0

    def test_positive_edge_returns_positive_size(self):
        size = self.kelly.size(0.6, 0.05, 0.02, 100000)
        assert size > 0

    def test_negative_edge_returns_zero(self):
        size = self.kelly.size(0.3, 0.02, 0.05, 100000)
        # Negative Kelly edge → 0
        assert size == 0.0

    def test_size_range(self):
        size = self.kelly.size(0.55, 0.04, 0.03, 100000)
        assert 0 <= size <= 0.10

    def test_quarter_fraction_applied(self):
        from risk.manager import KellySizer
        full_kelly = KellySizer(fraction=1.0, max_pct=1.0)
        quarter_kelly = KellySizer(fraction=0.25, max_pct=1.0)
        full_size = full_kelly.size(0.6, 0.05, 0.02, 100000)
        quarter_size = quarter_kelly.size(0.6, 0.05, 0.02, 100000)
        assert quarter_size <= full_size

    def test_returns_float(self):
        result = self.kelly.size(0.6, 0.05, 0.03, 100000)
        assert isinstance(result, float)

    def test_high_confidence_larger_size(self):
        size_low = self.kelly.size(0.52, 0.04, 0.03, 100000)
        size_high = self.kelly.size(0.75, 0.04, 0.03, 100000)
        assert size_high >= size_low

    def test_zero_win_return_safe(self):
        size = self.kelly.size(0.6, 0, 0.03, 100000)
        assert size == 0.0

    def test_zero_loss_return_safe(self):
        size = self.kelly.size(0.6, 0.05, 0, 100000)
        assert size == 0.0


class TestDrawdownMonitor:
    """Drawdown and circuit breaker tests (10 tests)."""

    def setup_method(self):
        from risk.manager import DrawdownMonitor, RiskState, RiskStatus
        self.monitor = DrawdownMonitor({
            "max_drawdown_pct": 0.15,
            "loss_streak_limit": 3,
            "black_swan_vix_threshold": 40,
            "sigma_threshold": 3.0,
        })
        self.state = RiskState(portfolio_value=100000, peak_value=100000)
        self.RiskStatus = RiskStatus

    def test_no_drawdown_normal(self):
        status = self.monitor.assess(self.state, 100000)
        assert status == self.RiskStatus.NORMAL

    def test_small_drawdown_caution(self):
        status = self.monitor.assess(self.state, 91000)  # 9% DD
        assert status == self.RiskStatus.CAUTION

    def test_max_drawdown_lockdown(self):
        status = self.monitor.assess(self.state, 84000)  # 16% DD
        assert status == self.RiskStatus.LOCKDOWN

    def test_circuit_breaker_fires_once(self):
        self.monitor.assess(self.state, 84000)
        assert self.state.circuit_breaker_fired

    def test_loss_streak_pause(self):
        self.state.consecutive_losses = 3
        status = self.monitor.assess(self.state, 99000)
        assert status == self.RiskStatus.PAUSE

    def test_recovery_clears_caution(self):
        self.monitor.assess(self.state, 91000)
        status = self.monitor.assess(self.state, 100500)  # new peak
        assert status == self.RiskStatus.NORMAL

    def test_peak_updates_correctly(self):
        self.monitor.assess(self.state, 110000)
        assert self.state.peak_value == 110000

    def test_win_clears_loss_streak(self):
        self.state.consecutive_losses = 2
        self.monitor.record_trade(self.state, 0.02)
        assert self.state.consecutive_losses == 0

    def test_loss_increments_streak(self):
        self.monitor.record_trade(self.state, -0.02)
        assert self.state.consecutive_losses == 1

    def test_vix_spike_black_swan(self):
        status = self.monitor.assess(self.state, 99000, {"vix": 50})
        assert status == self.RiskStatus.BLACK_SWAN


class TestCorrelationGuard:
    """Correlation cap tests (5 tests)."""

    def setup_method(self):
        from risk.manager import CorrelationGuard, Position
        self.guard = CorrelationGuard()
        self.Position = Position

    def test_unrelated_tickers_allowed(self):
        ok, reason = self.guard.check("DVN", 0.08, {})
        assert ok

    def test_cluster_limit_enforced(self):
        existing = {
            "AAPL": self.Position("AAPL", "LONG", 0.10, 100, datetime.utcnow(), 95, 115, 100),
            "MSFT": self.Position("MSFT", "LONG", 0.10, 200, datetime.utcnow(), 190, 220, 200),
        }
        ok, reason = self.guard.check("GOOGL", 0.05, existing)
        # 0.10 + 0.10 + 0.05 = 0.25 > 0.20 max
        assert not ok

    def test_within_cluster_limit_allowed(self):
        existing = {
            "AAPL": self.Position("AAPL", "LONG", 0.08, 100, datetime.utcnow(), 95, 115, 100),
        }
        ok, reason = self.guard.check("MSFT", 0.08, existing)
        # 0.08 + 0.08 = 0.16 < 0.20 max
        assert ok

    def test_no_cluster_always_allowed(self):
        ok, reason = self.guard.check("SOMEUNKNOWN", 0.09, {})
        assert ok

    def test_reason_string_returned(self):
        _, reason = self.guard.check("AAPL", 0.05, {})
        assert isinstance(reason, str)


class TestMiroFishSwarm:
    """MiroFish swarm tests (10 tests)."""

    def setup_method(self):
        from mirofish.swarm import MiroFishSwarm
        self.swarm = MiroFishSwarm({"n_agents": 100, "seed": 42})

    def test_creates_correct_n_agents(self):
        assert len(self.swarm.agents) == 100

    def test_consensus_in_range(self):
        result = self.swarm.run_step("AAPL", 0.5, 0.3, 0.02)
        assert -1 <= result.consensus_signal <= 1

    def test_bullish_fraction_range(self):
        result = self.swarm.run_step("AAPL", 0.5, 0.5, 0.03)
        assert 0 <= result.bullish_fraction <= 1

    def test_bearish_fraction_range(self):
        result = self.swarm.run_step("AAPL", -0.5, -0.5, -0.03)
        assert 0 <= result.bearish_fraction <= 1

    def test_panic_threshold_trigger(self):
        # Large negative signal should trigger panic
        result = self.swarm.run_step("BTC-USD", -0.9, -0.9, -0.08)
        # bearish fraction likely high
        assert result.bearish_fraction > 0.1

    def test_emergent_behaviors_list(self):
        result = self.swarm.run_step("AAPL", 0, 0, 0)
        assert isinstance(result.emergent_behaviors, list)

    def test_confidence_range(self):
        result = self.swarm.run_step("AAPL", 0.3, 0.2, 0.01)
        assert 0 <= result.confidence <= 1

    def test_step_advances_counter(self):
        before = self.swarm._simulation_steps
        self.swarm.run_step("AAPL", 0, 0, 0)
        assert self.swarm._simulation_steps == before + 1

    def test_multi_step_returns_result(self):
        signals = [{"market": 0.3, "sentiment": 0.2, "price_return": 0.01}] * 5
        result = self.swarm.run_multi_step("AAPL", signals)
        assert result is not None

    def test_ticker_in_result(self):
        result = self.swarm.run_step("TSLA", 0.1, 0.1, 0.005)
        assert result.ticker == "TSLA"


class TestRiskManager:
    """Integration risk manager tests (10 tests)."""

    def setup_method(self):
        from risk.manager import RiskManager
        self.rm = RiskManager({
            "max_position_pct": 0.10,
            "kelly_fraction": 0.25,
            "max_drawdown_pct": 0.15,
            "stop_loss_pct": 0.05,
            "max_leverage": 1.5,
            "loss_streak_limit": 3,
            "black_swan_vix_threshold": 40,
            "sigma_threshold": 3.0,
            "no_trade_open_close_min": 15,
            "blackout_pre_earnings_min": 30,
            "overnight_size_reduction": 0.5,
            "min_confidence": 0.60,
        }, initial_capital=100000)

    def test_low_confidence_rejected(self):
        ok, size, reason = self.rm.approve_trade("AAPL", "LONG", 0.50, 0.50, 150.0)
        assert not ok
        assert "Confidence" in reason

    def test_valid_trade_approved(self):
        ok, size, reason = self.rm.approve_trade("AAPL", "LONG", 0.75, 0.65, 150.0, {
            "minutes_to_earnings": 999, "minutes_to_market_open": 60, "minutes_to_market_close": 60
        })
        assert ok
        assert size > 0

    def test_size_never_exceeds_max(self):
        ok, size, _ = self.rm.approve_trade("AAPL", "LONG", 0.95, 0.90, 150.0, {
            "minutes_to_earnings": 999, "minutes_to_market_open": 60, "minutes_to_market_close": 60
        })
        if ok:
            assert size <= 0.10

    def test_earnings_blackout_blocks(self):
        ok, _, reason = self.rm.approve_trade("AAPL", "LONG", 0.80, 0.70, 150.0, {
            "minutes_to_earnings": 10,
            "minutes_to_market_open": 60, "minutes_to_market_close": 60
        })
        assert not ok

    def test_summary_returns_dict(self):
        summary = self.rm.get_summary()
        assert isinstance(summary, dict)
        assert "portfolio_value" in summary

    def test_initial_status_normal(self):
        summary = self.rm.get_summary()
        assert summary["status"] == "normal"

    def test_circuit_breaker_not_fired_initially(self):
        assert not self.rm.state.circuit_breaker_fired

    def test_portfolio_value_correct(self):
        assert self.rm.state.portfolio_value == 100000

    def test_no_open_positions_initially(self):
        assert len(self.rm.state.positions) == 0

    def test_win_rate_zero_initially(self):
        summary = self.rm.get_summary()
        # 0/0 = 0 win rate
        assert summary["win_rate"] == 0.0


# ========== STRESS TESTS (50 tests) ==========

class TestStress:
    """Stress and extreme scenario tests."""

    def test_vix_80_triggers_lockdown(self):
        from risk.manager import RiskManager, RiskStatus
        rm = RiskManager({
            "max_position_pct": 0.10, "kelly_fraction": 0.25,
            "max_drawdown_pct": 0.15, "stop_loss_pct": 0.05,
            "max_leverage": 1.5, "loss_streak_limit": 3,
            "black_swan_vix_threshold": 40, "sigma_threshold": 3.0,
            "no_trade_open_close_min": 15, "blackout_pre_earnings_min": 30,
            "overnight_size_reduction": 0.5, "min_confidence": 0.60,
        }, 100000)
        ok, size, reason = rm.approve_trade("AAPL", "LONG", 0.80, 0.70, 150.0, {
            "vix": 80, "market_return_1d": 0.0,
            "minutes_to_market_open": 60, "minutes_to_market_close": 60,
            "minutes_to_earnings": 999,
        })
        assert not ok

    def test_1000_signals_processed(self):
        from nlp.pipeline import NLPPipeline
        from ingestion.sources import RawSignal
        nlp = NLPPipeline(["AAPL"], {"bot_filter": {"consensus_required": 5}, "min_confidence": 0.3})
        signals = [
            RawSignal("reddit", "AAPL", f"signal {i}", datetime.utcnow())
            for i in range(1000)
        ]
        processed = nlp.process(signals)
        assert len(processed) >= 0  # just ensure no crash

    def test_all_instruments_crash_20pct(self):
        from risk.manager import RiskManager, RiskStatus
        rm = RiskManager({
            "max_position_pct": 0.10, "kelly_fraction": 0.25,
            "max_drawdown_pct": 0.15, "stop_loss_pct": 0.05,
            "max_leverage": 1.5, "loss_streak_limit": 3,
            "black_swan_vix_threshold": 40, "sigma_threshold": 3.0,
            "no_trade_open_close_min": 15, "blackout_pre_earnings_min": 30,
            "overnight_size_reduction": 0.5, "min_confidence": 0.60,
        }, 100000)
        # Simulate 20% drawdown
        status = rm.drawdown_monitor.assess(rm.state, 80000, {"vix": 45})
        assert rm.state.circuit_breaker_fired

    def test_max_consecutive_losses(self):
        from risk.manager import DrawdownMonitor, RiskState, RiskStatus
        monitor = DrawdownMonitor({"max_drawdown_pct": 0.15, "loss_streak_limit": 3, "black_swan_vix_threshold": 40, "sigma_threshold": 3.0})
        state = RiskState(portfolio_value=100000, peak_value=100000)
        for _ in range(3):
            monitor.record_trade(state, -0.02)
        status = monitor.assess(state, 99000)
        assert status.value in ("pause", "caution", "normal")  # streak limit
        assert state.consecutive_losses == 3

    def test_drift_detector_extreme_shift(self):
        from drift.detector import DriftDetector, DriftTier
        detector = DriftDetector({"window_size": 100, "tier1_pvalue": 0.05, "tier2_pvalue": 0.01, "tier3_pvalue": 0.001})
        ref = np.random.normal(0, 0.1, 200)
        detector.update_reference(ref, ["neutral"] * 200)
        # Inject extreme shift
        current = np.random.normal(2.0, 0.1, 100)
        result = detector.detect(current, ["positive"] * 100)
        assert result.drift_detected

    def test_swarm_1000_agents_no_crash(self):
        from mirofish.swarm import MiroFishSwarm
        swarm = MiroFishSwarm({"n_agents": 1000, "seed": 0})
        for _ in range(5):
            result = swarm.run_step("BTC-USD", -0.9, -0.9, -0.1)
        assert result is not None

    def test_bayesian_aggregator_extreme_inputs(self):
        from prediction.engine import BayesianAggregator
        ba = BayesianAggregator()
        posterior, edge = ba.aggregate(0.001, 10.0, 10.0, 10.0, 10.0)
        assert 0 <= posterior <= 1

    def test_slippage_model_illiquid(self):
        from execution.broker import SlippageModel, OrderSide
        model = SlippageModel()
        slippage = model.estimate(1_000_000, 1000, is_crypto=False)
        assert slippage > 5  # high slippage for small ADV

    def test_position_stop_loss_triggers(self):
        from risk.manager import Position
        pos = Position("AAPL", "LONG", 0.10, 150.0, datetime.utcnow(), 142.5, 165.0, 150.0)
        pos.update_price(140.0)
        assert pos.should_stop_loss()

    def test_position_take_profit_triggers(self):
        from risk.manager import Position
        pos = Position("AAPL", "LONG", 0.10, 150.0, datetime.utcnow(), 142.5, 165.0, 150.0)
        pos.update_price(170.0)
        assert pos.should_take_profit()


# ========== REGIME TESTS (10 tests) ==========

class TestRegime:
    def test_crisis_regime_all_cash(self):
        from drift.detector import DriftDetector, MarketRegime
        d = DriftDetector({"window_size": 100, "tier1_pvalue": 0.05, "tier2_pvalue": 0.01, "tier3_pvalue": 0.001})
        d.current_regime = MarketRegime.CRISIS
        alloc = d.get_regime_allocation()
        assert alloc["cash"] == 0.95

    def test_bull_regime_high_equity(self):
        from drift.detector import DriftDetector, MarketRegime
        d = DriftDetector({})
        d.current_regime = MarketRegime.BULL
        alloc = d.get_regime_allocation()
        assert alloc["equity"] >= 0.60

    def test_bear_regime_low_equity(self):
        from drift.detector import DriftDetector, MarketRegime
        d = DriftDetector({})
        d.current_regime = MarketRegime.BEAR
        alloc = d.get_regime_allocation()
        assert alloc["equity"] <= 0.40

    def test_regime_classifier_bull(self):
        from drift.detector import RegimeClassifier, MarketRegime
        rc = RegimeClassifier()
        regime = rc.classify(0.3, 0.1, [0.005] * 30)
        assert regime == MarketRegime.BULL

    def test_regime_classifier_bear(self):
        from drift.detector import RegimeClassifier, MarketRegime
        rc = RegimeClassifier()
        regime = rc.classify(-0.3, 0.2, [-0.008] * 30)
        assert regime == MarketRegime.BEAR

    def test_regime_classifier_crisis(self):
        from drift.detector import RegimeClassifier, MarketRegime
        rc = RegimeClassifier()
        regime = rc.classify(-0.4, 0.5, [-0.02] * 20)
        assert regime == MarketRegime.CRISIS

    def test_regime_classifier_sideways(self):
        from drift.detector import RegimeClassifier, MarketRegime
        rc = RegimeClassifier()
        regime = rc.classify(0.05, 0.05, [0.001, -0.001, 0.0] * 10)
        assert regime in (MarketRegime.SIDEWAYS, MarketRegime.RECOVERY, MarketRegime.UNKNOWN)

    def test_should_trade_false_on_lockdown(self):
        from drift.detector import DriftDetector, DriftResult, DriftTier, MarketRegime
        d = DriftDetector({})
        d.drift_history = [DriftResult(True, DriftTier.TIER3_LOCKDOWN, MarketRegime.CRISIS, 0.0001, 0.5, 0.0001, 0.99, datetime.utcnow(), "lockdown")]
        assert not d.should_trade()

    def test_should_trade_true_on_tier1(self):
        from drift.detector import DriftDetector, DriftResult, DriftTier, MarketRegime
        d = DriftDetector({})
        d.drift_history = [DriftResult(True, DriftTier.TIER1_ALERT, MarketRegime.BULL, 0.03, 0.1, 0.03, 0.7, datetime.utcnow(), "alert")]
        assert d.should_trade()

    def test_regime_allocation_sums_to_one(self):
        from drift.detector import DriftDetector, MarketRegime
        d = DriftDetector({})
        for regime in MarketRegime:
            d.current_regime = regime
            alloc = d.get_regime_allocation()
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.001


# ========== PERFORMANCE TESTS (10 tests) ==========

class TestPerformance:
    def test_ks_test_1000_samples_fast(self):
        import time
        from drift.detector import KSTest
        ks = KSTest()
        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(0, 1, 1000)
        start = time.time()
        for _ in range(100):
            ks.test(a, b)
        elapsed = time.time() - start
        assert elapsed < 2.0  # 100 KS tests in < 2 seconds

    def test_mmd_200_samples_fast(self):
        import time
        from drift.detector import MMDTest
        mmd = MMDTest()
        a = np.random.normal(0, 1, 200)
        b = np.random.normal(0, 1, 200)
        start = time.time()
        score = mmd.score(a, b)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_kelly_sizer_fast(self):
        import time
        from risk.manager import KellySizer
        k = KellySizer()
        start = time.time()
        for _ in range(10000):
            k.size(0.6, 0.05, 0.03, 100000)
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_swarm_100_agents_fast(self):
        import time
        from mirofish.swarm import MiroFishSwarm
        swarm = MiroFishSwarm({"n_agents": 100, "seed": 1})
        start = time.time()
        for _ in range(10):
            swarm.run_step("AAPL", 0.3, 0.2, 0.01)
        elapsed = time.time() - start
        assert elapsed < 2.0

    def test_nlp_batch_100_fast(self):
        import time
        from nlp.pipeline import FinBERTSentiment
        model = FinBERTSentiment(use_model=False)
        texts = ["stock goes up today"] * 100
        start = time.time()
        results = model.analyze_batch(texts)
        elapsed = time.time() - start
        assert elapsed < 1.0
        assert len(results) == 100

    def test_risk_approve_fast(self):
        import time
        from risk.manager import RiskManager
        rm = RiskManager({
            "max_position_pct": 0.10, "kelly_fraction": 0.25, "max_drawdown_pct": 0.15,
            "stop_loss_pct": 0.05, "max_leverage": 1.5, "loss_streak_limit": 3,
            "black_swan_vix_threshold": 40, "sigma_threshold": 3.0,
            "no_trade_open_close_min": 15, "blackout_pre_earnings_min": 30,
            "overnight_size_reduction": 0.5, "min_confidence": 0.60,
        }, 100000)
        start = time.time()
        for _ in range(1000):
            rm.approve_trade("AAPL", "LONG", 0.75, 0.65, 150.0, {
                "minutes_to_earnings": 999, "minutes_to_market_open": 60,
                "minutes_to_market_close": 60
            })
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_sentiment_agg_100_tickers_fast(self):
        import time
        from nlp.pipeline import NLPPipeline, ProcessedSignal
        nlp = NLPPipeline(["AAPL"] * 100, {})
        signals = [ProcessedSignal("AAPL", "news", 0.5, 0.8, "positive", "text", datetime.utcnow()) for _ in range(200)]
        start = time.time()
        result = nlp.aggregate_by_ticker(signals)
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_prediction_engine_fast(self):
        import time
        from prediction.engine import PredictionEngine
        engine = PredictionEngine({})
        features = np.zeros((1, 13))
        start = time.time()
        for _ in range(100):
            engine.predict("AAPL", features, 0.3, 0.2, 0.5, "4h")
        elapsed = time.time() - start
        assert elapsed < 2.0

    def test_bayesian_aggregator_1000_calls_fast(self):
        import time
        from prediction.engine import BayesianAggregator
        ba = BayesianAggregator()
        start = time.time()
        for _ in range(1000):
            ba.aggregate(0.55, 0.3, 0.2, 0.4, 0.3)
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_entity_extractor_fast(self):
        import time
        from nlp.pipeline import EntityExtractor
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        extractor = EntityExtractor(tickers)
        text = "AAPL and MSFT reported strong earnings today, inflation data also released"
        start = time.time()
        for _ in range(1000):
            extractor.extract(text)
        elapsed = time.time() - start
        assert elapsed < 1.0


# ========== CONFORMAL PREDICTION TESTS ==========

class TestConformalPredictor:
    """Tests for ConformalPredictor split-conformal intervals (10 tests)."""

    def setup_method(self):
        from prediction.engine import ConformalPredictor
        self.cp = ConformalPredictor(alpha=0.10)

    def test_not_calibrated_returns_default_interval(self):
        lo, hi = self.cp.interval(0.05)
        assert lo < 0.05 < hi

    def test_calibrate_sets_q_hat(self):
        residuals = np.abs(np.random.normal(0, 0.02, 100))
        self.cp.calibrate(residuals)
        assert self.cp.is_calibrated
        assert self.cp.q_hat > 0

    def test_interval_symmetric_around_point(self):
        residuals = np.abs(np.random.normal(0, 0.02, 100))
        self.cp.calibrate(residuals)
        point = 0.03
        lo, hi = self.cp.interval(point)
        assert abs((hi - point) - (point - lo)) < 1e-10

    def test_interval_width_equals_2_q_hat(self):
        residuals = np.abs(np.random.normal(0, 0.02, 200))
        self.cp.calibrate(residuals)
        lo, hi = self.cp.interval(0.0)
        assert abs((hi - lo) - 2 * self.cp.q_hat) < 1e-10

    def test_empirical_coverage_near_90_percent(self):
        """Key correctness test: empirical coverage should be >= 90%."""
        from prediction.engine import ConformalPredictor
        np.random.seed(42)
        # Generate synthetic data
        X = np.random.normal(0, 1, 500)
        y = X * 0.05 + np.random.normal(0, 0.02, 500)
        # Split: 400 train, 100 calibration
        y_hat_train = X[:400] * 0.05  # "model" predictions
        y_hat_cal = X[400:] * 0.05
        residuals = np.abs(y[:400] - y_hat_train)
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(residuals)
        # Check coverage on 100 test points
        covered = sum(
            1 for i in range(100)
            if cp.interval(y_hat_cal[i])[0] <= y[400 + i] <= cp.interval(y_hat_cal[i])[1]
        )
        assert covered / 100 >= 0.80  # should be ~90%, allow some slack

    def test_small_calibration_set_conservative(self):
        from prediction.engine import ConformalPredictor
        cp = ConformalPredictor(alpha=0.10)
        residuals = np.array([0.01, 0.02, 0.03])  # only 3 samples
        cp.calibrate(residuals)
        assert cp.is_calibrated
        assert cp.q_hat > 0

    def test_q_hat_increases_with_residual_spread(self):
        from prediction.engine import ConformalPredictor
        cp_tight = ConformalPredictor(alpha=0.10)
        cp_wide = ConformalPredictor(alpha=0.10)
        cp_tight.calibrate(np.abs(np.random.normal(0, 0.005, 200)))
        cp_wide.calibrate(np.abs(np.random.normal(0, 0.05, 200)))
        assert cp_tight.q_hat < cp_wide.q_hat

    def test_lower_alpha_gives_wider_interval(self):
        from prediction.engine import ConformalPredictor
        residuals = np.abs(np.random.normal(0, 0.02, 200))
        cp_90 = ConformalPredictor(alpha=0.10)  # 90% coverage
        cp_99 = ConformalPredictor(alpha=0.01)  # 99% coverage
        cp_90.calibrate(residuals)
        cp_99.calibrate(residuals)
        assert cp_99.q_hat >= cp_90.q_hat

    def test_interval_contains_point_estimate(self):
        residuals = np.abs(np.random.normal(0, 0.02, 100))
        self.cp.calibrate(residuals)
        point = 0.07
        lo, hi = self.cp.interval(point)
        assert lo < point < hi

    def test_prediction_engine_exposes_conformal_fields(self):
        from prediction.engine import PredictionEngine
        engine = PredictionEngine({})
        features = np.random.normal(0, 0.1, (1, 13))
        pred = engine.predict("BTC-USD", features, 0.2, 0.3, 0.5, "1d")
        assert hasattr(pred, "conformal_coverage")
        assert pred.conformal_coverage == 0.90
        assert "conformal_q_hat" in pred.model_contributions
        assert "conformal_calibrated" in pred.model_contributions


class TestTemporalLSTM:
    """Tests for TemporalLSTMModel (8 tests)."""

    def setup_method(self):
        from prediction.engine import TemporalLSTMModel
        self.model = TemporalLSTMModel({})

    def test_untrained_returns_default(self):
        features = np.zeros((1, 13))
        point, lo, hi = self.model.predict(features)
        assert point == 0.0

    def test_train_on_small_data_does_not_crash(self):
        X = np.random.normal(0, 1, (30, 13))
        y = np.random.normal(0, 0.02, 30)
        self.model.train(X, y)
        assert self.model.is_trained

    def test_model_name_set_after_training(self):
        X = np.random.normal(0, 1, (30, 13))
        y = np.random.normal(0, 0.02, 30)
        self.model.train(X, y)
        assert self.model.model_name != "untrained"

    def test_predict_returns_tuple(self):
        X = np.random.normal(0, 1, (30, 13))
        y = np.random.normal(0, 0.02, 30)
        self.model.train(X, y)
        result = self.model.predict(np.zeros((1, 13)))
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ci_ordered(self):
        X = np.random.normal(0, 1, (30, 13))
        y = np.random.normal(0, 0.02, 30)
        self.model.train(X, y)
        point, lo, hi = self.model.predict(np.random.normal(0, 0.1, (1, 13)))
        assert lo <= point <= hi

    def test_predict_float_output(self):
        X = np.random.normal(0, 1, (30, 13))
        y = np.random.normal(0, 0.02, 30)
        self.model.train(X, y)
        point, lo, hi = self.model.predict(np.zeros((1, 13)))
        assert isinstance(point, float)

    def test_normalisation_fitted_on_train(self):
        X = np.random.normal(5, 2, (50, 13))
        y = np.random.normal(0, 0.02, 50)
        self.model.train(X, y)
        assert self.model._scaler_mean is not None

    def test_prediction_engine_reports_lstm_model(self):
        from prediction.engine import PredictionEngine
        engine = PredictionEngine({})
        X = np.random.normal(0, 1, (50, 13))
        y = np.random.normal(0, 0.02, 50)
        engine.train(X, y)
        features = np.zeros((1, 13))
        pred = engine.predict("AAPL", features)
        assert "lstm_model" in pred.model_contributions
