"""
QuantSwarm v3 — FRAMEWORM-SHIFT Regime Detection
3-tier drift response: alert → pause+retrain → full lockdown
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("quantswarm.drift")


class DriftTier(Enum):
    NONE = "none"
    TIER1_ALERT = "tier1_alert"          # p < 0.05 — send alert, keep trading
    TIER2_RETRAIN = "tier2_retrain"      # p < 0.01 — pause, retrain model
    TIER3_LOCKDOWN = "tier3_lockdown"    # p < 0.001 — full capital lockdown


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class DriftResult:
    drift_detected: bool
    tier: DriftTier
    regime: MarketRegime
    ks_pvalue: float
    mmd_score: float
    chi2_pvalue: float
    confidence: float
    timestamp: datetime
    description: str


class KSTest:
    """Kolmogorov-Smirnov test for distribution shift."""

    def test(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Returns p-value. Low p-value = distributions differ significantly."""
        try:
            from scipy.stats import ks_2samp
            _, pvalue = ks_2samp(reference, current)
            return float(pvalue)
        except Exception:
            return 1.0  # fail safe: no drift detected


class MMDTest:
    """Maximum Mean Discrepancy — more sensitive to subtle distribution shifts."""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> float:
        """RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)"""
        diff = X[:, None] - Y[None, :]
        sq_dist = np.sum(diff ** 2, axis=-1) if X.ndim > 1 else diff ** 2
        return np.exp(-self.gamma * sq_dist)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns MMD score. Higher = more different.
        Threshold: > 0.1 = potential drift, > 0.3 = strong drift.
        """
        try:
            n, m = len(X), len(Y)
            if n < 10 or m < 10:
                return 0.0
            # Subsample for speed
            max_samples = 200
            if n > max_samples:
                X = X[np.random.choice(n, max_samples, replace=False)]
            if m > max_samples:
                Y = Y[np.random.choice(m, max_samples, replace=False)]
            X_flat = X.flatten() if X.ndim > 1 else X
            Y_flat = Y.flatten() if Y.ndim > 1 else Y
            kxx = self._rbf_kernel(X_flat, X_flat).mean()
            kyy = self._rbf_kernel(Y_flat, Y_flat).mean()
            kxy = self._rbf_kernel(X_flat, Y_flat).mean()
            mmd = float(kxx + kyy - 2 * kxy)
            return max(0.0, mmd)
        except Exception:
            return 0.0


class Chi2Test:
    """Chi-squared test for categorical distribution shift (sentiment labels)."""

    def test(self, reference_labels: List[str], current_labels: List[str]) -> float:
        """Returns p-value for label distribution shift."""
        try:
            from scipy.stats import chi2_contingency
            categories = list(set(reference_labels + current_labels))
            ref_counts = [reference_labels.count(c) for c in categories]
            cur_counts = [current_labels.count(c) for c in categories]
            # Need at least some counts
            if sum(ref_counts) < 5 or sum(cur_counts) < 5:
                return 1.0
            contingency = np.array([ref_counts, cur_counts])
            _, pvalue, _, _ = chi2_contingency(contingency)
            return float(pvalue)
        except Exception:
            return 1.0


class RegimeClassifier:
    """Classify current market regime from sentiment + price data."""

    def classify(
        self,
        sentiment_mean: float,
        sentiment_std: float,
        price_returns: List[float],
        vix_proxy: float = 0.0,
    ) -> MarketRegime:
        """Simple regime classification from available signals."""
        if not price_returns:
            return MarketRegime.UNKNOWN

        recent_return = np.mean(price_returns[-20:]) if len(price_returns) >= 20 else np.mean(price_returns)
        recent_vol = np.std(price_returns[-20:]) * np.sqrt(252) if len(price_returns) >= 20 else 0.0

        # Crisis: extreme sustained negative returns (>1.5% daily avg)
        # OR extreme negative + elevated vol. A -2% daily avg is crisis
        # regardless of whether volatility appears artificially low.
        if recent_return < -0.015 and (recent_vol > 0.3 or abs(recent_return) > 0.018):
            return MarketRegime.CRISIS
        # Bear: negative returns, negative sentiment
        if recent_return < -0.005 and sentiment_mean < -0.2:
            return MarketRegime.BEAR
        # Bull: positive returns, positive sentiment
        if recent_return > 0.003 and sentiment_mean > 0.1:
            return MarketRegime.BULL
        # Recovery: improving after recent bear
        if recent_return > 0.001 and sentiment_mean > -0.1 and sentiment_mean < 0.1:
            return MarketRegime.RECOVERY
        # Sideways: low vol, mixed sentiment
        if recent_vol < 0.15 and abs(sentiment_mean) < 0.15:
            return MarketRegime.SIDEWAYS

        return MarketRegime.UNKNOWN


class DriftDetector:
    """
    FRAMEWORM-SHIFT v2 — 3-tier drift response.
    Combines KS + MMD + Chi2 tests with regime classification.
    """

    def __init__(self, config: dict):
        self.window_size = config.get("window_size", 500)
        self.tier1_pvalue = config.get("tier1_pvalue", 0.05)
        self.tier2_pvalue = config.get("tier2_pvalue", 0.01)
        self.tier3_pvalue = config.get("tier3_pvalue", 0.001)
        self.ks_test = KSTest()
        self.mmd_test = MMDTest()
        self.chi2_test = Chi2Test()
        self.regime_classifier = RegimeClassifier()
        self.reference_window: Optional[np.ndarray] = None
        self.reference_labels: List[str] = []
        self.current_regime = MarketRegime.UNKNOWN
        self.drift_history: List[DriftResult] = []

    def update_reference(self, sentiment_scores: np.ndarray, labels: List[str]):
        """Set or update the reference distribution window."""
        self.reference_window = sentiment_scores[-self.window_size:]
        self.reference_labels = labels[-self.window_size:]

    def detect(
        self,
        current_sentiments: np.ndarray,
        current_labels: List[str],
        price_returns: List[float] = None,
    ) -> DriftResult:
        """
        Run drift detection. Returns DriftResult with tier and regime.
        """
        timestamp = datetime.utcnow()
        price_returns = price_returns or []

        # Need reference window
        if self.reference_window is None or len(self.reference_window) < 50:
            self.update_reference(current_sentiments, current_labels)
            return DriftResult(
                drift_detected=False, tier=DriftTier.NONE,
                regime=MarketRegime.UNKNOWN,
                ks_pvalue=1.0, mmd_score=0.0, chi2_pvalue=1.0,
                confidence=0.0, timestamp=timestamp,
                description="Initializing reference window"
            )

        # Run all three tests
        ks_pval = self.ks_test.test(self.reference_window, current_sentiments)
        mmd_score = self.mmd_test.score(self.reference_window, current_sentiments)
        chi2_pval = self.chi2_test.test(self.reference_labels, current_labels)

        # Regime classification
        sent_mean = float(np.mean(current_sentiments)) if len(current_sentiments) > 0 else 0.0
        sent_std = float(np.std(current_sentiments)) if len(current_sentiments) > 0 else 0.0
        regime = self.regime_classifier.classify(sent_mean, sent_std, price_returns)
        self.current_regime = regime

        # Determine drift tier
        # Use minimum p-value as the combined signal
        min_pval = min(ks_pval, chi2_pval)
        # MMD boost: if MMD > 0.2, reduce effective p-value
        if mmd_score > 0.2:
            min_pval = min(min_pval, 0.05)
        if mmd_score > 0.4:
            min_pval = min(min_pval, 0.005)

        if min_pval < self.tier3_pvalue or regime == MarketRegime.CRISIS:
            tier = DriftTier.TIER3_LOCKDOWN
            drift_detected = True
            desc = f"CRITICAL DRIFT: regime={regime.value}, ks_p={ks_pval:.4f}, mmd={mmd_score:.3f}"
        elif min_pval < self.tier2_pvalue:
            tier = DriftTier.TIER2_RETRAIN
            drift_detected = True
            desc = f"Significant drift detected — retraining required. ks_p={ks_pval:.4f}"
        elif min_pval < self.tier1_pvalue:
            tier = DriftTier.TIER1_ALERT
            drift_detected = True
            desc = f"Mild drift detected — monitoring closely. ks_p={ks_pval:.4f}"
        else:
            tier = DriftTier.NONE
            drift_detected = False
            desc = f"No drift. regime={regime.value}, ks_p={ks_pval:.4f}"

        confidence = 1.0 - min_pval

        result = DriftResult(
            drift_detected=drift_detected,
            tier=tier,
            regime=regime,
            ks_pvalue=ks_pval,
            mmd_score=mmd_score,
            chi2_pvalue=chi2_pval,
            confidence=confidence,
            timestamp=timestamp,
            description=desc,
        )

        self.drift_history.append(result)
        if len(self.drift_history) > 1000:
            self.drift_history = self.drift_history[-500:]

        logger.info(f"Drift: tier={tier.value}, regime={regime.value}, ks_p={ks_pval:.4f}, mmd={mmd_score:.3f}")
        return result

    def should_trade(self) -> bool:
        """Return False if current drift tier blocks trading."""
        if not self.drift_history:
            return True
        last = self.drift_history[-1]
        return last.tier not in (DriftTier.TIER2_RETRAIN, DriftTier.TIER3_LOCKDOWN)

    def should_alert(self) -> bool:
        if not self.drift_history:
            return False
        return self.drift_history[-1].tier != DriftTier.NONE

    def get_regime_allocation(self) -> dict:
        """
        Return suggested asset allocation based on current regime.
        Used by the risk manager to adjust portfolio weights.
        """
        allocations = {
            MarketRegime.BULL: {"equity": 0.70, "crypto": 0.20, "cash": 0.10},
            MarketRegime.BEAR: {"equity": 0.30, "crypto": 0.05, "cash": 0.65},
            MarketRegime.SIDEWAYS: {"equity": 0.50, "crypto": 0.10, "cash": 0.40},
            MarketRegime.CRISIS: {"equity": 0.05, "crypto": 0.00, "cash": 0.95},
            MarketRegime.RECOVERY: {"equity": 0.55, "crypto": 0.15, "cash": 0.30},
            MarketRegime.UNKNOWN: {"equity": 0.40, "crypto": 0.10, "cash": 0.50},
        }
        return allocations.get(self.current_regime, allocations[MarketRegime.UNKNOWN])
