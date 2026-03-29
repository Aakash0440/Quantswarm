"""
QuantSwarm v3 — Explainability Layer
SHAP-based per-prediction attribution. No black boxes.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("quantswarm.explain")


@dataclass
class Explanation:
    ticker: str
    direction: str
    magnitude: float
    top_drivers: List[dict]      # [{feature, value, contribution, pct}]
    human_readable: str          # Plain English explanation
    confidence: float


class SHAPExplainer:
    """
    SHAP values for XGBoost model + attribution for all models.
    Falls back to feature importance if SHAP unavailable.
    """

    FEATURE_LABELS = {
        "sentiment_weighted": "Multi-source sentiment",
        "sentiment_momentum": "Sentiment momentum (delta)",
        "sec_insider_signal": "SEC insider trading signal",
        "news_count_24h": "News volume (24h)",
        "reddit_sentiment": "Reddit sentiment",
        "price_return_1h": "Price return (1h)",
        "price_return_24h": "Price return (24h)",
        "price_return_7d": "Price return (7d)",
        "volume_ratio": "Volume vs avg",
        "volatility_7d": "7d realized volatility",
        "funding_rate": "Funding rate",
        "drift_score": "Regime drift score",
        "mirofish_consensus": "MiroFish swarm consensus",
    }

    def __init__(self, model=None, background_samples: int = 100):
        self.model = model
        self.background_samples = background_samples
        self._shap_available = False
        self._explainer = None
        if model is not None:
            self._try_init_shap(model)

    def _try_init_shap(self, model):
        try:
            import shap
            if hasattr(model, 'predict'):
                self._explainer = shap.TreeExplainer(model)
                self._shap_available = True
                logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.info(f"SHAP unavailable: {e} — using feature importance fallback")

    def explain(self, prediction, feature_values: dict) -> Explanation:
        """
        Generate explanation for a prediction.
        Returns Explanation with top drivers and human-readable text.
        """
        feature_names = list(self.FEATURE_LABELS.keys())
        feat_array = np.array([feature_values.get(f, 0.0) for f in feature_names])

        # Get contributions
        if self._shap_available and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(feat_array.reshape(1, -1))
                contributions = dict(zip(feature_names, shap_values[0]))
            except Exception:
                contributions = self._fallback_contributions(feat_array, feature_names, prediction)
        else:
            contributions = self._fallback_contributions(feat_array, feature_names, prediction)

        # Also add model-level contributions
        model_contribs = prediction.model_contributions
        if model_contribs:
            contributions["_tft_signal"] = model_contribs.get("tft", 0) * 0.5
            contributions["_xgb_signal"] = model_contribs.get("xgb", 0) * 0.3
            contributions["_mirofish"] = model_contribs.get("mirofish", 0) * 0.2

        # Sort by absolute contribution
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(v) for _, v in sorted_contribs) or 1.0

        top_drivers = []
        for feat, contrib in sorted_contribs[:5]:
            label = self.FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            top_drivers.append({
                "feature": feat,
                "label": label,
                "value": round(float(feature_values.get(feat, 0.0)), 4),
                "contribution": round(float(contrib), 4),
                "pct": round(abs(contrib) / total_abs * 100, 1),
                "direction": "bullish" if contrib > 0 else "bearish",
            })

        human = self._generate_human_text(prediction, top_drivers, model_contribs)

        return Explanation(
            ticker=prediction.ticker,
            direction=prediction.direction,
            magnitude=prediction.magnitude,
            top_drivers=top_drivers,
            human_readable=human,
            confidence=prediction.confidence,
        )

    def _fallback_contributions(
        self, feat_array: np.ndarray, feature_names: List[str], prediction
    ) -> dict:
        """
        Attribution without SHAP: use feature values * known weights as proxy.
        """
        # Approximate weights based on domain knowledge
        domain_weights = {
            "sec_insider_signal": 0.25,
            "sentiment_weighted": 0.20,
            "mirofish_consensus": 0.18,
            "price_return_24h": 0.15,
            "news_count_24h": 0.08,
            "reddit_sentiment": 0.07,
            "volume_ratio": 0.07,
            "price_return_1h": 0.05,
            "drift_score": 0.05,
            "funding_rate": 0.04,
            "volatility_7d": 0.03,
            "price_return_7d": 0.02,
            "sentiment_momentum": 0.02,
        }
        contribs = {}
        for i, fname in enumerate(feature_names):
            w = domain_weights.get(fname, 0.05)
            contribs[fname] = float(feat_array[i]) * w
        return contribs

    def _generate_human_text(self, prediction, top_drivers: List[dict], model_contribs: dict) -> str:
        """Generate plain-English explanation of the prediction."""
        lines = []
        edge = model_contribs.get("edge", 0)
        posterior = model_contribs.get("bayesian_posterior", 0.5)

        lines.append(
            f"Predicted {prediction.direction} {abs(prediction.magnitude)*100:.1f}% on {prediction.ticker} "
            f"({prediction.horizon} horizon) — confidence {prediction.confidence*100:.0f}%"
        )
        lines.append(f"Bayesian posterior: {posterior*100:.1f}% probability UP (edge: {edge*100:+.1f}%)")
        lines.append("\nTop drivers:")

        for d in top_drivers[:4]:
            direction_emoji = "↑" if d["direction"] == "bullish" else "↓"
            lines.append(
                f"  {direction_emoji} {d['label']}: {d['pct']:.0f}% weight "
                f"(value={d['value']:.3f}, contribution={d['contribution']:+.4f})"
            )

        # Filtered signals note
        lines.append(f"\nBot-filtered signals excluded before analysis.")
        if model_contribs.get("tft"):
            lines.append(
                f"Model ensemble: TFT={model_contribs['tft']:+.3f}, "
                f"XGB={model_contribs.get('xgb', 0):+.3f}, "
                f"Swarm={model_contribs.get('mirofish', 0):+.3f}"
            )

        return "\n".join(lines)
