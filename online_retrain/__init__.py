"""
QuantSwarm — Online Retrainer
Watches drift signals and triggers incremental model retraining
so the prediction engine stays calibrated to current market regimes.
"""
from __future__ import annotations
import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("quantswarm.retrain")

RETRAIN_DIR = Path("models/retrain")
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)


class RetrainTrigger:
    """Decides when retraining is needed."""

    def __init__(self, min_interval_hours: int = 6, drift_score_threshold: float = 0.6):
        self.min_interval = timedelta(hours=min_interval_hours)
        self.drift_threshold = drift_score_threshold
        self.last_retrain: Optional[datetime] = None
        self.retrain_count = 0

    def should_retrain(self, drift_score: float, force: bool = False) -> bool:
        if force:
            return True
        now = datetime.utcnow()
        if self.last_retrain and (now - self.last_retrain) < self.min_interval:
            return False
        return drift_score >= self.drift_threshold

    def mark_retrained(self):
        self.last_retrain = datetime.utcnow()
        self.retrain_count += 1
        logger.info(f"Retraining recorded — count={self.retrain_count}, time={self.last_retrain}")


class IncrementalBuffer:
    """
    Rolling buffer of recent (X, y) pairs for incremental training.
    Keeps at most `max_size` samples; oldest are dropped first.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def add(self, x: np.ndarray, label: float):
        self.X.append(x)
        self.y.append(label)
        if len(self.X) > self.max_size:
            self.X.pop(0)
            self.y.pop(0)

    @property
    def size(self) -> int:
        return len(self.X)

    def as_arrays(self):
        return np.array(self.X), np.array(self.y)

    def save(self, path: Path = RETRAIN_DIR / "buffer.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"X": self.X, "y": self.y}, f)
        logger.info(f"Buffer saved: {len(self.X)} samples → {path}")

    def load(self, path: Path = RETRAIN_DIR / "buffer.pkl"):
        if not path.exists():
            logger.info("No existing buffer found, starting fresh.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.X = data.get("X", [])
        self.y = data.get("y", [])
        logger.info(f"Buffer loaded: {len(self.X)} samples from {path}")


class OnlineRetrainer:
    """
    Runs in the background. Monitors drift score → triggers
    XGBoost incremental retraining whenever regime shifts.
    """

    def __init__(self, prediction_engine, drift_detector,
                 min_samples: int = 500, min_interval_hours: int = 6):
        self.engine = prediction_engine
        self.drift = drift_detector
        self.min_samples = min_samples
        self.trigger = RetrainTrigger(min_interval_hours=min_interval_hours)
        self.buffer = IncrementalBuffer(max_size=20_000)
        self.buffer.load()
        self._running = False
        self.retrain_log: List[Dict] = []

    def add_sample(self, features: np.ndarray, outcome: float):
        """Call after each trade resolves with actual outcome (+1 win / -1 loss)."""
        self.buffer.add(features, outcome)

    async def monitor(self, check_interval_sec: int = 300):
        """Background loop — checks every 5 min, retrains when needed."""
        self._running = True
        logger.info("OnlineRetrainer started")
        while self._running:
            try:
                drift_result = self.drift.get_latest_result()
                drift_score = getattr(drift_result, "composite_score", 0.0)

                if (self.buffer.size >= self.min_samples
                        and self.trigger.should_retrain(drift_score)):
                    await self._do_retrain(drift_score)

            except Exception as e:
                logger.error(f"Retrainer monitor error: {e}")

            await asyncio.sleep(check_interval_sec)

    async def _do_retrain(self, drift_score: float):
        logger.info(f"⚙️  Retraining triggered — drift_score={drift_score:.3f}, "
                    f"buffer_size={self.buffer.size}")
        X, y = self.buffer.as_arrays()

        try:
            # Incremental XGBoost refit using the rolling buffer
            from xgboost import XGBClassifier
            import numpy as np

            y_binary = (y > 0).astype(int)
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            # Time-series split: last 20% for validation
            split = int(len(X) * 0.8)
            model.fit(
                X[:split], y_binary[:split],
                eval_set=[(X[split:], y_binary[split:])],
                early_stopping_rounds=20,
                verbose=False,
            )
            val_acc = float(np.mean(model.predict(X[split:]) == y_binary[split:]))
            logger.info(f"Retrain complete — val_accuracy={val_acc:.3f}")

            # Hot-swap the model in prediction engine if accuracy improved
            old_acc = getattr(self.engine, "_xgb_val_acc", 0.0)
            if val_acc >= old_acc - 0.02:  # allow 2% tolerance
                self.engine.xgb_model = model
                self.engine._xgb_val_acc = val_acc
                logger.info("Model hot-swapped into PredictionEngine ✓")
            else:
                logger.warning(f"New model worse ({val_acc:.3f} vs {old_acc:.3f}), keeping old.")

            # Save checkpoint
            ckpt = RETRAIN_DIR / f"xgb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(ckpt, "wb") as f:
                pickle.dump(model, f)

            self.trigger.mark_retrained()
            self.buffer.save()
            self.retrain_log.append({
                "time": datetime.utcnow().isoformat(),
                "drift_score": drift_score,
                "buffer_size": self.buffer.size,
                "val_acc": val_acc,
                "hot_swapped": val_acc >= old_acc - 0.02,
            })

        except Exception as e:
            logger.error(f"Retrain failed: {e}")

    def stop(self):
        self._running = False
        logger.info("OnlineRetrainer stopped")

    def status(self) -> Dict:
        return {
            "buffer_size": self.buffer.size,
            "retrain_count": self.trigger.retrain_count,
            "last_retrain": self.trigger.last_retrain.isoformat() if self.trigger.last_retrain else None,
            "retrain_log": self.retrain_log[-10:],
        }
