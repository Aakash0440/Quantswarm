"""
QuantSwarm v3 — NLP Signal Processing Layer
Bot filter + FinBERT sentiment + entity extraction.
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("quantswarm.nlp")


@dataclass
class ProcessedSignal:
    ticker: str
    source: str
    sentiment: float          # -1.0 to +1.0
    confidence: float         # 0.0 to 1.0
    sentiment_label: str      # positive | negative | neutral
    text_preview: str
    timestamp: datetime
    is_bot: bool = False
    bot_score: float = 0.0    # 0=clean, 1=definitely bot
    entities: list = None
    raw_signal_id: str = ""

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


class BotFilter:
    """
    5-signal consensus bot filter.
    Detects manufactured/coordinated sentiment before it reaches the NLP layer.
    """

    def __init__(self, config: dict):
        self.min_account_age = config.get("min_account_age_days", 30)
        self.max_tweets_per_day = config.get("max_tweets_per_day", 100)
        self.min_follower_ratio = config.get("min_follower_ratio", 0.05)
        self.cosine_threshold = config.get("cosine_similarity_threshold", 0.88)
        self.consensus_required = config.get("consensus_required", 5)
        self._recent_texts: List[str] = []

    def _account_age_score(self, meta: dict) -> float:
        """Young accounts = higher bot probability."""
        age = meta.get("account_age_days", 999)
        if age < 7:
            return 1.0
        if age < self.min_account_age:
            return 0.6
        return 0.0

    def _follower_ratio_score(self, meta: dict) -> float:
        """Low follower/following ratio = bot signal."""
        followers = meta.get("followers", 100)
        following = meta.get("following", 100)
        if following == 0:
            return 0.0
        ratio = followers / following
        if ratio < self.min_follower_ratio:
            return 0.8
        if ratio < 0.1:
            return 0.4
        return 0.0

    def _posting_frequency_score(self, meta: dict) -> float:
        """Excessive posting = bot signal."""
        daily_posts = meta.get("posts_per_day", 5)
        if daily_posts > self.max_tweets_per_day:
            return 1.0
        if daily_posts > 50:
            return 0.5
        return 0.0

    def _text_similarity_score(self, text: str) -> float:
        """Identical/near-identical text across accounts = coordinated."""
        if not self._recent_texts:
            self._recent_texts.append(text)
            return 0.0

        # Simple word overlap similarity
        words = set(text.lower().split())
        for prev_text in self._recent_texts[-50:]:
            prev_words = set(prev_text.lower().split())
            if not prev_words or not words:
                continue
            overlap = len(words & prev_words) / max(len(words | prev_words), 1)
            if overlap > self.cosine_threshold:
                return 0.9

        self._recent_texts.append(text)
        if len(self._recent_texts) > 200:
            self._recent_texts = self._recent_texts[-100:]
        return 0.0

    def _financial_spam_score(self, text: str) -> float:
        """Detect pump & dump language patterns."""
        spam_patterns = [
            r'\b(100x|moon|lambo|wen\s+moon|to\s+the\s+moon|diamond\s+hands|hodl)\b',
            r'\b(guaranteed\s+profit|can\'t\s+lose|easy\s+money|insider\s+tip)\b',
            r'[A-Z]{3,}\s*🚀{2,}',
            r'(buy\s+now|last\s+chance|limited\s+time)',
        ]
        score = 0.0
        for pattern in spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25
        return min(score, 1.0)

    def score(self, text: str, author_meta: dict) -> Tuple[bool, float]:
        """
        Returns (is_bot, bot_probability_0_to_1).

        Formula: 0.4 * mean + 0.6 * max of the 5 sub-signals.
        This ensures a single very strong signal (e.g. 3-day-old account,
        duplicate text) registers clearly above 0.5 rather than being
        diluted by the 4 other clean signals.
        """
        scores = [
            self._account_age_score(author_meta),
            self._follower_ratio_score(author_meta),
            self._posting_frequency_score(author_meta),
            self._text_similarity_score(text),
            self._financial_spam_score(text),
        ]
        mean_score = float(np.mean(scores))
        max_score  = float(np.max(scores))
        bot_prob   = 0.4 * mean_score + 0.6 * max_score
        n_flagged  = sum(1 for s in scores if s > 0.5)
        is_bot     = n_flagged >= self.consensus_required or bot_prob > 0.75
        return is_bot, float(np.clip(bot_prob, 0.0, 1.0))


class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis for financial text.
    Falls back to lexicon-based if model unavailable.
    """

    FINBERT_MODEL = "ProsusAI/finbert"
    POSITIVE_WORDS = set(["beat", "surge", "record", "growth", "profit", "strong", "bull", "up", "gain", "positive"])
    NEGATIVE_WORDS = set(["miss", "crash", "loss", "weak", "bear", "down", "decline", "risk", "default", "fail"])

    def __init__(self, use_model: bool = True):
        self.pipe = None
        if use_model:
            try:
                from transformers import pipeline
                self.pipe = pipeline(
                    "text-classification",
                    model=self.FINBERT_MODEL,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT loaded successfully")
            except Exception as e:
                logger.warning(f"FinBERT load failed, using lexicon fallback: {e}")

    def _lexicon_fallback(self, text: str) -> Tuple[float, float, str]:
        """Simple lexicon-based fallback."""
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        total = max(pos + neg, 1)
        if pos > neg:
            score = pos / total
            return score, min(score + 0.1, 1.0), "positive"
        elif neg > pos:
            score = -(neg / total)
            return score, min(abs(score) + 0.1, 1.0), "negative"
        return 0.0, 0.5, "neutral"

    def analyze(self, text: str) -> Tuple[float, float, str]:
        """
        Returns (sentiment_score, confidence, label).
        sentiment_score: -1.0 (very negative) to +1.0 (very positive)
        """
        if not text or len(text.strip()) < 10:
            return 0.0, 0.3, "neutral"

        if self.pipe:
            try:
                result = self.pipe(text[:512])[0]
                scores = {r["label"].lower(): r["score"] for r in result}
                pos = scores.get("positive", 0)
                neg = scores.get("negative", 0)
                neu = scores.get("neutral", 0)
                # Convert to -1..+1
                sentiment = pos - neg
                label = max(scores, key=scores.get)
                confidence = max(scores.values())
                return float(sentiment), float(confidence), label
            except Exception as e:
                logger.debug(f"FinBERT inference error: {e}")

        return self._lexicon_fallback(text)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float, str]]:
        """Batch analysis for efficiency."""
        if self.pipe:
            try:
                # Process in chunks of 32
                results = []
                for i in range(0, len(texts), 32):
                    batch = texts[i:i+32]
                    batch_results = self.pipe(
                        [t[:512] for t in batch],
                        batch_size=len(batch),
                    )
                    for res in batch_results:
                        scores = {r["label"].lower(): r["score"] for r in res}
                        pos = scores.get("positive", 0)
                        neg = scores.get("negative", 0)
                        sentiment = pos - neg
                        label = max(scores, key=scores.get)
                        confidence = max(scores.values())
                        results.append((float(sentiment), float(confidence), label))
                return results
            except Exception as e:
                logger.debug(f"FinBERT batch error: {e}")

        return [self._lexicon_fallback(t) for t in texts]


class EntityExtractor:
    """Extract ticker symbols, executive names, macro events from text."""

    MACRO_EVENTS = {
        "fed rate": "macro_fed",
        "inflation": "macro_inflation",
        "recession": "macro_recession",
        "earnings": "event_earnings",
        "ipo": "event_ipo",
        "merger": "event_ma",
        "acquisition": "event_ma",
        "buyback": "event_buyback",
        "dividend": "event_dividend",
        "insider": "event_insider",
        "sec filing": "event_sec",
        "short squeeze": "event_short_squeeze",
    }

    def __init__(self, known_tickers: List[str]):
        self.known_tickers = set(t.replace("-USD", "").upper() for t in known_tickers)

    def extract(self, text: str) -> dict:
        """Extract structured entities from text."""
        upper = text.upper()
        entities = {
            "tickers": [],
            "events": [],
        }
        # Ticker detection: $AAPL or standalone AAPL
        for ticker in self.known_tickers:
            if f"${ticker}" in upper or re.search(rf'\b{ticker}\b', upper):
                entities["tickers"].append(ticker)

        # Event detection
        lower = text.lower()
        for keyword, event_type in self.MACRO_EVENTS.items():
            if keyword in lower:
                entities["events"].append(event_type)

        return entities


class NLPPipeline:
    """
    Full NLP pipeline:
    RawSignal -> bot filter -> FinBERT -> entity extraction -> ProcessedSignal
    """

    def __init__(self, tickers: List[str], config: dict):
        self.bot_filter = BotFilter(config.get("bot_filter", {}))
        self.finbert = FinBERTSentiment(use_model=config.get("use_finbert_model", True))
        self.entity_extractor = EntityExtractor(tickers)
        self.min_confidence = config.get("min_confidence", 0.65)

    def process(self, raw_signals: list) -> List[ProcessedSignal]:
        """Process a batch of RawSignals."""
        processed = []
        texts = [s.text for s in raw_signals]

        # Batch sentiment analysis
        sentiments = self.finbert.analyze_batch(texts)

        for raw, (sentiment, confidence, label) in zip(raw_signals, sentiments):
            # Trusted machine sources (on-chain, market data, SEC) bypass bot filter
            # They are structured data feeds, not social-media authors
            TRUSTED_SOURCES = {"onchain", "market", "sec"}
            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
                confidence = 1.0
            else:
                is_bot, bot_score = self.bot_filter.score(raw.text, raw.author_meta)

            # Skip confirmed bots
            if is_bot:
                logger.debug(f"Bot filtered: {raw.author} ({bot_score:.2f})")
                continue

            # Trusted sources skip confidence floor and get boosted to 1.0
            if raw.source in TRUSTED_SOURCES:
                confidence = 1.0
                label = "neutral"
            elif confidence < self.min_confidence:
                continue
            if raw.source in TRUSTED_SOURCES:
                confidence = 1.0
                label = "neutral"

            # Entity extraction
            entities = self.entity_extractor.extract(raw.text)

            processed.append(ProcessedSignal(
                ticker=raw.ticker,
                source=raw.source,
                sentiment=sentiment,
                confidence=confidence,
                sentiment_label=label,
                text_preview=raw.text[:200],
                timestamp=raw.timestamp,
                is_bot=is_bot,
                bot_score=bot_score,
                entities=entities,
                raw_signal_id=raw.signal_id,
            ))

        bot_count = len(raw_signals) - len(processed)
        logger.info(f"NLP: {len(raw_signals)} raw -> {len(processed)} clean ({bot_count} bots filtered)")
        return processed

    def aggregate_by_ticker(self, signals: List[ProcessedSignal]) -> dict:
        """
        Aggregate signals per ticker -> weighted sentiment score.
        Source weights: sec > news > reddit > twitter
        """
        source_weights = {"sec": 1.5, "news": 1.2, "market": 1.0, "reddit": 0.8, "twitter": 0.6, "onchain": 1.0}
        ticker_scores = {}

        for sig in signals:
            t = sig.ticker
            if t not in ticker_scores:
                ticker_scores[t] = {"weighted_sum": 0.0, "weight_total": 0.0, "count": 0, "signals": []}
            w = source_weights.get(sig.source, 0.7) * sig.confidence
            ticker_scores[t]["weighted_sum"] += sig.sentiment * w
            ticker_scores[t]["weight_total"] += w
            ticker_scores[t]["count"] += 1
            ticker_scores[t]["signals"].append(sig)

        results = {}
        for ticker, data in ticker_scores.items():
            if data["weight_total"] > 0:
                results[ticker] = {
                    "sentiment": data["weighted_sum"] / data["weight_total"],
                    "signal_count": data["count"],
                    "signals": data["signals"],
                }
        return results
