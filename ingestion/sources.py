"""
QuantSwarm v3 — Data Ingestion Layer
Unified interface for all 6 signal sources.
"""
from __future__ import annotations
import os
import time
import hashlib
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional
import logging

# Optional heavy deps — imported lazily so the module loads in test/paper mode
# without requiring all data-source packages to be installed.
try:
    import praw as _praw
    logging.getLogger("praw").setLevel(logging.ERROR)
except ImportError:
    _praw = None

try:
    import feedparser as _feedparser
except ImportError:
    _feedparser = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    from sec_edgar_downloader import Downloader as _SECDownloader
except ImportError:
    _SECDownloader = None
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("quantswarm.ingestion")


@dataclass
class RawSignal:
    source: str          # reddit | twitter | news | sec | onchain | market
    ticker: str          # AAPL, BTC-USD, etc.
    text: str            # raw content
    timestamp: datetime
    url: str = ""
    author: str = ""
    author_meta: dict = field(default_factory=dict)  # for bot filtering
    signal_id: str = ""

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = hashlib.md5(
                f"{self.source}{self.ticker}{self.timestamp}{self.text[:50]}".encode()
            ).hexdigest()[:12]


class SignalSource(ABC):
    """Base class for all data sources."""

    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    @abstractmethod
    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        """Fetch signals for a specific ticker."""
        pass

    async def fetch_all(self, lookback_hours: int = 24) -> List[RawSignal]:
        """Fetch signals for all tickers."""
        results = []
        for ticker in self.tickers:
            try:
                signals = await self.fetch(ticker, lookback_hours)
                results.extend(signals)
                await asyncio.sleep(0.1)  # rate limiting
            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] Failed for {ticker}: {e}")
        return results


class RedditSource(SignalSource):
    """Pull posts and comments from finance subreddits."""

    SUBREDDITS = ["wallstreetbets", "investing", "stocks", "CryptoCurrency", "SecurityAnalysis"]

    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        if _praw is None:
            raise ImportError("praw not installed — run: pip install praw")
        cid = os.getenv("REDDIT_CLIENT_ID")
        csecret = os.getenv("REDDIT_CLIENT_SECRET")
        if not cid or not csecret:
            raise ValueError("REDDIT_CLIENT_ID/SECRET missing in .env")
        self.reddit = _praw.Reddit(
            client_id=cid,
            client_secret=csecret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "QuantSwarm/4.0"),
        )

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        clean_ticker = ticker.replace("-USD", "").replace("-", "")

        for sub_name in self.SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                # Search for ticker mentions
                for post in subreddit.search(f"${clean_ticker}", limit=20, time_filter="day"):
                    post_time = datetime.utcfromtimestamp(post.created_utc)
                    if post_time < cutoff:
                        continue
                    signals.append(RawSignal(
                        source="reddit",
                        ticker=ticker,
                        text=f"{post.title} {post.selftext[:500]}",
                        timestamp=post_time,
                        url=f"https://reddit.com{post.permalink}",
                        author=str(post.author),
                        author_meta={
                            "account_age_days": max(0, (datetime.utcnow() - datetime.utcfromtimestamp(
                                post.author.created_utc if post.author else 0
                            )).days) if post.author else 0,
                            "comment_karma": getattr(post.author, "comment_karma", 0) if post.author else 0,
                            "post_karma": getattr(post.author, "link_karma", 0) if post.author else 0,
                        }
                    ))
            except Exception as e:
                logger.debug(f"Reddit fetch error for {sub_name}/{clean_ticker}: {e}")

        return signals


class TwitterNitterSource(SignalSource):
    """Fetch Twitter/X data via Nitter RSS (free, no API key needed)."""

    NITTER_INSTANCES = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ]

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        clean_ticker = ticker.replace("-USD", "")
        query = f"%24{clean_ticker}"  # $TICKER URL encoded
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for instance in self.NITTER_INSTANCES:
            try:
                url = f"{instance}/search/rss?f=tweets&q={query}&since_id=0"
                feed = _feedparser.parse(url) if _feedparser else {"entries": []}
                for entry in feed.entries[:30]:
                    try:
                        pub = datetime(*entry.published_parsed[:6])
                        if pub < cutoff:
                            continue
                        signals.append(RawSignal(
                            source="twitter",
                            ticker=ticker,
                            text=entry.title,
                            timestamp=pub,
                            url=entry.link,
                            author=getattr(entry, "author", "unknown"),
                            author_meta={"via": "nitter_rss"},
                        ))
                    except Exception:
                        pass
                if signals:
                    break  # got results, don't try other instances
            except Exception as e:
                logger.debug(f"Nitter fetch error {instance}: {e}")

        return signals


class NewsRSSSource(SignalSource):
    """Fetch financial news from RSS feeds."""

    RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ]

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        clean_ticker = ticker.replace("-USD", "").replace("-", "")
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for feed_url in self.RSS_FEEDS:
            try:
                feed = _feedparser.parse(feed_url) if _feedparser else {"entries": []}
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    full_text = f"{title} {summary}"
                    # Only include if ticker is mentioned
                    if clean_ticker.upper() not in full_text.upper():
                        continue
                    try:
                        pub = datetime(*entry.published_parsed[:6])
                    except Exception:
                        pub = datetime.utcnow()
                    if pub < cutoff:
                        continue
                    signals.append(RawSignal(
                        source="news",
                        ticker=ticker,
                        text=full_text[:1000],
                        timestamp=pub,
                        url=entry.get("link", ""),
                        author=feed_url.split("/")[2],  # domain as source
                        author_meta={"feed": feed_url},
                    ))
            except Exception as e:
                logger.debug(f"RSS fetch error {feed_url}: {e}")

        return signals


class SECEdgarSource(SignalSource):
    """Fetch SEC filings: Form 4 (insider trades), 8-K (events), 10-Q."""

    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        self.dl = _SECDownloader(
            "QuantSwarm",
            os.getenv("SEC_USER_AGENT", "admin@quantswarm.io"),
            "/tmp/sec_downloads"
        )

    async def fetch(self, ticker: str, lookback_hours: int = 168) -> List[RawSignal]:
        """Lookback default 7 days for SEC filings."""
        signals = []
        clean_ticker = ticker.replace("-USD", "").replace("-", "")

        # Skip crypto — no SEC filings
        if "-USD" in ticker or ticker in ["BTC", "ETH", "SOL"]:
            return signals

        for form_type in ["4", "8-K", "10-Q"]:
            try:
                import os as _os
                dl_path = f"/tmp/sec_downloads/sec-edgar-filings/{clean_ticker}/{form_type}"
                self.dl.get(form_type, clean_ticker, limit=3, download_details=False)

                if _os.path.exists(dl_path):
                    for root, dirs, files in _os.walk(dl_path):
                        for fname in files[:3]:
                            fpath = _os.path.join(root, fname)
                            try:
                                with open(fpath, "r", errors="ignore") as f:
                                    content = f.read(2000)
                                signals.append(RawSignal(
                                    source="sec",
                                    ticker=ticker,
                                    text=f"Form {form_type}: {content[:800]}",
                                    timestamp=datetime.utcnow(),
                                    url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={clean_ticker}&type={form_type}",
                                    author_meta={"form_type": form_type},
                                ))
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"SEC fetch error {clean_ticker}/{form_type}: {e}")

        return signals


class MarketDataSource(SignalSource):
    """Fetch OHLCV, funding rates, and technical signals via yfinance."""

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        try:
            tf = ticker if ticker.endswith("-USD") else ticker
            data = yf.download(tf, period="5d", interval="1h", progress=False)
            if data.empty:
                return signals
            latest = data.tail(1)
            close = float(latest["Close"].iloc[0])
            vol = float(latest["Volume"].iloc[0])
            # Simple return
            if len(data) >= 24:
                prev = float(data["Close"].iloc[-24])
                ret_24h = (close - prev) / prev if prev > 0 else 0
            else:
                ret_24h = 0

            signals.append(RawSignal(
                source="market",
                ticker=ticker,
                text=f"Price: {close:.4f}, 24h return: {ret_24h:.4%}, Volume: {vol:.0f}",
                timestamp=datetime.utcnow(),
                author_meta={
                    "price": close,
                    "return_24h": ret_24h,
                    "volume": vol,
                },
            ))
        except Exception as e:
            logger.debug(f"Market data error {ticker}: {e}")

        return signals


class OnChainSource(SignalSource):
    """Fetch on-chain signals: funding rates, whale moves, DEX volume.
    Requires GLASSNODE_API_KEY or DUNE_API_KEY in .env.
    Falls back gracefully if no key present.
    """

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        # Only for crypto
        if "-USD" not in ticker:
            return signals

        api_key = os.getenv("GLASSNODE_API_KEY")
        if not api_key:
            return signals

        asset = ticker.replace("-USD", "").lower()
        try:
            async with httpx.AsyncClient() as client:
                # Funding rate
                r = await client.get(
                    "https://api.glassnode.com/v1/metrics/derivatives/futures_funding_rate_perpetual",
                    params={"a": asset, "api_key": api_key, "i": "24h"},
                    timeout=10.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        rate = data[-1].get("v", 0)
                        signals.append(RawSignal(
                            source="onchain",
                            ticker=ticker,
                            text=f"Funding rate {asset.upper()}: {rate:.6f}",
                            timestamp=datetime.utcnow(),
                            author_meta={"funding_rate": rate},
                        ))
        except Exception as e:
            logger.debug(f"OnChain fetch error {ticker}: {e}")

        return signals


class EtherscanSource(SignalSource):
    """Free-tier Etherscan fallback for on-chain signals.
    Works without any paid API key — uses Etherscan's public endpoints.
    Tracks ETH gas price, large transaction volume, and ERC-20 transfer counts
    as a proxy for network activity / whale sentiment.
    Set ETHERSCAN_API_KEY in .env for higher rate limits (still free).
    """

    BASE = "https://api.etherscan.io/api"
    CRYPTO_MAP = {"ETH-USD": "eth", "BTC-USD": None}  # Etherscan is ETH-native

    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
        self._fetched_this_cycle: bool = False  # only fetch once per cycle (chain-level data)

    async def fetch(self, ticker: str, lookback_hours: int = 24) -> List[RawSignal]:
        signals = []
        # Etherscan is Ethereum network data — only meaningful for ETH-USD, and only once per cycle
        if ticker != "ETH-USD":
            return signals
        if self._fetched_this_cycle:
            return signals
        self._fetched_this_cycle = True

        api_key = os.getenv("ETHERSCAN_API_KEY", "")  # free public key works without auth
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Gas oracle — high gas = congestion = on-chain activity spike
                r = await client.get(
                    self.BASE,
                    params={
                        "module": "gastracker",
                        "action": "gasoracle",
                        "apikey": api_key,
                    },
                )
                if r.status_code == 200 and r.json().get("status") == "1":
                    result = r.json()["result"]
                    safe_gas = float(result.get("SafeGasPrice", 0))
                    fast_gas = float(result.get("FastGasPrice", 0))
                    text = (
                        f"ETH network gas: safe={safe_gas:.0f} gwei, fast={fast_gas:.0f} gwei. "
                        f"{'High congestion — potential whale activity' if fast_gas > 50 else 'Normal network activity'}."
                    )
                    signals.append(RawSignal(
                        source="onchain",
                        ticker="ETH-USD",
                        text=text,
                        timestamp=datetime.utcnow(),
                        author_meta={"safe_gas": safe_gas, "fast_gas": fast_gas},
                    ))

                # ETH supply stats (free endpoint, no key required)
                r2 = await client.get(
                    self.BASE,
                    params={
                        "module": "stats",
                        "action": "ethsupply2",
                        "apikey": api_key,
                    },
                )
                if r2.status_code == 200 and r2.json().get("status") == "1":
                    supply = r2.json()["result"]
                    burned = int(supply.get("BurntFees", 0)) / 1e18
                    signals.append(RawSignal(
                        source="onchain",
                        ticker="ETH-USD",
                        text=f"ETH burned fees (EIP-1559): {burned:,.2f} ETH. Deflationary pressure.",
                        timestamp=datetime.utcnow(),
                        author_meta={"burned_eth": burned},
                    ))

        except Exception as e:
            logger.debug(f"EtherscanSource fetch error: {e}")

        return signals

    async def fetch_all(self, lookback_hours: int = 24) -> List[RawSignal]:
        """Override to reset per-cycle dedup flag before scanning all tickers."""
        self._fetched_this_cycle = False
        return await super().fetch_all(lookback_hours)


class IngestionManager:
    """Orchestrates all sources, deduplicates signals, rate-limits."""

    def __init__(self, tickers: List[str], config: dict):
        self.tickers = tickers
        self.config = config
        self.seen_ids: set = set()
        self.sources: List[SignalSource] = []
        self._init_sources()

    def _init_sources(self):
        c = self.config.get("sources", {})
        if c.get("reddit", True):
            try:
                self.sources.append(RedditSource(self.tickers))
            except (ValueError, ImportError) as e:
                logger.warning(f"RedditSource disabled: {e}")
        if c.get("twitter_nitter", True):
            self.sources.append(TwitterNitterSource(self.tickers))
        if c.get("news_rss", True):
            self.sources.append(NewsRSSSource(self.tickers))
        if c.get("sec_edgar", True):
            self.sources.append(SECEdgarSource(self.tickers))
        if c.get("market_data", True):
            self.sources.append(MarketDataSource(self.tickers))
        if c.get("on_chain", False):
            self.sources.append(OnChainSource(self.tickers))
        # EtherscanSource is always active — free tier, no key required
        # Set ETHERSCAN_API_KEY in .env for higher rate limits
        self.sources.append(EtherscanSource(self.tickers))
        logger.info(f"IngestionManager: {len(self.sources)} sources active")

    async def fetch_all(self, lookback_hours: int = 24) -> List[RawSignal]:
        """Fetch from all sources, deduplicate."""
        all_signals = []
        for source in self.sources:
            try:
                signals = await source.fetch_all(lookback_hours)
                for sig in signals:
                    if sig.signal_id not in self.seen_ids:
                        self.seen_ids.add(sig.signal_id)
                        all_signals.append(sig)
            except Exception as e:
                logger.error(f"Source {source.__class__.__name__} failed: {e}")

        logger.info(f"Ingestion complete: {len(all_signals)} unique signals")
        return all_signals

    async def stream(self, interval_sec: int = 900) -> AsyncGenerator[List[RawSignal], None]:
        """Continuous streaming mode."""
        while True:
            signals = await self.fetch_all(lookback_hours=1)
            if signals:
                yield signals
            await asyncio.sleep(interval_sec)
