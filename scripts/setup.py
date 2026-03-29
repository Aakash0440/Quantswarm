#!/usr/bin/env python3
"""
QuantSwarm v3 — Setup Script
Run once to initialize database, validate config, test connections.
Usage: python scripts/setup.py
"""
import os
import sys
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("setup")


def check_python_version():
    if sys.version_info < (3, 11):
        logger.error("Python 3.11+ required. Current: %s", sys.version)
        sys.exit(1)
    logger.info("✓ Python version: %s", sys.version.split()[0])


def check_env():
    from dotenv import load_dotenv
    load_dotenv()
    required = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
    optional = ["ALPACA_API_KEY", "GLASSNODE_API_KEY", "TELEGRAM_BOT_TOKEN"]
    missing_required = [k for k in required if not os.getenv(k)]
    missing_optional = [k for k in optional if not os.getenv(k)]

    if missing_required:
        logger.warning("Missing required env vars: %s", missing_required)
        logger.warning("Reddit data source will be unavailable. Continuing with RSS+SEC only.")
    else:
        logger.info("✓ Required env vars present")

    if missing_optional:
        logger.info("Optional env vars not set (features limited): %s", missing_optional)


def check_config():
    path = "config/base.yaml"
    if not os.path.exists(path):
        logger.error("Config not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        config = yaml.safe_load(f)
    n_stocks = len(config["instruments"]["stocks"])
    n_crypto = len(config["instruments"]["crypto"])
    logger.info("✓ Config loaded: %d stocks + %d crypto = %d instruments", n_stocks, n_crypto, n_stocks + n_crypto)
    return config


def init_database():
    try:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL", "sqlite:///./quantswarm.db")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, source TEXT, sentiment REAL,
                    confidence REAL, timestamp TEXT, is_bot INTEGER
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, direction TEXT, magnitude REAL,
                    confidence REAL, horizon TEXT, timestamp TEXT,
                    model_contributions TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, direction TEXT, size_pct REAL,
                    entry_price REAL, exit_price REAL, pnl_pct REAL,
                    entry_time TEXT, exit_time TEXT, slippage_bps REAL
                )
            """))
            conn.commit()
        logger.info("✓ Database initialized: %s", db_url)
    except Exception as e:
        logger.warning("Database init error (non-fatal): %s", e)


def test_market_data():
    try:
        import yfinance as yf
        data = yf.download("SPY", period="5d", progress=False)
        if not data.empty:
            logger.info("✓ yfinance market data: SPY latest close = %.2f", float(data["Close"].iloc[-1]))
        else:
            logger.warning("yfinance returned empty data")
    except Exception as e:
        logger.warning("Market data test failed: %s", e)


def test_sec_edgar():
    try:
        from sec_edgar_downloader import Downloader
        logger.info("✓ SEC EDGAR downloader available")
    except ImportError:
        logger.warning("sec-edgar-downloader not installed: pip install sec-edgar-downloader")


def print_summary():
    print("\n" + "="*50)
    print("QuantSwarm v3 — Setup Complete")
    print("="*50)
    print("\nNext steps:")
    print("  1. Fill in .env (copy from .env.example)")
    print("  2. Run paper trade: python scripts/run_paper.py")
    print("  3. View dashboard: uvicorn dashboard.api:app --port 8000")
    print("  4. Run tests: pytest tests/ -v")
    print("\nDO NOT set TRADING_MODE=live until 60+ days of paper validation.")
    print("="*50)


if __name__ == "__main__":
    check_python_version()
    check_env()
    config = check_config()
    init_database()
    test_market_data()
    test_sec_edgar()
    print_summary()
