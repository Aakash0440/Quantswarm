"""
QuantSwarm — 100 Instruments: Stocks + Crypto
"""

STOCKS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    # Semiconductors
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR",
    # Consumer
    "WMT", "COST", "TGT", "NKE", "SBUX", "MCD", "HD",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO",
    # Industrial / Aerospace
    "BA", "GE", "HON", "CAT", "LMT", "RTX", "UPS", "FDX",
    # Cloud / SaaS
    "CRM", "ORCL", "NOW", "SNOW", "PLTR", "DDOG", "NET", "ZS",
    # ETFs
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "GLD",
    # Mid-cap / high-vol
    "COIN", "RBLX", "ABNB", "UBER", "HOOD",
]

CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "DOT-USD",
    "MATIC-USD", "LTC-USD", "UNI-USD", "ATOM-USD", "ALGO-USD",
    "XLM-USD", "NEAR-USD", "FIL-USD", "ICP-USD", "APT-USD",
]

# Deduplicate and cap at 100
ALL_INSTRUMENTS = list(dict.fromkeys(STOCKS + CRYPTO))[:100]

SECTOR_CLUSTERS = {
    "mega_tech":   ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "semis":       ["AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "AVGO"],
    "financials":  ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "healthcare":  ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR"],
    "energy":      ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO"],
    "cloud_saas":  ["CRM", "ORCL", "NOW", "SNOW", "PLTR", "DDOG", "NET", "ZS"],
    "crypto":      CRYPTO,
}

# Human-readable sector label lookup
TICKER_SECTOR = {}
for sector, tickers in SECTOR_CLUSTERS.items():
    for t in tickers:
        TICKER_SECTOR[t] = sector
