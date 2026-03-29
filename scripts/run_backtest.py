#!/usr/bin/env python3
"""
QuantSwarm v3 — Backtest Runner
Usage: python scripts/run_backtest.py [--start 2019-01-01] [--end 2024-12-31]
"""
import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--windows", type=int, default=12)
    args = parser.parse_args()

    import yaml
    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    tickers = config["instruments"]["stocks"] + config["instruments"]["crypto"]

    from backtester.walk_forward import WalkForwardBacktester, BacktestConfig
    bt_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )
    backtester = WalkForwardBacktester(bt_config)
    result = backtester.run(tickers, n_windows=args.windows)
    backtester.print_report(result)

    # Save result
    import json
    output = {
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "calmar_ratio": result.calmar_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "benchmark_return": result.benchmark_return,
        "alpha": result.alpha,
        "per_window_sharpes": result.per_window_sharpes,
    }
    with open("backtest_result.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResult saved to backtest_result.json")

if __name__ == "__main__":
    main()
