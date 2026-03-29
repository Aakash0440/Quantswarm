#!/usr/bin/env python3
"""
QuantSwarm v4 — Paper Trading Runner
Usage: python scripts/run_paper.py [--capital 100000]
"""
import asyncio
import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantswarm.log"),
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=float(os.getenv("INITIAL_CAPITAL", "100000")))
    parser.add_argument("--interval", type=int, default=900, help="Cycle interval in seconds")
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════╗
║       QuantSwarm v4 — Paper Trading      ║
║  Capital: ${args.capital:,.0f}                   
║  Interval: {args.interval}s ({args.interval//60}min)              ║
║  Mode: PAPER (no real money)             ║
╚══════════════════════════════════════════╝
    """)

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force paper mode
    config["execution"]["mode"] = "paper"

    from agent.orchestrator import QuantSwarmAgent
    agent = QuantSwarmAgent(config, initial_capital=args.capital)
    # right after agent = QuantSwarmAgent(...)
    
    try:
        asyncio.run(agent.run(interval_sec=args.interval))
    except KeyboardInterrupt:
        print("\nQuantSwarm stopped.")

if __name__ == "__main__":
    main()
