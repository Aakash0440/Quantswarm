"""
QuantSwarm v4 — One-shot patch script
Run from quantswarm_v4 folder:  python fix_all.py
Fixes:
  1. orchestrator.py  — self.risk.positions -> self.risk.state.positions
  2. nlp/pipeline.py  — trusted sources (onchain/market/sec) bypass bot filter AND confidence floor
  3. scripts/run_paper.py — Unicode arrow crash on Windows cp1252 console
  4. nlp/pipeline.py  — replace arrow → with -> in log messages (Windows safe)
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def patch(filepath, old, new, label):
    full = os.path.join(ROOT, filepath)
    with open(full, "r", encoding="utf-8") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  PATCHED: {label}")
    elif new in content:
        print(f"  ALREADY DONE: {label}")
    else:
        print(f"  NOT FOUND (check manually): {label}")
        print(f"    File: {filepath}")

print("\n=== QuantSwarm v4 Patch Script ===\n")

# ── Fix 1: orchestrator.py positions attribute ──────────────────────────────
patch(
    "agent/orchestrator.py",
    "list(self.risk.positions.keys())",
    "list(self.risk.state.positions.keys())",
    "orchestrator: risk.positions -> risk.state.positions"
)

# ── Fix 2a: nlp/pipeline.py — trusted source bypass (bot filter) ────────────
patch(
    "nlp/pipeline.py",
    '''        for raw, (sentiment, confidence, label) in zip(raw_signals, sentiments):
            # Bot filter
            is_bot, bot_score = self.bot_filter.score(raw.text, raw.author_meta)

            # Skip confirmed bots
            if is_bot:
                logger.debug(f"Bot filtered: {raw.author} ({bot_score:.2f})")
                continue

            # Skip low confidence
            if confidence < self.min_confidence:
                continue''',
    '''        for raw, (sentiment, confidence, label) in zip(raw_signals, sentiments):
            # Structured data sources bypass social-media bot filter entirely
            TRUSTED_SOURCES = {"onchain", "market", "sec"}
            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
                confidence = 1.0   # structured data always confident
                label = "neutral"
            else:
                is_bot, bot_score = self.bot_filter.score(raw.text, raw.author_meta)

            # Skip confirmed bots
            if is_bot:
                logger.debug(f"Bot filtered: {raw.author} ({bot_score:.2f})")
                continue

            # Skip low-confidence social signals (trusted sources already exempt above)
            if raw.source not in TRUSTED_SOURCES and confidence < self.min_confidence:
                continue''',
    "pipeline: trusted sources bypass bot filter + confidence floor"
)

# ── Fix 2b: nlp/pipeline.py — already-patched version (different wording) ───
patch(
    "nlp/pipeline.py",
    '''            # Trusted sources skip the confidence floor — they carry structured numerical data
            if raw.source not in TRUSTED_SOURCES and confidence < self.min_confidence:
                continue''',
    '''            # Trusted sources get confidence=1.0 and skip the floor
            if raw.source in TRUSTED_SOURCES:
                confidence = 1.0
                label = "neutral"
            elif confidence < self.min_confidence:
                continue''',
    "pipeline: trusted sources get confidence=1.0 (alt patch)"
)

# ── Fix 3: Replace arrow characters in log messages (Windows cp1252 safe) ───
patch(
    "nlp/pipeline.py",
    'logger.info(f"NLP: {len(raw_signals)} raw \u2192 {len(processed)} clean ({bot_count} bots filtered)")',
    'logger.info(f"NLP: {len(raw_signals)} raw -> {len(processed)} clean ({bot_count} bots filtered)")',
    "pipeline: replace arrow in NLP log message"
)

patch(
    "agent/orchestrator.py",
    'logger.info(f"Observe: {len(raw_signals)} raw \u2192 {len(processed)} processed for {len(ticker_sentiments)} tickers")',
    'logger.info(f"Observe: {len(raw_signals)} raw -> {len(processed)} processed for {len(ticker_sentiments)} tickers")',
    "orchestrator: replace arrow in Observe log message"
)

# Catch any other arrow in orchestrator and pipeline
for filepath in ["agent/orchestrator.py", "nlp/pipeline.py", "ingestion/sources.py"]:
    full = os.path.join(ROOT, filepath)
    try:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if "\u2192" in content:
            fixed = content.replace("\u2192", "->")
            with open(full, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"  PATCHED: removed all arrow chars from {filepath}")
    except Exception as e:
        print(f"  ERROR on {filepath}: {e}")

# ── Fix 4: scripts/run_paper.py — UTF-8 console handler ─────────────────────
patch(
    "scripts/run_paper.py",
    '''    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("quantswarm.log"),
        ]
    )''',
    '''    import sys, io
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(
                stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            ),
            logging.FileHandler("quantswarm.log", encoding="utf-8"),
        ]
    )''',
    "run_paper: UTF-8 console handler (fixes Windows arrow crash)"
)

print("\n=== Done. Run: python scripts/run_paper.py --capital 100000 --interval 30 ===\n")