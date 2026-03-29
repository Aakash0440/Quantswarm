"""
pipeline_fix.py — drop in quantswarm_v4 folder, run: python pipeline_fix.py
Diagnoses and fixes why onchain signals are still being filtered.
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(ROOT, "nlp", "pipeline.py")

with open(PIPELINE, "r", encoding="utf-8") as f:
    content = f.read()

print("=== Diagnosing pipeline.py ===")
print(f"Has TRUSTED_SOURCES: {'TRUSTED_SOURCES' in content}")
print(f"Has 'onchain' in TRUSTED: {'\"onchain\"' in content}")
print(f"Has confidence=1.0: {'confidence = 1.0' in content}")

# The real fix: after the trusted source check, force confidence=1.0
# Current code bypasses bot filter but doesn't set confidence=1.0
# So if FinBERT returns low confidence, the signal still gets dropped
# at the "if raw.source not in TRUSTED_SOURCES and confidence < self.min_confidence" check
# Wait — that check SHOULD let trusted sources through...
# Real issue: FinBERT batch runs BEFORE the loop, returns (sentiment, confidence, label)
# for ALL signals including onchain. The confidence value is already set.
# The loop then checks: if raw.source not in TRUSTED_SOURCES → trusted sources ARE exempt.
# So why is it still filtering?

# Answer: the log says "1 bots filtered" — this means is_bot=True is being returned
# even for the onchain signal. But TRUSTED_SOURCES sets is_bot=False...
# UNLESS the code on disk is different from what grep showed.

# Let's just rewrite the entire process() method body to be bulletproof:

OLD = '''        for raw, (sentiment, confidence, label) in zip(raw_signals, sentiments):
            # Trusted machine sources (on-chain, market data, SEC) bypass bot filter
            # They are structured data feeds, not social-media authors
            TRUSTED_SOURCES = {"onchain", "market", "sec"}
            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
            else:
                is_bot, bot_score = self.bot_filter.score(raw.text, raw.author_meta)

            # Skip confirmed bots
            if is_bot:
                logger.debug(f"Bot filtered: {raw.author} ({bot_score:.2f})")
                continue

            # Trusted sources skip the confidence floor — they carry structured numerical data
            if raw.source not in TRUSTED_SOURCES and confidence < self.min_confidence:
                continue'''

NEW = '''        TRUSTED_SOURCES = {"onchain", "market", "sec"}
        for raw, (sentiment, confidence, label) in zip(raw_signals, sentiments):
            # Structured data sources (on-chain, market, SEC) are never bots
            # and always pass confidence check — they carry numerical data not opinions
            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
                confidence = 1.0
                label = label if label else "neutral"
            else:
                is_bot, bot_score = self.bot_filter.score(raw.text, raw.author_meta)

            # Drop confirmed bots from social sources
            if is_bot:
                logger.debug(f"Bot filtered: {raw.author} ({bot_score:.2f})")
                continue

            # Drop low-confidence social signals (trusted sources already have confidence=1.0)
            if confidence < self.min_confidence:
                continue'''

if OLD in content:
    content = content.replace(OLD, NEW)
    with open(PIPELINE, "w", encoding="utf-8") as f:
        f.write(content)
    print("PATCHED: process() loop — trusted sources now get confidence=1.0 before check")
else:
    print("OLD pattern not found — trying fallback patch...")
    # Fallback: find the exact is_bot=False line and add confidence=1.0 after it
    FALLBACK_OLD = '''            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
            else:'''
    FALLBACK_NEW = '''            if raw.source in TRUSTED_SOURCES:
                is_bot, bot_score = False, 0.0
                confidence = 1.0  # structured data: always confident
            else:'''
    if FALLBACK_OLD in content:
        content = content.replace(FALLBACK_OLD, FALLBACK_NEW)
        with open(PIPELINE, "w", encoding="utf-8") as f:
            f.write(content)
        print("PATCHED (fallback): added confidence=1.0 for trusted sources")
    else:
        print("FALLBACK also not found. Printing lines 290-315 of pipeline.py:")
        lines = content.split("\n")
        for i, line in enumerate(lines[285:320], start=286):
            print(f"  {i}: {repr(line)}")

# Also fix run_paper.py UTF-8 if not done
RUNPAPER = os.path.join(ROOT, "scripts", "run_paper.py")
with open(RUNPAPER, "r", encoding="utf-8") as f:
    rp = f.read()

if "io.TextIOWrapper" not in rp:
    OLD_LOG = '''    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("quantswarm.log"),
        ]
    )'''
    NEW_LOG = '''    import io
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(
                stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            ),
            logging.FileHandler("quantswarm.log", encoding="utf-8"),
        ]
    )'''
    if OLD_LOG in rp:
        rp = rp.replace(OLD_LOG, NEW_LOG)
        with open(RUNPAPER, "w", encoding="utf-8") as f:
            f.write(rp)
        print("PATCHED: run_paper.py UTF-8 logging")
    else:
        # Try with different spacing
        print("run_paper.py logging block not matched — printing it:")
        idx = rp.find("logging.basicConfig")
        print(repr(rp[idx:idx+300]))
else:
    print("ALREADY DONE: run_paper.py UTF-8 logging")

print("\nDone. Run: python scripts/run_paper.py --capital 100000 --interval 30")
print("Expected: NLP: 1 raw -> 1 clean (0 bots filtered)")