#!/usr/bin/env python
"""CLI entry point.

    python main.py --query 'Tech Sector Q3 Analysis'
    python main.py --offline            # keyless demo run
"""
from __future__ import annotations

import argparse
import logging
import sys

from config.settings import DEFAULT_QUERY


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Agent Market Research & Report Generator (CrewAI)")
    p.add_argument("--query", "-q", default=DEFAULT_QUERY, help="research topic")
    p.add_argument("--output-dir", "-o", default=None, help="where to write market_report.pdf etc. (default: ./output)")
    p.add_argument("--offline", action="store_true",
                   help="run the pipeline without LLM/network calls using bundled sample data")
    p.add_argument("--quiet", action="store_true", help="only log warnings")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    from crew.pipeline import run_pipeline
    from crew.crew import ConfigError

    try:
        result = run_pipeline(args.query, offline=args.offline, output_dir=args.output_dir)
    except ConfigError as exc:
        print(f"\nConfiguration error: {exc}", file=sys.stderr)
        return 2
    print("\nFinished.")
    for key in ("pdf", "markdown", "benchmarks"):
        print(f"  {key:<10}: {result[key]}")
    if result.get("data_fallback"):
        print("\nWARNING: the Researcher produced no usable metrics, so the PDF tables use the bundled "
              "SYNTHETIC sample dataset. Do not treat those numbers as real. Check your search key/quota "
              "and re-run.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
