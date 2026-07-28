"""CLI entry point for market-brief."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .fetch import CnbcProvider, YahooProvider
from .models import BriefData
from .normalize import load_symbols_config, normalize
from .rules import CATEGORY_ORDER, apply_rules


def brief_to_json(brief: BriefData) -> dict:
    """Convert BriefData to a JSON-serializable dict."""
    return {
        "headline": brief.headline,
        "generated_at": brief.generated_at,
        "categories": [
            {
                "category": cat.category,
                "sentiment": cat.sentiment,
                "quotes": [
                    {
                        "symbol": aq.quote.symbol,
                        "name_he": aq.quote.name_he,
                        "gender": aq.quote.gender,
                        "category": aq.quote.category,
                        "last": aq.quote.last,
                        "change_pct": round(aq.quote.change_pct, 4),
                        "unit": aq.quote.unit,
                        "decimals": aq.quote.decimals,
                        "direction": aq.direction,
                        "intensity": aq.intensity,
                        "formatted_change": aq.format_change_pct(),
                        "formatted_last": aq.format_last(),
                    }
                    for aq in cat.quotes
                ],
            }
            for cat in brief.categories
        ],
    }


def publish_brief(brief_text: str, project_root: Path) -> None:
    """Render brief into docs/index.html, commit, and push to GitHub."""
    template_path = project_root / "config" / "template.html"
    template = template_path.read_text(encoding="utf-8")

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    rendered = template.replace(
        "{{BRIEF_TEXT}}", html.escape(brief_text)
    ).replace(
        "{{TIMESTAMP}}", f"\u05e2\u05d5\u05d3\u05db\u05df \u05dc\u05d0\u05d7\u05e8\u05d5\u05e0\u05d4: {now}"
    )

    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    index_path = docs_dir / "index.html"
    index_path.write_text(rendered, encoding="utf-8")

    # Git commit and push
    subprocess.run(["git", "add", "docs/index.html"], cwd=project_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Update market brief"],
        cwd=project_root, check=True,
    )
    subprocess.run(["git", "push"], cwd=project_root, check=True)
    print("Published to GitHub Pages.", file=sys.stderr)


def build_provider(name: str):
    """Return a provider instance by name."""
    if name == "yahoo":
        return YahooProvider()
    if name == "cnbc":
        return CnbcProvider()
    raise ValueError(f"Unknown provider: {name!r}. Available: yahoo, cnbc")


def main(argv: list[str] | None = None) -> int:
    # Load .env from project root (market-brief/)
    _project_root = Path(__file__).resolve().parent.parent
    load_dotenv(_project_root / ".env")

    parser = argparse.ArgumentParser(
        prog="market-brief",
        description="Generate a Hebrew real-time market brief.",
    )
    parser.add_argument(
        "--provider",
        choices=["yahoo", "cnbc"],
        default="yahoo",
        help="Data provider (default: yahoo)",
    )
    parser.add_argument(
        "--categories",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated category filter (e.g. europe,us_futures)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON instead of plain text",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run stages 1-3 only (no LLM call)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write output to FILE instead of stdout",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "claude"],
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish brief to GitHub Pages (generates docs/index.html, commits and pushes)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Stage 1: Fetch ---
    provider = build_provider(args.provider)
    symbols_config = load_symbols_config()

    # Filter symbols based on market hours
    from zoneinfo import ZoneInfo
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    _log = logging.getLogger(__name__)

    # Check if Tel Aviv Stock Exchange is open (Sun-Thu 9:45-17:25 Israel time)
    israel_now = now_utc.astimezone(ZoneInfo("Asia/Jerusalem"))
    israel_time = israel_now.time()
    israel_weekday = israel_now.weekday()  # 0=Mon ... 6=Sun
    tase_open = (
        israel_weekday in (0, 1, 2, 3, 6)  # Sun-Thu (6=Sun, 0=Mon..3=Thu)
        and israel_time >= datetime.strptime("09:45", "%H:%M").time()
        and israel_time <= datetime.strptime("17:25", "%H:%M").time()
    )

    open_symbols = []
    for sym, cfg in symbols_config.items():
        tz_name = cfg.get("timezone")
        category = cfg.get("category")

        # US futures: only show when TASE is closed
        if category == "us_futures":
            if tase_open:
                _log.info("Skipping %s — TASE is open, hiding pre-market", sym)
                continue
            # When TASE is closed, include US futures regardless of US local time
            open_symbols.append(sym)
            continue

        # Commodities (gold, oil): always include
        if category == "commodities":
            open_symbols.append(sym)
            continue

        if tz_name is None:
            # No timezone (e.g. crypto) — always include
            open_symbols.append(sym)
            continue

        # Other assets: include if their local market is open (9:00-17:30)
        local_now = now_utc.astimezone(ZoneInfo(tz_name))
        local_time = local_now.time()
        market_open = local_time >= datetime.strptime("09:00", "%H:%M").time()
        market_close = local_time <= datetime.strptime("17:30", "%H:%M").time()
        if market_open and market_close:
            open_symbols.append(sym)
        else:
            _log.info(
                "Skipping %s — local time %s in %s is outside 9:00-17:30",
                sym, local_time.strftime("%H:%M"), tz_name,
            )

    if not open_symbols:
        print("ERROR: No markets are currently open.", file=sys.stderr)
        return 1

    raw = provider.fetch(open_symbols)
    if not raw:
        print("ERROR: No data fetched from provider.", file=sys.stderr)
        return 1

    # --- Stage 2: Normalize ---
    quotes = normalize(raw, symbols_config)

    # --- Stage 3: Rules ---
    if args.categories:
        for c in args.categories:
            if c not in CATEGORY_ORDER:
                print(f"ERROR: Unknown category {c!r}. Valid: {CATEGORY_ORDER}", file=sys.stderr)
                return 1

    brief = apply_rules(quotes, categories_filter=args.categories)

    # Check minimum 2 categories
    if len(brief.categories) < 2:
        print(
            f"ERROR: Only {len(brief.categories)} category(ies) have data. "
            "Need at least 2 to produce a brief. Aborting.",
            file=sys.stderr,
        )
        return 1

    # --- Dry-run: stop here (stages 1-3 only) ---
    if args.dry_run:
        output_data = brief_to_json(brief)
        output = json.dumps(output_data, ensure_ascii=False, indent=2)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output + "\n")
        else:
            print(output)
        return 0

    # --- Stage 4+5: Generate + Validate ---
    backend = args.backend
    if backend == "claude":
        key_var = "ANTHROPIC_API_KEY"
    else:
        key_var = "GEMINI_API_KEY"

    if not os.environ.get(key_var):
        print(
            f"ERROR: {key_var} is not set.\n"
            "Set it in your environment or in a .env file at the project root:\n"
            f"  {_project_root / '.env'}",
            file=sys.stderr,
        )
        return 1

    from .generate import generate

    try:
        hebrew_text = generate(brief, backend=backend)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    # --- Output ---
    if args.json_output:
        output_data = brief_to_json(brief)
        output_data["text"] = hebrew_text
        output = json.dumps(output_data, ensure_ascii=False, indent=2)
    else:
        # Strip the headline (first non-empty line + blank line after it)
        lines = hebrew_text.split("\n")
        while lines and (not lines[0].strip() or lines[0].startswith("שווקי העולם")):
            lines.pop(0)
        output = "\n".join(lines)

    # Determine output file path
    out_path = args.out
    if not out_path:
        output_dir = _project_root / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = ".json" if args.json_output else ".txt"
        out_path = str(output_dir / f"brief_{timestamp}{ext}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print(output)
    print(f"\nSaved to: {out_path}", file=sys.stderr)

    if args.publish:
        try:
            publish_brief(output, _project_root)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Publish failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
