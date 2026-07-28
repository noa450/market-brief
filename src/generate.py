"""
LLM stage: generate Hebrew market brief from structured data.

Supports two backends:
  - claude: claude-sonnet-4-6 via Anthropic API (requires ANTHROPIC_API_KEY)
  - gemini: gemini-2.5-flash via Google GenAI (requires GEMINI_API_KEY)

Validates output and retries once on failure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from .models import BriefData
from .validate import validate

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

TEMPERATURE = 0.3
MAX_TOKENS = 2048


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")


def _load_style_guide() -> str:
    return (CONFIG_DIR / "style.md").read_text(encoding="utf-8")


def _load_examples() -> list[dict]:
    with open(PROMPTS_DIR / "examples.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _build_brief_json(brief: BriefData) -> dict:
    """Build the JSON payload sent to the LLM."""
    return {
        "headline": brief.headline,
        "categories": [
            {
                "category": cat.category,
                "sentiment": cat.sentiment,
                "quotes": [
                    {
                        "name_he": aq.quote.name_he,
                        "gender": aq.quote.gender,
                        "direction": aq.direction,
                        "intensity": aq.intensity,
                        "formatted_change": aq.format_change_pct(),
                        "formatted_last": aq.format_last(),
                        "unit": aq.quote.unit,
                    }
                    for aq in cat.quotes
                ],
            }
            for cat in brief.categories
        ],
    }


def _build_user_message(brief: BriefData, style_guide: str, correction: str | None = None) -> str:
    """Assemble the user message with data JSON, style guide, and optional correction."""
    brief_json = json.dumps(_build_brief_json(brief), ensure_ascii=False, indent=2)

    parts = [
        "נתוני השוק:",
        brief_json,
        "",
        "מדריך סגנון:",
        style_guide,
    ]

    if correction:
        parts.extend([
            "",
            "הניסיון הקודם נדחה על ידי מערכת האימות. תקן את הבעיות הבאות:",
            correction,
        ])

    return "\n".join(parts)


# ── Claude backend ──

def _call_claude(system_prompt: str, messages: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text.strip()


def _build_claude_messages(brief: BriefData, style_guide: str, examples: list[dict],
                           correction: str | None = None) -> list[dict]:
    """Build the messages array with few-shot examples for Claude."""
    messages = []

    for ex in examples:
        ex_input = json.dumps(ex["input"], ensure_ascii=False, indent=2)
        messages.append({
            "role": "user",
            "content": f"נתוני השוק:\n{ex_input}\n\nמדריך סגנון:\n{style_guide}",
        })
        messages.append({
            "role": "assistant",
            "content": ex["output"],
        })

    messages.append({
        "role": "user",
        "content": _build_user_message(brief, style_guide, correction),
    })

    return messages


# ── Gemini backend ──

def _call_gemini(system_prompt: str, messages: list[dict]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Convert messages to Gemini Content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])],
        ))

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text.strip()


# ── Main generate function ──

def generate(brief: BriefData, backend: str = "gemini") -> str:
    """
    Generate Hebrew market brief text using the LLM.

    Args:
        brief: The structured brief data from the rules stage.
        backend: "claude" or "gemini".

    Validates the output and retries once on failure.
    On second failure, prints diagnostics to stderr and raises SystemExit(1).

    Returns the validated Hebrew text.
    """
    system_prompt = _load_system_prompt()
    style_guide = _load_style_guide()
    examples = _load_examples()

    if backend == "claude":
        call_fn = _call_claude
    elif backend == "gemini":
        call_fn = _call_gemini
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    correction = None

    for attempt in range(2):
        # Both backends use the same message format
        messages = _build_claude_messages(brief, style_guide, examples, correction)

        logger.info("LLM call attempt %d/%d (backend=%s)", attempt + 1, 2, backend)

        try:
            generated_text = call_fn(system_prompt, messages)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

        logger.debug("LLM output (%d chars):\n%s", len(generated_text), generated_text)

        # Validate
        result = validate(generated_text, brief)

        if result.passed:
            logger.info("Validation passed on attempt %d", attempt + 1)
            return generated_text

        # Validation failed
        error_summary = result.error_summary()
        logger.warning("Validation failed on attempt %d:\n%s", attempt + 1, error_summary)

        if attempt == 0:
            correction = error_summary
        else:
            brief_json = json.dumps(
                _build_brief_json(brief), ensure_ascii=False, indent=2
            )
            print("ERROR: Validation failed twice. Aborting.", file=sys.stderr)
            print("\n--- Input JSON ---", file=sys.stderr)
            print(brief_json, file=sys.stderr)
            print("\n--- Rejected text ---", file=sys.stderr)
            print(generated_text, file=sys.stderr)
            print("\n--- Validation errors ---", file=sys.stderr)
            print(error_summary, file=sys.stderr)
            raise SystemExit(1)

    raise SystemExit(1)
