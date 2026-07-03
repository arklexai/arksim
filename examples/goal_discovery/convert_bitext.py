# SPDX-License-Identifier: Apache-2.0
"""Convert the Bitext retail/e-commerce chatbot CSV to the conversations-sample.json format.

Source CSV columns:
  instruction  - user message
  intent       - intent label (e.g. "add_product")
  category     - broad category (e.g. "CART")
  tags         - variation/quality tags (e.g. "BCILZ")
  response     - assistant response

Output schema (matches conversations-sample.json):
  id           - "conv-{n:06d}"
  created_at   - null (not in source)
  modified_at  - null (not in source)
  user_rating  - null (not in source)
  messages     - [{role, content, intent}, {role, content}]

Usage:
    python examples/goal_discovery/convert_bitext.py
    python examples/goal_discovery/convert_bitext.py --max-rows 1000
    python examples/goal_discovery/convert_bitext.py --input path/to/file.csv --output out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_INPUT = (
    Path.home()
    / "Downloads"
    / "bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
DEFAULT_OUTPUT = Path.home() / "Downloads" / "bitext_conversations.json"
DEFAULT_MAX_ROWS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Bitext CSV to conversations JSON")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Max conversations to include (0 = all rows)",
    )
    return p.parse_args()


def convert(input_path: Path, max_rows: int) -> list[dict]:
    conversations = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break

            instruction = (row.get("instruction") or "").strip()
            response = (row.get("response") or "").strip()
            intent = (row.get("intent") or "").strip()
            category = (row.get("category") or "").strip()
            tags = (row.get("tags") or "").strip()

            if not instruction or not response:
                continue

            conv: dict = {
                "id": f"conv-{i + 1:06d}",
                "created_at": None,
                "modified_at": None,
                "user_rating": None,
                "messages": [
                    {"role": "user", "content": instruction, "intent": intent},
                    {"role": "assistant", "content": response},
                ],
            }

            # Carry category and tags as top-level metadata so ConversationInput
            # can store them in its meta dict if needed.
            if category:
                conv["category"] = category
            if tags:
                conv["tags"] = tags

            conversations.append(conv)

    return conversations


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    conversations = convert(args.input, args.max_rows)
    print(f"Converted {len(conversations)} conversations")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
