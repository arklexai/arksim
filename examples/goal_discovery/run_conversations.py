# SPDX-License-Identifier: Apache-2.0
"""Run the LLM-light goal discovery pipeline on a conversations-format JSON file.

Supports any JSON file that follows the conversations-sample.json schema:
  {id, created_at, modified_at, user_rating, messages: [{role, content, intent?}],
   category?, tags?}

This is the format produced by convert_bitext.py.

Usage:
    python examples/goal_discovery/run_conversations.py
    python examples/goal_discovery/run_conversations.py --input ~/Downloads/bitext_conversations.json
    python examples/goal_discovery/run_conversations.py --clustering hdbscan --max-input 5000

Requires:
    OPENAI_API_KEY set in environment
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from _utils import print_results  # noqa: E402

DEFAULT_INPUT = Path.home() / "Downloads" / "bitext_conversations.json"
DEFAULT_CLUSTERING = "kmeans"
DEFAULT_K_MIN = 3
DEFAULT_K_MAX = 15
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_EXEMPLAR_COUNT = 3
DEFAULT_MIN_WORDS = 3
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_MAX_INPUT = 50_000
DEFAULT_MERGE_SIMILAR = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run goal discovery on conversations JSON")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--clustering", choices=["hdbscan", "kmeans"], default=DEFAULT_CLUSTERING
    )
    p.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    p.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    p.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    p.add_argument("--exemplars", type=int, default=DEFAULT_EXEMPLAR_COUNT)
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    p.add_argument("--model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    p.add_argument(
        "--embedding-provider",
        choices=["openai", "sentence-transformers"],
        default="openai",
    )
    p.add_argument(
        "--max-input",
        type=int,
        default=DEFAULT_MAX_INPUT,
        help="Max conversations to embed (0 = no cap)",
    )
    p.add_argument("--no-merge", action="store_true")
    return p.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("ERROR: expected a JSON array of records", file=sys.stderr)
        sys.exit(1)
    return data


def main() -> None:
    args = parse_args()

    provider_key_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    env_key = provider_key_map.get(args.provider)
    if env_key and not os.environ.get(env_key):
        print(
            f"WARNING: {env_key} is not set. LLM cluster naming will fail.",
            file=sys.stderr,
        )

    try:
        from arksim.scenario.goal_discovery import (
            ConversationInput,
            GoalDiscoveryPipeline,
            sample_conversations,
        )
        from arksim.scenario.goal_discovery.preprocessing import extract_first_turns
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Load records
    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # 2. Convert to ConversationInput using the conversations schema
    conversations = [ConversationInput.from_conversations_record(r) for r in records]
    print(f"Converted {len(conversations)} conversations")

    out_path = args.input.parent / (args.input.stem + "_conversations.json")
    with open(out_path, "w") as f:
        json.dump(
            [dataclasses.asdict(c) for c in conversations],
            f,
            indent=2,
            default=str,
        )
    print(f"Saved conversations to {out_path}")

    # 3. Sample if max_input is set
    if args.max_input and len(conversations) > args.max_input:
        conversations = sample_conversations(conversations, n=args.max_input, seed=42)
        print(
            f"Sampled down to {len(conversations)} conversations (max_input={args.max_input})"
        )

    # 4. Dry-run preview of first turns
    turns = extract_first_turns(conversations, min_words=args.min_words)
    print(f"Qualifying first turns: {len(turns)} / {len(conversations)}")
    print()
    print("Sample turns to be embedded:")
    for idx, text in turns[:5]:
        short = text if len(text) <= 90 else text[:87] + "..."
        intent = conversations[idx].meta.get("intent", "—")
        print(f"  [{intent:25s}] {short}")
    if len(turns) > 5:
        print(f"  ... and {len(turns) - 5} more")
    print()

    # 5. Build and run the pipeline
    pipeline = GoalDiscoveryPipeline(
        embedding_provider=args.embedding_provider,
        clustering_method=args.clustering,
        min_cluster_size=args.min_cluster_size,
        k_range=(args.k_min, args.k_max),
        exemplar_count=args.exemplars,
        min_words=args.min_words,
        llm_model=args.model,
        llm_provider=args.provider,
        merge_similar=not args.no_merge,
        max_input=args.max_input or None,
    )

    print(f"Running GoalDiscoveryPipeline (clustering={args.clustering}) ...")
    result = pipeline.discover(conversations)

    print_results(result)


if __name__ == "__main__":
    main()
