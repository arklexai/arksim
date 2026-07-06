# SPDX-License-Identifier: Apache-2.0
"""Load MAA chat history and run the LLM-light goal discovery pipeline.

Usage:
    python examples/goal_discovery/run_maa.py
    python examples/goal_discovery/run_maa.py --input ~/Downloads/maa_chat_history_mock.json
    python examples/goal_discovery/run_maa.py --clustering kmeans --max-input 5000

Requires:
    pip install sentence-transformers scikit-learn
    OPENAI_API_KEY set in environment (or pass --provider anthropic with ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from _utils import print_results  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults: edit these or pass CLI flags instead
# ---------------------------------------------------------------------------
# DEFAULT_INPUT = Path.home() / "Downloads" / "maa_chat_history_mock.json"
DEFAULT_INPUT = Path.home() / "Downloads" / "bitext_conversations.json"
DEFAULT_CLUSTERING = "kmeans"  # "hdbscan" needs more data (use with large corpora)
DEFAULT_K_MIN = 3
DEFAULT_K_MAX = 7
DEFAULT_MIN_CLUSTER_SIZE = 5  # lower threshold suitable for small test data
DEFAULT_EXEMPLAR_COUNT = 3
DEFAULT_MIN_WORDS = 3
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_MAX_INPUT = 50_000  # cap before embedding; 0 = no cap
DEFAULT_MERGE_SIMILAR = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MAA goal discovery pipeline")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to MAA chat history JSON file",
    )
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
        help="Embedding backend (default: openai, avoids torch/torchvision dependency)",
    )
    p.add_argument(
        "--max-input",
        type=int,
        default=DEFAULT_MAX_INPUT,
        help="Max conversations to embed (0 = no cap)",
    )
    p.add_argument(
        "--no-merge", action="store_true", help="Skip the merge-similar-goals LLM pass"
    )
    p.add_argument(
        "--reformulated",
        action="store_true",
        default=True,
        help="Prefer reformulated_user_question for embedding (default: on)",
    )
    p.add_argument(
        "--intent-key",
        default="intent",
        help="meta key for HD-provided intent labels (default: 'intent')",
    )
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

    # Check for API key before doing any expensive work
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_key = provider_key_map.get(args.provider)
    if env_key and not os.environ.get(env_key):
        print(
            f"WARNING: {env_key} is not set. LLM cluster naming will fail.",
            file=sys.stderr,
        )

    # Import here so missing optional deps produce a clear error
    try:
        from arksim.scenario.goal_discovery import (
            ConversationInput,
            GoalDiscoveryPipeline,
            sample_conversations,
        )
        from arksim.scenario.goal_discovery.preprocessing import extract_first_turns
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Run: pip install -e '.[dev]' sentence-transformers scikit-learn",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Load records
    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # 2. Convert to ConversationInput
    conversations = [ConversationInput.from_maa_record(r) for r in records]
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

    # 4. Show what extract_first_turns will produce (dry-run preview)
    reformulated_key = "reformulated_question" if args.reformulated else None
    turns = extract_first_turns(
        conversations, min_words=args.min_words, reformulated_key=reformulated_key
    )
    print(f"Qualifying first turns: {len(turns)} / {len(conversations)}")
    print()
    print("Sample turns to be embedded:")
    for idx, text in turns[:5]:
        short = text if len(text) <= 90 else text[:87] + "..."
        intent = conversations[idx].meta.get("intent", "—")
        print(f"  [{intent:22s}] {short}")
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
