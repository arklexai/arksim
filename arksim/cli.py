# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import yaml

from arksim.evaluator import Evaluation, EvaluationInput, run_evaluation
from arksim.simulation_engine import SimulationInput, run_simulation
from arksim.utils.logger import get_logger

logger = get_logger("arksim")

# Suppress various logging and warnings
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("a2a.client.card_resolver").setLevel(logging.WARNING)

_EXAMPLES_REPO_URL = "https://github.com/arklexai/arksim/archive/main.tar.gz"
_EXAMPLES_PREFIX = "examples/"


def _check_score_threshold(
    evaluator_output: Evaluation,
    score_threshold: float | None,
) -> bool:
    """Check if any conversation's overall_agent_score is below
    the threshold.

    Args:
        evaluator_output: Evaluation with conversations
        score_threshold: Threshold value (0.0 to 1.0),
            None to skip check

    Returns:
        True if all scores pass (or threshold is None),
            False if any score fails
    """
    if score_threshold is None:
        return True

    failed_conversations = []
    for convo in evaluator_output.conversations:
        if convo.overall_agent_score < score_threshold:
            failed_conversations.append(
                {
                    "conversation_id": convo.conversation_id,
                    "overall_agent_score": convo.overall_agent_score,
                }
            )

    if failed_conversations:
        logger.error(
            f"Score threshold check failed! "
            f"Threshold: {score_threshold}, "
            f"Failed conversations: {len(failed_conversations)}",
        )
        for fc in failed_conversations:
            logger.error(
                f"  Conversation {fc['conversation_id']}: "
                f"overall_agent_score={fc['overall_agent_score']:.3f}"
                f" < {score_threshold}",
            )
        return False

    logger.info(
        f"Score threshold check passed! "
        f"All {len(evaluator_output.conversations)} conversations "
        f"have overall_agent_score >= {score_threshold}",
    )
    return True


def _merge_cli_overrides(yaml_settings: dict, cli_overrides: dict) -> dict:
    """Merge CLI overrides into YAML settings.
    CLI values take priority.

    Args:
        yaml_settings: Settings loaded from YAML file
        cli_overrides: Settings provided via CLI options
            (None values are ignored)

    Returns:
        Merged settings dictionary
    """
    merged = yaml_settings.copy()
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged


# ============================================================================
# Argparse CLI - Dynamic argument parsing
# ============================================================================


def parse_extra_args(extra_args: list) -> dict:
    """Parse extra CLI arguments in --key value format."""
    overrides = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            # Convert --key-name to key_name (match YAML format)
            key = arg[2:].replace("-", "_")

            # Check for --key=value format
            if "=" in key:
                key, value = key.split("=", 1)
                overrides[key] = _parse_value(value)
                i += 1
            elif i + 1 < len(extra_args) and not extra_args[i + 1].startswith("-"):
                # --key value format
                value = extra_args[i + 1]
                overrides[key] = _parse_value(value)
                i += 2
            else:
                # Boolean flag without value (--flag means True)
                overrides[key] = True
                i += 1
        else:
            i += 1
    return overrides


def _parse_value(value: str) -> bool | int | float | str:
    """Parse a string value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value


def validate_overrides(overrides: dict, valid_keys: set) -> None:
    invalid_keys = set(overrides.keys()) - valid_keys
    if invalid_keys:
        logger.error(f"Unknown options: {', '.join(sorted(invalid_keys))}")
        logger.info(f"Valid options: {', '.join(sorted(valid_keys))}")
        sys.exit(1)


def _log_config_summary(label: str, settings: dict) -> None:
    """Log a compact summary of resolved configuration."""
    logger.info(f"\n{label} configuration:")
    for key, value in sorted(settings.items()):
        logger.info(f"  {key}: {value}")
    logger.info("")


def _run_show_prompts(category: str | None) -> None:
    """Print evaluation prompts, optionally filtered by category."""
    from arksim.evaluator.prompt_registry import (
        get_categories,
        get_prompts_by_category,
    )

    matches = get_prompts_by_category(category)
    if not matches:
        print(
            f"Unknown category: '{category}'. Available: {', '.join(get_categories())}"
        )
        sys.exit(1)

    for cat in matches:
        print(f"{'=' * 60}")
        print(f"Category: {cat.category}")
        print(f"Description: {cat.description}")
        print(f"{'=' * 60}")
        for entry in cat.prompts:
            print(f"\n--- {entry.name} ---")
            print(entry.text.strip())
            print()


def _run_examples(
    name: str | None = None,
    list_only: bool = False,
) -> None:
    """Download example projects from the arksim GitHub repo."""
    import io
    import tarfile
    from urllib.request import urlopen

    logger.info("Fetching examples from arksim GitHub repo...")
    with urlopen(_EXAMPLES_REPO_URL) as resp:
        data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        top = tar.getmembers()[0].name.split("/")[0]
        example_root = f"{top}/{_EXAMPLES_PREFIX}"

        available = sorted(
            {
                m.name.removeprefix(example_root).split("/")[0]
                for m in tar.getmembers()
                if m.name.startswith(example_root)
                and m.name != example_root.rstrip("/")
            }
        )

        if list_only:
            print("Available examples:")
            for ex in available:
                print(f"  {ex}")
            return

        if name and name not in available:
            logger.error(f"Unknown example '{name}'. Available: {', '.join(available)}")
            sys.exit(1)

        if name:
            filter_prefix = f"{top}/{_EXAMPLES_PREFIX}{name}/"
            dest_path = os.path.join("examples", name)
        else:
            filter_prefix = example_root
            dest_path = "examples"

        if os.path.exists(dest_path):
            logger.error(
                f"'{dest_path}' already exists. "
                "Remove it first or choose a "
                "different directory."
            )
            sys.exit(1)

        for member in tar.getmembers():
            if member.name.startswith(filter_prefix):
                member.name = member.name.removeprefix(f"{top}/")
                tar.extract(member, ".", filter="data")

    logger.info(f"Downloaded to {os.path.abspath(dest_path)}")


def build_parser(valid_commands: list[str] | None = None) -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Args:
        valid_commands: List of valid command names.
            Defaults to ["simulate", "evaluate",
            "simulate-evaluate"].
    """
    if valid_commands is None:
        valid_commands = [
            "simulate",
            "evaluate",
            "simulate-evaluate",
            "show-prompts",
            "examples",
            "ui",
        ]

    commands_str = ", ".join(valid_commands)

    examples = []
    if "simulate" in valid_commands:
        examples.append(
            "  arksim simulate config.yaml --scenario-file-path ./scenario.json  # Simulate conversations"
        )
    if "evaluate" in valid_commands:
        examples.append(
            "  arksim evaluate config.yaml --simulation-file-path ./results/simulation/simulation.json  # Evaluate results"
        )
    if "simulate-evaluate" in valid_commands:
        examples.append(
            "  arksim simulate-evaluate config.yaml                           # Simulate then evaluate"
        )
    if "show-prompts" in valid_commands:
        examples.append(
            "  arksim show-prompts --category agent_behavior_failure          # Show prompts by category"
        )
    if "examples" in valid_commands:
        examples.append(
            "  arksim examples                                                # Download all example projects"
        )
        examples.append(
            "  arksim examples bank-insurance                                 # Download one example"
        )
        examples.append(
            "  arksim examples --list                                         # List available examples"
        )
    if "ui" in valid_commands:
        examples.append(
            "  arksim ui                                                      # Launch web UI control plane"
        )
    examples_str = "\n".join(examples)

    parser = argparse.ArgumentParser(
        prog="arksim",
        description="Arksim CLI - Run agent simulations and evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"\nCommands: {commands_str}\n\nExamples:\n{examples_str}\n",
    )

    parser.add_argument(
        "command",
        type=str,
        nargs="?",
        default="",
        help=f"Command to execute ({commands_str})",
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "additional_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments (e.g. --seed 42 --enable-topic-modeling false)",
    )

    return parser


def main() -> None:
    """Main entry point for the arksim CLI."""
    s_time = time.time()
    parser = build_parser()
    args = parser.parse_args()

    valid_commands = [
        "simulate",
        "evaluate",
        "simulate-evaluate",
        "show-prompts",
        "examples",
        "ui",
    ]

    # Check execution command and config file
    if not args.command or args.command not in valid_commands:
        parser.print_help()
        sys.exit(1)

    if args.command == "show-prompts":
        overrides = parse_extra_args(args.additional_args)
        category = overrides.get("category")
        _run_show_prompts(category)
        return

    if args.command == "examples":
        overrides = parse_extra_args(args.additional_args)
        list_only = overrides.get("list", False)
        _run_examples(name=args.config_file, list_only=list_only)
        return

    if args.command == "ui":
        from arksim.ui.app import launch_ui

        overrides = parse_extra_args(args.additional_args)
        port = int(overrides.get("port", 8080))
        launch_ui(port=port)
        return

    use_config_file = (
        args.config_file
        and os.path.exists(args.config_file)
        and args.config_file.endswith((".yaml", ".yml"))
    )

    overrides = parse_extra_args(args.additional_args)

    # Load settings from YAML file if valid, otherwise use empty dict
    settings = {}
    if use_config_file:
        try:
            with open(args.config_file) as f:
                settings = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(
                f"Could not load config file '{args.config_file}': {e}",
            )
            sys.exit(1)
    else:
        logger.warning("No config YAML file provided.")

    # override with the environment variables
    if os.getenv("LLM_PROVIDER"):
        settings["provider"] = os.getenv("LLM_PROVIDER")

    # Resolve log level: env var takes priority over YAML key
    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        logging.getLogger("arksim").setLevel(log_level.upper())

    # Extract verbose flag before building model inputs
    verbose = overrides.pop("verbose", False)

    if args.command == "simulate":
        valid_keys = set(SimulationInput.model_fields.keys())
        validate_overrides(overrides, valid_keys)
        settings = _merge_cli_overrides(settings, overrides)
        simulation_input = SimulationInput(**settings)
        _log_config_summary("Simulation", simulation_input.model_dump())
        asyncio.run(run_simulation(simulation_input, verbose=verbose))
    elif args.command == "evaluate":
        valid_keys = set(EvaluationInput.model_fields.keys())
        validate_overrides(overrides, valid_keys)
        settings = _merge_cli_overrides(settings, overrides)
        evaluation_input = EvaluationInput(**settings)
        _log_config_summary("Evaluation", evaluation_input.model_dump())
        evaluator_output = run_evaluation(evaluation_input)

        # Check score threshold if specified
        if not _check_score_threshold(
            evaluator_output, evaluation_input.score_threshold
        ):
            sys.exit(1)
    elif args.command == "simulate-evaluate":
        valid_keys = set(SimulationInput.model_fields.keys()) | set(
            EvaluationInput.model_fields.keys()
        )
        validate_overrides(overrides, valid_keys)
        settings = _merge_cli_overrides(settings, overrides)

        simulation_settings = {
            k: v for k, v in settings.items() if k in SimulationInput.model_fields
        }
        simulation_input = SimulationInput(**simulation_settings)
        _log_config_summary("Simulation", simulation_input.model_dump())

        sim_start = time.time()
        simulation_output = asyncio.run(
            run_simulation(simulation_input, verbose=verbose)
        )
        sim_elapsed = time.time() - sim_start
        logger.info(f"Simulation completed in {sim_elapsed:.2f} seconds")

        evaluation_settings = {
            k: v for k, v in settings.items() if k in EvaluationInput.model_fields
        }
        evaluation_input = EvaluationInput.model_validate(
            evaluation_settings,
            context={"skip_input_dir_validation": True},
        )
        _log_config_summary("Evaluation", evaluation_input.model_dump())

        eval_start = time.time()
        evaluator_output = run_evaluation(
            evaluation_input, simulation=simulation_output
        )
        eval_elapsed = time.time() - eval_start
        logger.info(f"Evaluation completed in {eval_elapsed:.2f} seconds")

        if not _check_score_threshold(
            evaluator_output, evaluation_input.score_threshold
        ):
            sys.exit(1)

    logger.info(f"Total elapsed: {time.time() - s_time:.2f} seconds")


if __name__ == "__main__":
    main()
