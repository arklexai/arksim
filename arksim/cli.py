# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap
import time

import yaml
from pydantic import ValidationError

from arksim import __version__
from arksim.evaluator import (
    Evaluation,
    EvaluationInput,
    check_numeric_thresholds,
    check_qualitative_failure_labels,
    check_score_threshold,
    run_evaluation,
)
from arksim.simulation_engine import SimulationInput, run_simulation
from arksim.utils.logger import get_logger

# Exit codes
EXIT_OK = 0
EXIT_EVAL_FAILED = 1
EXIT_CONFIG_ERROR = 2
EXIT_INTERNAL_ERROR = 3


logger = get_logger("arksim")

# Suppress various logging and warnings
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("a2a.client.card_resolver").setLevel(logging.WARNING)

_version_tag = f"v{__version__}" if "+" not in __version__ else "main"
_EXAMPLES_REPO_URL = (
    f"https://github.com/arklexai/arksim/archive/refs/tags/{_version_tag}.tar.gz"
)
_EXAMPLES_PREFIX = "examples/"


def _enforce_thresholds(
    evaluator_output: Evaluation, evaluation_input: EvaluationInput
) -> None:
    """Run all threshold gates and exit with EXIT_EVAL_FAILED if any fail."""
    threshold_ok = check_score_threshold(
        evaluator_output, evaluation_input.score_threshold
    )
    metric_ok = check_numeric_thresholds(
        evaluator_output, evaluation_input.numeric_thresholds
    )
    qual_ok = check_qualitative_failure_labels(
        evaluator_output, evaluation_input.qualitative_failure_labels
    )
    if not threshold_ok or not metric_ok or not qual_ok:
        sys.exit(EXIT_EVAL_FAILED)


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
            elif i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                # --key value format; single-dash values like -0.5 are treated as values
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


def _coerce_list_overrides(overrides: dict, model_cls: type) -> None:
    """Wrap scalar CLI values in lists for model fields that expect list types."""
    import types
    from typing import Union, get_args, get_origin

    for key in list(overrides):
        if key not in model_cls.model_fields:
            continue
        annotation = model_cls.model_fields[key].annotation
        # Unwrap Optional/Union (e.g. list[str] | None) to find the inner list type.
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            for arg in get_args(annotation):
                if get_origin(arg) is list or arg is list:
                    origin = list
                    break
        if origin is not list:
            continue
        val = overrides[key]
        # Comma-separated values are split into a list (e.g. "a.py,b.py" → ["a.py", "b.py"]).
        if isinstance(val, str):
            overrides[key] = [v.strip() for v in val.split(",")]
        elif not isinstance(val, list):
            overrides[key] = [val]


def validate_overrides(overrides: dict, valid_keys: set) -> None:
    invalid_keys = set(overrides.keys()) - valid_keys
    if invalid_keys:
        logger.error(f"Unknown options: {', '.join(sorted(invalid_keys))}")
        logger.info(f"Valid options: {', '.join(sorted(valid_keys))}")
        sys.exit(EXIT_CONFIG_ERROR)


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
        sys.exit(EXIT_CONFIG_ERROR)

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
    with urlopen(_EXAMPLES_REPO_URL) as resp:  # nosec B310
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
            sys.exit(EXIT_CONFIG_ERROR)

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
            sys.exit(EXIT_CONFIG_ERROR)

        for member in tar.getmembers():
            if member.name.startswith(filter_prefix):
                member.name = member.name.removeprefix(f"{top}/")
                # Block path traversal (CVE-2007-4559)
                if os.path.isabs(member.name) or ".." in member.name.split("/"):
                    continue
                if sys.version_info >= (3, 12):
                    tar.extract(member, ".", filter="data")
                else:
                    tar.extract(member, ".")

    logger.info(f"Downloaded to {os.path.abspath(dest_path)}")


def _add_config_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    """Add a subparser for config-based commands.

    These commands accept a YAML config file and
    arbitrary --key value overrides.
    """
    sp = subparsers.add_parser(name, help=help_text)
    sp.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to YAML config file",
    )
    sp.add_argument(
        "additional_args",
        nargs=argparse.REMAINDER,
        help="Additional --key value overrides",
    )
    return sp


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="arksim",
        description="⛵️ ArkSim - Know how your agent performs before it goes live.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\

            Examples:
              arksim --version
              arksim simulate config.yaml
              arksim evaluate config.yaml
              arksim simulate-evaluate config.yaml
              arksim show-prompts --category agent_behavior_failure
              arksim examples --list
              arksim examples bank-insurance
              arksim ui --port 9090
        """),
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    sub = parser.add_subparsers(dest="command")

    # Config-based commands (YAML + --key value overrides)
    _add_config_subparser(
        sub,
        "simulate",
        "Run agent simulations",
    )
    _add_config_subparser(
        sub,
        "evaluate",
        "Evaluate simulation results",
    )
    _add_config_subparser(
        sub,
        "simulate-evaluate",
        "Simulate then evaluate",
    )

    # show-prompts
    sp_prompts = sub.add_parser(
        "show-prompts",
        help="Display evaluation prompts",
    )
    sp_prompts.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter prompts by category",
    )

    # examples
    sp_examples = sub.add_parser(
        "examples",
        help="Download example projects from GitHub",
    )
    sp_examples.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Example to download (e.g. bank-insurance)",
    )
    sp_examples.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List available examples",
    )

    # ui
    sp_ui = sub.add_parser(
        "ui",
        help="Launch web UI control plane",
    )
    sp_ui.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve on (default: 8080)",
    )

    return parser


def _cmd_simulate(
    settings: dict,
    overrides: dict,
    config_path: str | None,
    cli_overrides: set,
    verbose: bool,
) -> None:
    """Validate config and run simulation."""
    try:
        valid_keys = set(SimulationInput.model_fields.keys())
        validate_overrides(overrides, valid_keys)
        _coerce_list_overrides(overrides, SimulationInput)
        settings = _merge_cli_overrides(settings, overrides)
        simulation_input = SimulationInput.model_validate(
            settings,
            context={"config_path": config_path, "cli_overrides": cli_overrides},
        )
        _log_config_summary("Simulation", simulation_input.model_dump())
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(EXIT_CONFIG_ERROR)

    asyncio.run(run_simulation(simulation_input, verbose=verbose))


def _cmd_evaluate(
    settings: dict,
    overrides: dict,
    config_path: str | None,
    cli_overrides: set,
) -> None:
    """Validate config and run evaluation."""
    try:
        valid_keys = set(EvaluationInput.model_fields.keys())
        validate_overrides(overrides, valid_keys)
        _coerce_list_overrides(overrides, EvaluationInput)
        settings = _merge_cli_overrides(settings, overrides)
        evaluation_input = EvaluationInput.model_validate(
            settings,
            context={"config_path": config_path, "cli_overrides": cli_overrides},
        )
        if not evaluation_input.simulation_file_path:
            logger.error("simulation_file_path is required.")
            sys.exit(EXIT_CONFIG_ERROR)
        if not os.path.isfile(evaluation_input.simulation_file_path):
            logger.error(
                f"simulation_file_path does not exist: "
                f"{evaluation_input.simulation_file_path}"
            )
            sys.exit(EXIT_CONFIG_ERROR)
        _log_config_summary("Evaluation", evaluation_input.model_dump())
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(EXIT_CONFIG_ERROR)

    evaluator_output = run_evaluation(evaluation_input)
    _enforce_thresholds(evaluator_output, evaluation_input)


def _cmd_simulate_evaluate(
    settings: dict,
    overrides: dict,
    config_path: str | None,
    cli_overrides: set,
    verbose: bool,
) -> None:
    """Validate config, run simulation, then run evaluation."""
    try:
        valid_keys = set(SimulationInput.model_fields.keys()) | set(
            EvaluationInput.model_fields.keys()
        )
        validate_overrides(overrides, valid_keys)
        _coerce_list_overrides(overrides, SimulationInput)
        _coerce_list_overrides(overrides, EvaluationInput)
        settings = _merge_cli_overrides(settings, overrides)

        simulation_settings = {
            k: v for k, v in settings.items() if k in SimulationInput.model_fields
        }
        simulation_input = SimulationInput.model_validate(
            simulation_settings,
            context={"config_path": config_path, "cli_overrides": cli_overrides},
        )
        _log_config_summary("Simulation", simulation_input.model_dump())

        evaluation_settings = {
            k: v for k, v in settings.items() if k in EvaluationInput.model_fields
        }
        evaluation_input = EvaluationInput.model_validate(
            evaluation_settings,
            context={"config_path": config_path, "cli_overrides": cli_overrides},
        )
        _log_config_summary("Evaluation", evaluation_input.model_dump())
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(EXIT_CONFIG_ERROR)

    sim_start = time.time()
    simulation_output = asyncio.run(run_simulation(simulation_input, verbose=verbose))
    logger.info(f"Simulation completed in {time.time() - sim_start:.2f} seconds")

    eval_start = time.time()
    evaluator_output = run_evaluation(evaluation_input, simulation=simulation_output)
    logger.info(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")

    _enforce_thresholds(evaluator_output, evaluation_input)


def main() -> None:
    """Main entry point for the arksim CLI."""
    s_time = time.time()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(EXIT_CONFIG_ERROR)

    if args.command == "show-prompts":
        _run_show_prompts(args.category)
        return

    if args.command == "examples":
        _run_examples(name=args.name, list_only=args.list_only)
        return

    if args.command == "ui":
        from arksim.ui.app import launch_ui

        launch_ui(port=args.port)
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
            sys.exit(EXIT_CONFIG_ERROR)
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

    config_path = os.path.abspath(args.config_file) if use_config_file else None
    cli_overrides = set(overrides.keys())

    try:
        if args.command == "simulate":
            _cmd_simulate(settings, overrides, config_path, cli_overrides, verbose)
        elif args.command == "evaluate":
            _cmd_evaluate(settings, overrides, config_path, cli_overrides)
        elif args.command == "simulate-evaluate":
            _cmd_simulate_evaluate(
                settings, overrides, config_path, cli_overrides, verbose
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Internal error: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(EXIT_INTERNAL_ERROR)

    logger.info(f"Total elapsed: {time.time() - s_time:.2f} seconds")


if __name__ == "__main__":
    main()
