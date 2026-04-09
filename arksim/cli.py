# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import textwrap
import time
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from arksim import __version__
from arksim.evaluator import (
    Evaluation,
    EvaluationInput,
    check_numeric_thresholds,
    check_qualitative_failure_labels,
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
    metric_ok = check_numeric_thresholds(
        evaluator_output, evaluation_input.numeric_thresholds
    )
    qual_ok = check_qualitative_failure_labels(
        evaluator_output, evaluation_input.qualitative_failure_labels
    )
    if not metric_ok or not qual_ok:
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


_INIT_AGENT_TYPES = ("custom", "chat_completions", "a2a")

_INIT_CONFIG_MAP = {
    "custom": "config.yaml",
    "chat_completions": "config_chat_completions.yaml",
    "a2a": "config_a2a.yaml",
}

_INIT_NEXT_STEPS = {
    "custom": (
        "\nNext steps:\n"
        "  1. Open my_agent.py and replace the execute() body with your agent logic\n"
        "  2. Run: arksim simulate-evaluate config.yaml"
    ),
    "chat_completions": (
        "\nNext steps:\n"
        "  1. Edit config.yaml with your agent's Chat Completions endpoint\n"
        "  2. Run: arksim simulate-evaluate config.yaml"
    ),
    "a2a": (
        "\nNext steps:\n"
        "  1. Edit config.yaml with your agent's A2A endpoint\n"
        "  2. Run: arksim simulate-evaluate config.yaml"
    ),
}


def _run_init(agent_type: str, force: bool = False) -> None:
    """Scaffold starter files for agent testing in the current directory.

    Generates a config.yaml, scenarios.json, and (for custom agent type)
    a starter my_agent.py. Files are copied from package templates.
    Existing files are not overwritten unless --force is passed.
    """
    cwd = os.getcwd()
    config_dest = os.path.join(cwd, "config.yaml")
    scenarios_dest = os.path.join(cwd, "scenarios.json")
    agent_dest = os.path.join(cwd, "my_agent.py")

    # Only check files that this agent type would write
    targets = [("config.yaml", config_dest), ("scenarios.json", scenarios_dest)]
    if agent_type == "custom":
        targets.append(("my_agent.py", agent_dest))

    if not force:
        existing = [name for name, path in targets if os.path.exists(path)]
        if existing:
            logger.error(
                f"{', '.join(existing)} already exist in the current directory. "
                "Use --force to overwrite, or run from a different directory."
            )
            sys.exit(EXIT_CONFIG_ERROR)

    templates = resources.files("arksim.templates")
    config_template = _INIT_CONFIG_MAP[agent_type]

    with resources.as_file(templates.joinpath(config_template)) as src:
        shutil.copy2(src, config_dest)
    with resources.as_file(templates.joinpath("scenarios.json")) as src:
        shutil.copy2(src, scenarios_dest)

    created = [config_dest, scenarios_dest]

    if agent_type == "custom":
        with resources.as_file(templates.joinpath("my_agent.py")) as src:
            shutil.copy2(src, agent_dest)
        created.append(agent_dest)

    for path in created:
        logger.info(f"Created {path}")

    logger.info(_INIT_NEXT_STEPS[agent_type])


# ============================================================================
# setup-claude - Install/uninstall Claude Code integration
# ============================================================================

_MCP_SERVER_CONFIG = {
    "command": sys.executable,
    "args": ["-m", "integrations.claude_code.mcp_server.server"],
}


def _find_integration_dir() -> Path:
    """Locate the integrations/claude_code directory.

    Checks the development layout first (relative to the arksim package),
    then falls back to importlib.resources for installed packages.
    """
    dev_path = Path(__file__).resolve().parent.parent / "integrations" / "claude_code"
    if dev_path.is_dir():
        return dev_path

    try:
        pkg_path = Path(str(resources.files("integrations.claude_code")))
        if pkg_path.is_dir():
            return pkg_path
    except (ModuleNotFoundError, TypeError):
        pass

    logger.error(
        "Could not locate integrations/claude_code directory. "
        "Ensure arksim is installed with the claude-code extra."
    )
    sys.exit(EXIT_CONFIG_ERROR)


def _run_setup_claude(
    project_dir: str = ".",
    force: bool = False,
    uninstall: bool = False,
) -> None:
    """Install or uninstall the Claude Code integration for a project.

    Install mode (default):
      - Creates/merges .claude/settings.json with an mcpServers.arksim entry
      - Copies skill directories from integrations/claude_code/skills/arksim-*/
        to .claude/skills/arksim-*/

    Uninstall mode:
      - Removes arksim from mcpServers in settings.json
      - Removes .claude/skills/arksim-*/ directories
    """
    root = Path(project_dir).resolve()
    claude_dir = root / ".claude"
    settings_path = claude_dir / "settings.json"
    skills_dir = claude_dir / "skills"

    if uninstall:
        _uninstall_claude(settings_path, skills_dir)
        return

    integration_dir = _find_integration_dir()
    _install_claude(integration_dir, claude_dir, settings_path, skills_dir, force)


def _uninstall_claude(settings_path: Path, skills_dir: Path) -> None:
    """Remove arksim MCP config and skills from a project."""
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON in {settings_path}. Fix or delete the file and retry."
            )
            sys.exit(EXIT_CONFIG_ERROR)
        mcp_servers = settings.get("mcpServers", {})
        mcp_servers.pop("arksim", None)
        if not mcp_servers:
            settings.pop("mcpServers", None)
        else:
            settings["mcpServers"] = mcp_servers
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # Remove all arksim-* skill directories
    if skills_dir.is_dir():
        for skill_dir in sorted(skills_dir.glob("arksim-*")):
            if skill_dir.is_dir():
                shutil.rmtree(skill_dir)

    logger.info("Removed arksim Claude Code integration.")


def _install_claude(
    integration_dir: Path,
    claude_dir: Path,
    settings_path: Path,
    skills_dir: Path,
    force: bool,
) -> None:
    """Install arksim MCP config and skills into a project."""
    # Check for existing arksim skills
    existing_skills = list(skills_dir.glob("arksim-*")) if skills_dir.is_dir() else []
    if existing_skills and not force:
        logger.error(
            "arksim skills already installed. "
            "Use --force to overwrite, or --uninstall to remove first."
        )
        sys.exit(EXIT_CONFIG_ERROR)

    # Merge settings.json
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON in {settings_path}. Fix or delete the file and retry."
            )
            sys.exit(EXIT_CONFIG_ERROR)

    mcp_servers = settings.setdefault("mcpServers", {})
    mcp_servers["arksim"] = _MCP_SERVER_CONFIG
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # Copy skills (each skill is a directory with SKILL.md)
    skills_src = integration_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing arksim skills if --force
    for old_skill in skills_dir.glob("arksim-*"):
        if old_skill.is_dir():
            shutil.rmtree(old_skill)

    copied = []
    for skill_dir in sorted(skills_src.iterdir()):
        if not skill_dir.is_dir() or not skill_dir.name.startswith("arksim-"):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        dest_dir = skills_dir / skill_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(skill_md, dest_dir / "SKILL.md")
        copied.append(dest_dir / "SKILL.md")

    if not copied:
        logger.error(
            f"No skill directories found in {skills_src}. Installation may be incomplete."
        )
        sys.exit(EXIT_CONFIG_ERROR)

    logger.info("Installed arksim Claude Code integration:")
    logger.info(f"  Settings: {settings_path}")
    for path in copied:
        logger.info(f"  Skill:    {path}")
    logger.info('\nStart Claude Code and type "/arksim:test" to begin.')


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

    # init
    sp_init = sub.add_parser(
        "init",
        help="Scaffold starter files for agent testing",
    )
    sp_init.add_argument(
        "--agent-type",
        type=str,
        choices=_INIT_AGENT_TYPES,
        default="custom",
        dest="agent_type",
        help=(
            "Agent connection type (default: custom). "
            "'custom' generates a Python agent file (no server needed). "
            "'chat_completions' for HTTP endpoints. "
            "'a2a' for Agent-to-Agent protocol."
        ),
    )
    sp_init.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing files",
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

    # setup-claude
    sp_setup_claude = sub.add_parser(
        "setup-claude",
        help="Install Claude Code integration (MCP server + skills)",
    )
    sp_setup_claude.add_argument(
        "--project-dir",
        type=str,
        default=".",
        dest="project_dir",
        help="Project root directory (default: current directory)",
    )
    sp_setup_claude.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing skills",
    )
    sp_setup_claude.add_argument(
        "--uninstall",
        action="store_true",
        default=False,
        help="Remove arksim Claude Code integration",
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

    if args.command == "init":
        _run_init(agent_type=args.agent_type, force=args.force)
        return

    if args.command == "ui":
        from arksim.ui.app import launch_ui

        launch_ui(port=args.port)
        return

    if args.command == "setup-claude":
        _run_setup_claude(
            project_dir=args.project_dir,
            force=args.force,
            uninstall=args.uninstall,
        )
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
        raise  # this preserves exit codes for CI
    except Exception as e:
        logger.error(f"Internal error: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(EXIT_INTERNAL_ERROR)

    logger.info(f"Total elapsed: {time.time() - s_time:.2f} seconds")


if __name__ == "__main__":
    main()
