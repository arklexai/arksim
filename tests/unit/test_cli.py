# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument parsing and utility functions."""

import io
import os
import sys
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from arksim.cli import (
    _check_score_threshold,
    _merge_cli_overrides,
    _parse_value,
    _run_examples,
    build_parser,
    parse_extra_args,
    validate_overrides,
)

# ── _parse_value ─────────────────────────────────────────


class TestParseValue:
    """Tests for _parse_value type coercion."""

    def test_true_strings(self) -> None:
        """Parses 'true' and 'yes' as True."""
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True
        assert _parse_value("YES") is True

    def test_false_strings(self) -> None:
        """Parses 'false' and 'no' as False."""
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False
        assert _parse_value("NO") is False

    def test_integer(self) -> None:
        """Parses integer strings."""
        assert _parse_value("42") == 42
        assert _parse_value("0") == 0
        assert _parse_value("-1") == -1

    def test_float(self) -> None:
        """Parses float strings."""
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.5") == 0.5

    def test_string_fallback(self) -> None:
        """Returns string when no other type matches."""
        assert _parse_value("hello") == "hello"
        assert _parse_value("./path/to/file") == "./path/to/file"


# ── parse_extra_args ─────────────────────────────────────


class TestParseExtraArgs:
    """Tests for parse_extra_args CLI override parser."""

    def test_empty_list(self) -> None:
        """Empty input returns empty dict."""
        assert parse_extra_args([]) == {}

    def test_key_value_pair(self) -> None:
        """Parses --key value format."""
        result = parse_extra_args(["--model", "gpt-4"])
        assert result == {"model": "gpt-4"}

    def test_key_equals_value(self) -> None:
        """Parses --key=value format."""
        result = parse_extra_args(["--seed=42"])
        assert result == {"seed": 42}

    def test_boolean_flag(self) -> None:
        """Bare --flag is treated as True."""
        result = parse_extra_args(["--verbose"])
        assert result == {"verbose": True}

    def test_dash_to_underscore(self) -> None:
        """Dashes in key names are converted to underscores."""
        result = parse_extra_args(["--max-turns", "5"])
        assert result == {"max_turns": 5}

    def test_multiple_args(self) -> None:
        """Parses multiple arguments together."""
        result = parse_extra_args(
            ["--model", "gpt-4", "--max-turns", "10", "--verbose"]
        )
        assert result == {"model": "gpt-4", "max_turns": 10, "verbose": True}

    def test_type_coercion(self) -> None:
        """Values are coerced to appropriate types."""
        result = parse_extra_args(["--count", "5", "--rate", "0.5", "--flag", "true"])
        assert result == {"count": 5, "rate": 0.5, "flag": True}

    def test_skips_non_flag_args(self) -> None:
        """Non-flag arguments are silently skipped."""
        result = parse_extra_args(["positional", "--key", "val"])
        assert result == {"key": "val"}


# ── _merge_cli_overrides ────────────────────────────────


class TestMergeCliOverrides:
    """Tests for _merge_cli_overrides."""

    def test_override_replaces_yaml(self) -> None:
        """CLI values override YAML values."""
        yaml = {"model": "gpt-3.5", "max_turns": 5}
        cli = {"model": "gpt-4"}
        result = _merge_cli_overrides(yaml, cli)
        assert result["model"] == "gpt-4"
        assert result["max_turns"] == 5

    def test_none_values_skipped(self) -> None:
        """None values in overrides are ignored."""
        yaml = {"model": "gpt-3.5"}
        cli = {"model": None}
        result = _merge_cli_overrides(yaml, cli)
        assert result["model"] == "gpt-3.5"

    def test_new_keys_added(self) -> None:
        """New keys from CLI are added to result."""
        yaml = {"model": "gpt-3.5"}
        cli = {"verbose": True}
        result = _merge_cli_overrides(yaml, cli)
        assert result == {"model": "gpt-3.5", "verbose": True}

    def test_original_not_mutated(self) -> None:
        """Original YAML dict is not modified."""
        yaml = {"model": "gpt-3.5"}
        cli = {"model": "gpt-4"}
        _merge_cli_overrides(yaml, cli)
        assert yaml["model"] == "gpt-3.5"


# ── validate_overrides ──────────────────────────────────


class TestValidateOverrides:
    """Tests for validate_overrides."""

    def test_valid_keys_pass(self) -> None:
        """No exit when all keys are valid."""
        validate_overrides({"model": "gpt-4"}, {"model", "provider"})

    def test_invalid_keys_exit(self) -> None:
        """Exits with code 1 when unknown keys are present."""
        with pytest.raises(SystemExit) as exc_info:
            validate_overrides({"bad_key": "val"}, {"model"})
        assert exc_info.value.code == 1

    def test_empty_overrides_pass(self) -> None:
        """Empty overrides always pass."""
        validate_overrides({}, {"model"})


# ── _check_score_threshold ──────────────────────────────


class TestCheckScoreThreshold:
    """Tests for _check_score_threshold."""

    @staticmethod
    def _make_evaluation(scores: list[float]) -> MagicMock:
        """Build a mock Evaluation with given scores."""
        convos = []
        for i, score in enumerate(scores):
            c = MagicMock()
            c.conversation_id = f"convo-{i}"
            c.overall_agent_score = score
            convos.append(c)
        ev = MagicMock()
        ev.conversations = convos
        return ev

    def test_none_threshold_passes(self) -> None:
        """None threshold always returns True."""
        ev = self._make_evaluation([0.1])
        assert _check_score_threshold(ev, None) is True

    def test_all_above_threshold(self) -> None:
        """Returns True when all scores pass."""
        ev = self._make_evaluation([0.9, 0.8, 0.85])
        assert _check_score_threshold(ev, 0.7) is True

    def test_some_below_threshold(self) -> None:
        """Returns False when any score is below threshold."""
        ev = self._make_evaluation([0.9, 0.3, 0.8])
        assert _check_score_threshold(ev, 0.5) is False

    def test_exact_threshold_passes(self) -> None:
        """Score equal to threshold passes."""
        ev = self._make_evaluation([0.7])
        assert _check_score_threshold(ev, 0.7) is True


# ── build_parser ────────────────────────────────────────


class TestBuildParser:
    """Tests for build_parser subcommand structure."""

    def test_all_subcommands_registered(self) -> None:
        """Parser has all expected subcommands."""
        parser = build_parser()
        # Parse each command to verify it's registered
        for cmd in [
            "simulate",
            "evaluate",
            "simulate-evaluate",
            "show-prompts",
            "examples",
            "ui",
        ]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_simulate_with_config_and_overrides(self) -> None:
        """Simulate parses config_file and REMAINDER overrides."""
        parser = build_parser()
        args = parser.parse_args(["simulate", "config.yaml", "--max-turns", "3"])
        assert args.command == "simulate"
        assert args.config_file == "config.yaml"
        assert args.additional_args == ["--max-turns", "3"]

    def test_simulate_without_config(self) -> None:
        """Simulate with no config sets config_file to None."""
        parser = build_parser()
        args = parser.parse_args(["simulate"])
        assert args.config_file is None

    def test_examples_list_flag(self) -> None:
        """Examples --list sets list_only to True."""
        parser = build_parser()
        args = parser.parse_args(["examples", "--list"])
        assert args.command == "examples"
        assert args.list_only is True
        assert args.name is None

    def test_examples_with_name(self) -> None:
        """Examples with a name sets args.name."""
        parser = build_parser()
        args = parser.parse_args(["examples", "bank-insurance"])
        assert args.name == "bank-insurance"
        assert args.list_only is False

    def test_show_prompts_category(self) -> None:
        """Show-prompts --category sets args.category."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts", "--category", "faithfulness"])
        assert args.category == "faithfulness"

    def test_show_prompts_default_category(self) -> None:
        """Show-prompts without --category defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts"])
        assert args.category is None

    def test_ui_custom_port(self) -> None:
        """UI --port sets custom port."""
        parser = build_parser()
        args = parser.parse_args(["ui", "--port", "9090"])
        assert args.port == 9090

    def test_ui_default_port(self) -> None:
        """UI without --port defaults to 8080."""
        parser = build_parser()
        args = parser.parse_args(["ui"])
        assert args.port == 8080


# ── _run_examples ───────────────────────────────────────


def _make_tarball(members: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz with given name→content entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestRunExamples:
    """Tests for _run_examples."""

    @pytest.fixture()
    def fake_tarball(self) -> bytes:
        """A minimal tarball mimicking the GitHub archive."""
        return _make_tarball(
            {
                "arksim-main/examples/demo/config.yaml": "model: gpt-4",
                "arksim-main/examples/demo/scenario.json": "[]",
                "arksim-main/examples/bank-insurance/config.yaml": "model: gpt-4",
                "arksim-main/README.md": "# arksim",
            }
        )

    def test_list_only(
        self, fake_tarball: bytes, capsys: pytest.CaptureFixture
    ) -> None:
        """--list prints available example names."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            _run_examples(list_only=True)

        out = capsys.readouterr().out
        assert "demo" in out
        assert "bank-insurance" in out

    def test_unknown_name_exits(self, fake_tarball: bytes) -> None:
        """Unknown example name exits with code 1."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(SystemExit) as exc_info:
                _run_examples(name="nonexistent")
            assert exc_info.value.code == 1

    def test_dest_exists_exits(self, fake_tarball: bytes, temp_dir: str) -> None:
        """Exits when destination already exists."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        # Create the destination so it already exists
        dest = os.path.join(temp_dir, "examples")
        os.makedirs(dest)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_examples()
        assert exc_info.value.code == 1

    def test_path_traversal_skipped(self, temp_dir: str) -> None:
        """Members with '..' or absolute paths are skipped."""
        tarball = _make_tarball(
            {
                "arksim-main/examples/demo/safe.txt": "ok",
                "arksim-main/examples/demo/../../etc/passwd": "evil",
                "arksim-main/examples/demo//tmp/evil": "evil",
            }
        )
        resp = MagicMock()
        resp.read.return_value = tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=False),
            patch("tarfile.TarFile.extract") as mock_extract,
        ):
            _run_examples(name="demo")

        # Only the safe member should have been extracted
        extracted_names = [call.args[0].name for call in mock_extract.call_args_list]
        for name in extracted_names:
            assert ".." not in name.split("/")
            assert not os.path.isabs(name)

        # Verify filter="data" is passed on 3.12+ and omitted on older versions
        for call in mock_extract.call_args_list:
            if sys.version_info >= (3, 12):
                assert call.kwargs.get("filter") == "data"
            else:
                assert "filter" not in call.kwargs
