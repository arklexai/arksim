# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Claude Code MCP server security helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from integrations.claude_code.mcp_server.security import (
    MAX_CAPTURED_BYTES,
    PathValidationError,
    is_inside,
    redact_secrets,
    tail_capture,
    validate_path_arg,
)

# ── redact_secrets ──────────────────────────────────────────


class TestRedactSecretsEmpty:
    def test_empty_string_returns_empty(self) -> None:
        assert redact_secrets("") == ""

    def test_no_match_returns_unchanged(self) -> None:
        assert redact_secrets("hello world") == "hello world"

    def test_short_id_not_redacted(self) -> None:
        # Scenario IDs and similar short tokens must survive.
        assert redact_secrets("scenario_id=abc-123") == "scenario_id=abc-123"


class TestRedactSecretsLLMProviders:
    def test_redacts_openai_sk(self) -> None:
        out = redact_secrets("token=sk-proj-abcdef0123456789ABCDEFXX")
        assert "sk-proj-" not in out
        assert "[REDACTED]" in out

    def test_redacts_anthropic_sk_ant(self) -> None:
        out = redact_secrets("auth=sk-ant-abcdef0123456789ABCDEFXX")
        assert "sk-ant-" not in out
        assert "[REDACTED]" in out

    def test_redacts_google_aiza(self) -> None:
        out = redact_secrets("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012")
        assert "AIza" not in out
        assert "[REDACTED]" in out


class TestRedactSecretsCloudProviders:
    def test_redacts_aws_access_key(self) -> None:
        out = redact_secrets("aws_key=AKIAIOSFODNN7EXAMPLE")
        assert "AKIA" not in out
        assert "[REDACTED]" in out

    def test_redacts_github_pat(self) -> None:
        out = redact_secrets("token=ghp_abcdefghijklmnopqrstuvwxyz0123456789")
        assert "ghp_" not in out
        assert "[REDACTED]" in out

    def test_redacts_github_pat_v2(self) -> None:
        out = redact_secrets("token=github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef")
        assert "github_pat_" not in out
        assert "[REDACTED]" in out

    def test_redacts_slack_bot_token(self) -> None:
        out = redact_secrets("token=xoxb-1234567890-ABCDEFGHIJKL")
        assert "xoxb-" not in out
        assert "[REDACTED]" in out

    def test_redacts_huggingface_token(self) -> None:
        out = redact_secrets("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456")
        assert "hf_" not in out
        assert "[REDACTED]" in out

    def test_redacts_stripe_live_key(self) -> None:
        # Split the literal so push-protection scanners don't flag the
        # fixture; runtime string is unchanged.
        out = redact_secrets("STRIPE=sk_live" + "_abcdefghijklmnopABCDEFGH")
        assert "sk_live_" not in out
        assert "[REDACTED]" in out


class TestRedactSecretsHeaders:
    def test_redacts_authorization_bearer(self) -> None:
        out = redact_secrets("Authorization: Bearer abc.def.ghi-jkl_mno=")
        assert "Bearer" not in out
        assert "[REDACTED]" in out

    def test_redacts_x_api_key(self) -> None:
        out = redact_secrets("x-api-key: somesecret123")
        assert "somesecret123" not in out
        assert "[REDACTED]" in out


class TestRedactSecretsEnvShape:
    def test_redacts_provider_env(self) -> None:
        out = redact_secrets("OPENAI_API_KEY=sk-thisisalongkey1234567890")
        assert "[REDACTED]" in out

    def test_redacts_generic_token_env(self) -> None:
        out = redact_secrets("MY_SERVICE_TOKEN=abc123xyz456789")
        assert "[REDACTED]" in out

    def test_redacts_secret_env(self) -> None:
        out = redact_secrets("APP_SECRET=topsecretvalue1234567")
        assert "[REDACTED]" in out

    def test_does_not_swallow_json_tail(self) -> None:
        # Trailing JSON tokens like `"}, "next":` must not be eaten by
        # the redaction regex.
        out = redact_secrets('OPENAI_API_KEY=sk-foo1234567890abcdef", "next": null')
        assert "next" in out


class TestRedactSecretsJWT:
    def test_redacts_jwt(self) -> None:
        out = redact_secrets(
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        assert "[REDACTED]" in out


# ── tail_capture ─────────────────────────────────────────────


class TestTailCapture:
    def test_empty_returns_empty(self) -> None:
        assert tail_capture("") == ""

    def test_under_limit_unchanged(self) -> None:
        small = "hello\n" * 100
        assert tail_capture(small) == small

    def test_over_limit_truncates(self) -> None:
        chunk = "abcdefghij" * 50_000  # 500 KB
        out = tail_capture(chunk)
        assert "[truncated]" in out
        assert len(out.encode("utf-8")) <= MAX_CAPTURED_BYTES + 50

    def test_custom_max_bytes(self) -> None:
        text = "x" * 1024
        out = tail_capture(text, max_bytes=100)
        assert "[truncated]" in out


# ── validate_path_arg ───────────────────────────────────────


class TestValidatePathArgNone:
    def test_none_with_allow_none(self) -> None:
        assert validate_path_arg(None, allow_none=True) is None

    def test_none_without_allow_none_raises(self) -> None:
        with pytest.raises(PathValidationError):
            validate_path_arg(None, allow_none=False)


class TestValidatePathArgRejections:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(PathValidationError):
            validate_path_arg("")

    def test_nul_byte_raises(self) -> None:
        with pytest.raises(PathValidationError, match="NUL"):
            validate_path_arg("foo\x00bar")

    def test_filesystem_root_raises(self) -> None:
        with pytest.raises(PathValidationError, match="root"):
            validate_path_arg("/", require_exists=False)

    def test_home_directory_raises(self) -> None:
        home = str(Path.home())
        with pytest.raises(PathValidationError, match="home"):
            validate_path_arg(home, require_exists=False)


class TestValidatePathArgRequirements:
    def test_nonexistent_with_require_exists_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="does not exist"):
            validate_path_arg(
                str(tmp_path / "missing"),
                require_exists=True,
            )

    def test_nonexistent_without_require_exists_passes(self, tmp_path: Path) -> None:
        result = validate_path_arg(
            str(tmp_path / "missing"),
            require_exists=False,
        )
        assert isinstance(result, Path)

    def test_file_when_require_dir_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "real.txt"
        f.write_text("x")
        with pytest.raises(PathValidationError, match="not a directory"):
            validate_path_arg(str(f), require_dir=True)

    def test_dir_when_require_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="not a file"):
            validate_path_arg(str(tmp_path), require_file=True)


class TestValidatePathArgValid:
    def test_existing_dir_returns_resolved_path(self, tmp_path: Path) -> None:
        result = validate_path_arg(
            str(tmp_path),
            require_exists=True,
            require_dir=True,
        )
        assert result == tmp_path.resolve()

    def test_subdir_of_home_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Subdirectories of $HOME are explicitly allowed.
        sub = tmp_path / "project"
        sub.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = validate_path_arg(str(sub), require_dir=True)
        assert result == sub.resolve()


# ── is_inside ─────────────────────────────────────────────


class TestIsInside:
    def test_path_inside_root(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        assert is_inside(sub, tmp_path) is True

    def test_path_outside_root(self, tmp_path: Path) -> None:
        other = tmp_path.parent
        assert is_inside(other, tmp_path) is False

    def test_root_is_inside_itself(self, tmp_path: Path) -> None:
        assert is_inside(tmp_path, tmp_path) is True
