# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.simulation_engine.agent.utils (_parse_retry_after, rate_limit_handler)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arksim.simulation_engine.agent.utils import (
    _parse_retry_after,
    rate_limit_handler,
)


class TestParseRetryAfter:
    def test_valid_integer_string(self) -> None:
        assert _parse_retry_after("10") == 10

    def test_invalid_string_returns_default(self) -> None:
        assert _parse_retry_after("Thu, 01 Jan 2099 00:00:00 GMT") == 5

    def test_custom_default(self) -> None:
        assert _parse_retry_after("bad", default=30) == 30


class TestRateLimitHandlerSync:
    @patch("arksim.simulation_engine.agent.utils.time.sleep")
    def test_success_passthrough(self, mock_sleep: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200

        @rate_limit_handler
        def fn() -> MagicMock:
            return response

        result = fn()
        assert result is response
        response.raise_for_status.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("arksim.simulation_engine.agent.utils.time.sleep")
    def test_429_retries_then_succeeds(self, mock_sleep: MagicMock) -> None:
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        ok_response = MagicMock()
        ok_response.status_code = 200

        call_count = 0

        @rate_limit_handler
        def fn() -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return rate_limited
            return ok_response

        result = fn()
        assert result is ok_response
        assert mock_sleep.call_count == 2

    @patch("arksim.simulation_engine.agent.utils.time.sleep")
    def test_429_exhausts_retries(self, mock_sleep: MagicMock) -> None:
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        @rate_limit_handler
        def fn() -> MagicMock:
            return rate_limited

        fn()
        rate_limited.raise_for_status.assert_called()


class TestRateLimitHandlerAsync:
    @pytest.mark.asyncio
    @patch("arksim.simulation_engine.agent.utils.asyncio.sleep")
    async def test_success_passthrough(self, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None
        response = MagicMock()
        response.status_code = 200

        @rate_limit_handler
        async def fn() -> MagicMock:
            return response

        result = await fn()
        assert result is response
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("arksim.simulation_engine.agent.utils.asyncio.sleep")
    async def test_429_retries(self, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "2"}

        ok_response = MagicMock()
        ok_response.status_code = 200

        call_count = 0

        @rate_limit_handler
        async def fn() -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return rate_limited
            return ok_response

        result = await fn()
        assert result is ok_response
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2)
