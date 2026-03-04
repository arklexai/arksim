# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.llms.chat.utils (retry decorator)."""

from __future__ import annotations

from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

from arksim.llms.chat.utils import retry


class TestRetrySyncWrapper:
    def test_success_first_try(self) -> None:
        @retry(max_retries=3)
        def fn() -> str:
            return "ok"

        assert fn() == "ok"

    @patch("arksim.llms.chat.utils.time.sleep")
    def test_success_after_retries(self, mock_sleep: MagicMock) -> None:
        call_count = 0

        @retry(max_retries=3)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "recovered"

        assert fn() == "recovered"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("arksim.llms.chat.utils.time.sleep")
    def test_max_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        @retry(max_retries=2)
        def fn() -> NoReturn:
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            fn()
        assert mock_sleep.call_count == 1

    @patch("arksim.llms.chat.utils.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        @retry(max_retries=4)
        def fn() -> NoReturn:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fn()

        delays = [c[0][0] for c in mock_sleep.call_args_list]
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0


class TestRetryAsyncWrapper:
    @pytest.mark.asyncio
    async def test_success_first_try(self) -> None:
        @retry(max_retries=3)
        async def fn() -> str:
            return "ok"

        assert await fn() == "ok"

    @pytest.mark.asyncio
    @patch("arksim.llms.chat.utils.asyncio.sleep")
    async def test_success_after_retries(self, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None
        call_count = 0

        @retry(max_retries=3)
        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "recovered"

        assert await fn() == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("arksim.llms.chat.utils.asyncio.sleep")
    async def test_max_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None

        @retry(max_retries=2)
        async def fn() -> NoReturn:
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await fn()

    def test_preserves_function_name(self) -> None:
        @retry(max_retries=2)
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    @pytest.mark.asyncio
    async def test_preserves_async_function_name(self) -> None:
        @retry(max_retries=2)
        async def my_async_func() -> None:
            pass

        assert my_async_func.__name__ == "my_async_func"
