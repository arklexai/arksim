# SPDX-License-Identifier: Apache-2.0
"""Tests for __init__.py version fallback."""

from __future__ import annotations

import importlib
import sys


class TestVersionFallback:
    def test_version_fallback_on_import_error(self) -> None:
        # Remove _version module if cached, and make it unimportable
        saved = sys.modules.pop("arksim._version", None)
        saved_arksim = sys.modules.pop("arksim", None)
        try:
            # Insert a loader that raises ImportError for arksim._version
            sys.modules["arksim._version"] = None  # type: ignore[assignment]
            import arksim

            importlib.reload(arksim)
            assert arksim.__version__ == "0.0.0+unknown"
        finally:
            # Restore original state
            sys.modules.pop("arksim._version", None)
            sys.modules.pop("arksim", None)
            if saved is not None:
                sys.modules["arksim._version"] = saved
            if saved_arksim is not None:
                sys.modules["arksim"] = saved_arksim
            else:
                import arksim  # noqa: F811

                importlib.reload(arksim)
