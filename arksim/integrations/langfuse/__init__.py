# SPDX-License-Identifier: Apache-2.0
"""Export arksim simulations and evaluations to Langfuse.

This subpackage is optional. It requires the ``langfuse`` package
(``pip install arksim[langfuse]``). The heavy import is deferred so that
``import arksim.integrations.langfuse`` succeeds even when ``langfuse`` is
not installed; the ImportError is raised only when you actually build an
exporter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arksim.integrations.langfuse.exporter import (
        LangfuseExporter,
        export_to_langfuse,
    )

__all__ = ["LangfuseExporter", "export_to_langfuse"]


def __getattr__(name: str) -> object:
    """Lazily import the exporter so the optional dep loads only on use."""
    if name in __all__:
        from arksim.integrations.langfuse import exporter

        return getattr(exporter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
