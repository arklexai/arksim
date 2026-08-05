# SPDX-License-Identifier: Apache-2.0
"""Built-in speech providers, imported for registration side effects.

Guarded so importing ``arksim.speech`` never fails when the optional voice
extra (``pip install 'arksim[voice]'``) is not installed.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from arksim.speech.providers import faster_whisper, kokoro  # noqa: F401
