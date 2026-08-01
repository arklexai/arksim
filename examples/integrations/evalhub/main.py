# SPDX-License-Identifier: Apache-2.0
"""Container entrypoint. EvalHub runs ``python main.py`` inside the job pod."""

from __future__ import annotations

from arksim_evalhub import main

if __name__ == "__main__":
    main()
