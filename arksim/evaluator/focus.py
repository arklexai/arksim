# SPDX-License-Identifier: Apache-2.0
"""Generate focused scenario files for rerunning failed error groups.

After evaluation, this module joins UniqueError occurrences back to their
originating scenarios and writes filtered Scenarios JSON files that plug
directly into ``arksim simulate-evaluate --scenario_file_path``.
"""

from __future__ import annotations

import logging

from arksim.evaluator.entities import UniqueError

logger = logging.getLogger(__name__)


def build_error_scenario_map(
    unique_errors: list[UniqueError],
    conv_to_scenario: dict[str, str],
) -> dict[str, set[str]]:
    """Map each unique error to the set of scenario IDs that triggered it.

    Args:
        unique_errors: Errors detected by the evaluator.
        conv_to_scenario: Mapping of conversation_id to scenario_id,
            built from ``Simulation.conversations``.

    Returns:
        Dict of ``{unique_error_id: {scenario_id, ...}}``.
    """
    error_to_scenarios: dict[str, set[str]] = {}
    for error in unique_errors:
        scenario_ids: set[str] = set()
        for occ in error.occurrences:
            sid = conv_to_scenario.get(occ.conversation_id)
            if sid is None:
                logger.debug(
                    "Skipping occurrence with unknown conversation_id=%s",
                    occ.conversation_id,
                )
                continue
            scenario_ids.add(sid)
        if not scenario_ids:
            logger.warning(
                "Dropping error %s: all %d occurrence(s) have unknown conversation_ids",
                error.unique_error_id,
                len(error.occurrences),
            )
            continue
        error_to_scenarios[error.unique_error_id] = scenario_ids
    return error_to_scenarios
