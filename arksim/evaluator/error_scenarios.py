# SPDX-License-Identifier: Apache-2.0
"""Map unique errors to the scenarios that triggered them.

Pure computation module with no file I/O.  Takes UniqueError occurrences
and a conversation-to-scenario mapping, and returns typed
ErrorScenarioMapping objects ready for serialization into
``evaluation.json`` or for writing focus files via ``Evaluator.save_results``.
"""

from __future__ import annotations

import logging

from arksim.evaluator.entities import ErrorScenarioMapping, UniqueError
from arksim.evaluator.utils.constants import SEVERITY_RANK
from arksim.scenario.entities import Scenario, Scenarios

logger = logging.getLogger(__name__)


def _build_error_scenario_map(
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


def _sort_key(error: UniqueError) -> tuple[int, int]:
    """Return a sort key for severity-first, then descending occurrence count."""
    severity_rank = SEVERITY_RANK.get(error.severity, len(SEVERITY_RANK))
    return (severity_rank, -len(error.occurrences))


def build_error_scenario_data(
    unique_errors: list[UniqueError],
    conv_to_scenario: dict[str, str],
    scenarios: Scenarios,
) -> tuple[list[ErrorScenarioMapping], list[Scenario]]:
    """Compute error-to-scenario mappings and the combined failure set.

    Pure function with no I/O.  Returns the mappings and the union of
    all failing scenarios (deduplicated, sorted by ID).

    Args:
        unique_errors: Errors detected by the evaluator.
        conv_to_scenario: Mapping of conversation_id to scenario_id.
        scenarios: Full scenario set from the simulation run.

    Returns:
        Tuple of (mappings, all_failing_scenarios).  Both lists are empty
        when there are no errors or no matching scenarios.
    """
    if not unique_errors:
        return [], []

    error_scenario_map = _build_error_scenario_map(unique_errors, conv_to_scenario)
    scenario_lookup: dict[str, Scenario] = {
        s.scenario_id: s for s in scenarios.scenarios
    }

    sorted_errors = sorted(unique_errors, key=_sort_key)

    mappings: list[ErrorScenarioMapping] = []
    all_scenario_ids: set[str] = set()
    error_index = 0

    for error in sorted_errors:
        mapped_ids = error_scenario_map.get(error.unique_error_id)
        if mapped_ids is None:
            continue

        matched: list[Scenario] = []
        for sid in sorted(mapped_ids):
            scenario = scenario_lookup.get(sid)
            if scenario is None:
                logger.debug(
                    "Scenario %s referenced by error %s not found in scenario file; skipping",
                    sid,
                    error.unique_error_id,
                )
                continue
            matched.append(scenario)

        if not matched:
            logger.debug(
                "Skipping error %s: no matching scenarios found in scenario file",
                error.unique_error_id,
            )
            continue

        error_index += 1
        matched_ids = [s.scenario_id for s in matched]
        all_scenario_ids.update(matched_ids)
        mappings.append(
            ErrorScenarioMapping(
                error_index=error_index,
                unique_error_id=error.unique_error_id,
                error_description=error.unique_error_description,
                severity=error.severity,
                scenario_ids=matched_ids,
            )
        )

    if not mappings:
        return [], []

    all_scenarios = [
        scenario_lookup[sid]
        for sid in sorted(all_scenario_ids)
        if sid in scenario_lookup
    ]

    return mappings, all_scenarios
