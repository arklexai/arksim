# SPDX-License-Identifier: Apache-2.0
"""Generate focused scenario files for rerunning failed error groups.

After evaluation, this module joins UniqueError occurrences back to their
originating scenarios and writes filtered Scenarios JSON files that plug
directly into ``arksim simulate-evaluate --scenario_file_path``.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel

from arksim.evaluator.entities import UniqueError
from arksim.evaluator.utils.constants import SEVERITY_RANK
from arksim.scenario.entities import Scenario, Scenarios
from arksim.utils.output import save_json_file

logger = logging.getLogger(__name__)


class FocusFileInfo(BaseModel):
    """Metadata about a generated focus file."""

    error_index: int
    unique_error_id: str
    error_description: str
    severity: str
    scenario_ids: list[str]
    file_path: str


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


def generate_focus_files(
    unique_errors: list[UniqueError],
    conv_to_scenario: dict[str, str],
    scenarios: Scenarios,
    output_dir: str,
) -> list[FocusFileInfo]:
    """Write per-error and combined scenario files for targeted reruns.

    For each unique error that maps to known scenarios, writes a filtered
    ``Scenarios`` JSON file under ``<output_dir>/focus/error_N.json``.
    Also writes ``<output_dir>/focus/all_failures.json`` with the union of
    all failing scenarios.  Files are ordered by severity (critical first)
    then by descending occurrence count.

    Args:
        unique_errors: Errors detected by the evaluator.
        conv_to_scenario: Mapping of conversation_id to scenario_id.
        scenarios: Full scenario set from the simulation run.
        output_dir: Root output directory (``focus/`` sub-dir is created).

    Returns:
        One :class:`FocusFileInfo` per error group that had matching
        scenarios, in the same order as the written files.  Returns an
        empty list when there are no errors.
    """
    if not unique_errors:
        return []

    error_scenario_map = _build_error_scenario_map(unique_errors, conv_to_scenario)
    scenario_lookup: dict[str, Scenario] = {
        s.scenario_id: s for s in scenarios.scenarios
    }

    sorted_errors = sorted(unique_errors, key=_sort_key)

    focus_dir = os.path.join(output_dir, "focus")
    results: list[FocusFileInfo] = []
    all_scenario_ids: set[str] = set()
    error_index = 0

    for error in sorted_errors:
        mapped_ids = error_scenario_map.get(error.unique_error_id)
        if mapped_ids is None:
            # All occurrences had unknown conv_ids; already warned in _build_error_scenario_map
            continue

        matched: list[Scenario] = []
        for sid in mapped_ids:
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
        file_path = os.path.join(focus_dir, f"error_{error_index}.json")
        filtered = Scenarios(
            schema_version=scenarios.schema_version,
            scenarios=matched,
        )
        save_json_file(filtered.model_dump(), file_path, overwrite=True)

        matched_ids = [s.scenario_id for s in matched]
        all_scenario_ids.update(matched_ids)
        results.append(
            FocusFileInfo(
                error_index=error_index,
                unique_error_id=error.unique_error_id,
                error_description=error.unique_error_description,
                severity=error.severity,
                scenario_ids=matched_ids,
                file_path=file_path,
            )
        )

    if not results:
        return []

    all_scenarios = [
        scenario_lookup[sid]
        for sid in sorted(all_scenario_ids)
        if sid in scenario_lookup
    ]
    all_file_path = os.path.join(focus_dir, "all_failures.json")
    all_bundle = Scenarios(
        schema_version=scenarios.schema_version,
        scenarios=all_scenarios,
    )
    save_json_file(all_bundle.model_dump(), all_file_path, overwrite=True)

    return results
