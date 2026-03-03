# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel

from arksim.utils.output import load_json_file


class KnowledgeItem(BaseModel):
    """Knowledge used in a scenario."""

    content: str = ""


class Scenario(BaseModel):
    """A single scenario item."""

    scenario_id: str
    user_id: str
    goal: str
    agent_context: str
    knowledge: list[KnowledgeItem]
    user_attributes: dict
    origin: dict


class Scenarios(BaseModel):
    """Output of the scenario builder."""

    schema_version: str
    scenarios: list[Scenario]

    @classmethod
    def load(cls, path: str) -> "Scenarios":
        """Load scenarios from a JSON file."""
        return cls.model_validate(load_json_file(path))
