# SPDX-License-Identifier: Apache-2.0
import logging

from arksim.llms.chat import LLM
from arksim.simulation_engine.utils.prompts import (
    ATTR_TO_PROFILE,
    ATTR_TO_PROFILE_WO_LABEL,
    FIND_MATCHED_ATTRIBUTE_PROMPT,
)
from arksim.simulation_engine.utils.schema import MatchedAttributeSchema

logger = logging.getLogger(__name__)


def attributes_to_text(attributes: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in attributes.items())


async def find_matched_attribute_async(
    llm: LLM, goal: str, user_attribute: dict, attribute_options: list[str]
) -> str:
    """Find the most relevant user attribute for achieving a given goal.

    Used during conversation simulation to determine
    which attribute to emphasize in the user profile.
    """
    try:
        system_instruction = FIND_MATCHED_ATTRIBUTE_PROMPT.format(
            goal=goal,
            user_profile=attributes_to_text(user_attribute),
            attribute_options=attribute_options,
        )
        response = await llm.call_async(
            [{"role": "user", "content": system_instruction}],
            schema=MatchedAttributeSchema,
        )
        return response.attribute
    except Exception as e:
        logger.warning(
            f"Failed to find matched attribute: {e}. Using no matched attribute."
        )
        return ""


async def convert_attribute_to_profile_async(
    llm: LLM,
    user_attributes: dict,
    label: str,
    agent_context: str,
) -> str:
    """Convert user attribute dicts into natural-language profile
    descriptions suitable for conversation system prompts.
    """
    try:
        attr_text = attributes_to_text(user_attributes)
        if label:
            message = [
                {
                    "role": "user",
                    "content": ATTR_TO_PROFILE.format(
                        agent_context=agent_context,
                        user_attr=attr_text,
                        label=label,
                    ),
                }
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": ATTR_TO_PROFILE_WO_LABEL.format(
                        agent_context=agent_context,
                        user_attr=attr_text,
                    ),
                }
            ]
        profile = await llm.call_async(message)
        return profile
    except Exception as e:
        logger.warning(
            f"Failed to convert attributes to profiles: {e}. "
            f"Using stringified attributes as profiles."
        )
        return "; ".join(f"{k}: {v}" for k, v in user_attributes.items())
