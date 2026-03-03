# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("arksim")


def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents."""
    with open(file_path) as f:
        return json.load(f)


def save_json_file(
    data: dict, file_path: str, indent: int = 4, overwrite: bool = False
) -> None:
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path}. Set overwrite=True to replace it."
        )

    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


async def save_json_file_async(
    data: dict, file_path: str, indent: int = 4, overwrite: bool = False
) -> None:
    await asyncio.to_thread(save_json_file, data, file_path, indent, overwrite)


def resolve_output_dir(output_path: str) -> str:
    if not os.path.exists(output_path):
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if os.path.isfile(output_path):
        root, ext = os.path.splitext(output_path)
        new_path = f"{root}_{timestamp}{ext}"
        logger.warning(
            f"File '{output_path}' already exists. Using '{new_path}' instead.",
        )
    else:
        new_path = f"{output_path}_{timestamp}"
        logger.warning(
            f"Directory '{output_path}' already exists. Using '{new_path}' instead.",
        )

    return new_path
