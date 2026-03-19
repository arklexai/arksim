# SPDX-License-Identifier: Apache-2.0

"""
Adds or updates a versioned entry in docs/docs.json based on the current `main` version.

For a new minor version (patch=0), a new entry is created pointing to the minor series folder.
For a patch release, the existing minor series entry's version label is updated in place.

Usage: python3 version-docs.py <version> <minor-series>
Example: python3 version-docs.py 0.4.0 v0.4.x
         python3 version-docs.py 0.4.1 v0.4.x
"""

import json
import sys


def replace_prefix(obj: object, old: str, new: str) -> object:
    """Recursively replace page path prefixes in a navigation pages list."""
    if isinstance(obj, str):
        return (
            obj.replace(f"{old}/", f"{new}/", 1) if obj.startswith(f"{old}/") else obj
        )
    if isinstance(obj, dict):
        return {k: replace_prefix(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_prefix(item, old, new) for item in obj]
    return obj


def _flatten_pages(pages: object) -> list[str]:
    """Recursively extract all string values from a pages structure."""
    result = []
    if isinstance(pages, str):
        result.append(pages)
    elif isinstance(pages, dict):
        for v in pages.values():
            result.extend(_flatten_pages(v))
    elif isinstance(pages, list):
        for item in pages:
            result.extend(_flatten_pages(item))
    return result


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: version-docs.py <version> <minor-series>")
        sys.exit(1)

    new_version = sys.argv[1]
    minor_series = sys.argv[2]  # e.g. "v0.4.x"
    is_patch = int(new_version.split(".")[2]) != 0
    docs_path = "docs/docs.json"

    with open(docs_path) as f:
        config = json.load(f)

    versions = config["navigation"]["versions"]

    # Strip Latest tag and default flag from all existing versions
    for v in versions:
        v.pop("tag", None)
        v.pop("default", None)

    if is_patch:
        # Find the existing minor series entry by its page paths and update the version label
        entry = next(
            (
                v
                for v in versions
                if any(
                    p.startswith(f"{minor_series}/")
                    for p in _flatten_pages(v.get("pages", []))
                )
            ),
            None,
        )
        if entry is None:
            print(f"Error: no existing entry found for minor series {minor_series}")
            sys.exit(1)
        entry["version"] = minor_series.lstrip("v")  # display as "0.4.x"
        entry["tag"] = "Latest"
        entry["default"] = True
        print(
            f"Updated minor series {minor_series} to version {new_version} in {docs_path}"
        )
    else:
        # Remove any existing entry for this minor series so a re-publish overwrites it cleanly
        existing = next(
            (v for v in versions if v["version"] == minor_series.lstrip("v")), None
        )
        if existing is not None:
            print(
                f"Version {minor_series.lstrip('v')} already exists in {docs_path}, overwriting."
            )
            versions.remove(existing)

        # Find the main version entry
        main_entry = next((v for v in versions if v["version"] == "main"), None)
        if main_entry is None:
            print("Error: no 'main' version found in docs.json")
            sys.exit(1)

        # Build the new version entry by copying main's pages with updated paths
        new_entry = {
            "version": minor_series.lstrip("v"),  # display as "0.4.x"
            "tag": "Latest",
            "default": True,
            "pages": replace_prefix(main_entry["pages"], "main", minor_series),
        }

        # Insert new version at the front, before main
        main_index = next(i for i, v in enumerate(versions) if v["version"] == "main")
        versions.insert(main_index, new_entry)
        print(f"Added version {new_version} to {docs_path}")

    with open(docs_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
