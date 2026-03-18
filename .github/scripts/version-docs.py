# SPDX-License-Identifier: Apache-2.0

"""
Adds a new versioned entry to docs/docs.json based on the current `main` version.

Usage: python3 version-docs.py <version>
Example: python3 version-docs.py 0.4.0
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


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: version-docs.py <version>")
        sys.exit(1)

    new_version = sys.argv[1]
    docs_path = "docs/docs.json"

    with open(docs_path) as f:
        config = json.load(f)

    versions = config["navigation"]["versions"]

    # Remove any existing entry for this version so a re-publish overwrites it cleanly
    existing = next((v for v in versions if v["version"] == new_version), None)
    if existing is not None:
        print(f"Version {new_version} already exists in {docs_path}, overwriting.")
        versions.remove(existing)

    # Find the main version entry
    main_entry = next((v for v in versions if v["version"] == "main"), None)
    if main_entry is None:
        print("Error: no 'main' version found in docs.json")
        sys.exit(1)

    # Strip Latest tag and default flag from all existing versions
    for v in versions:
        v.pop("tag", None)
        v.pop("default", None)

    # Build the new version entry by copying main's pages with updated paths
    new_entry = {
        "version": new_version,
        "tag": "Latest",
        "default": True,
        "pages": replace_prefix(main_entry["pages"], "main", f"v{new_version}"),
    }

    # Insert new version at the front, before main
    main_index = next(i for i, v in enumerate(versions) if v["version"] == "main")
    versions.insert(main_index, new_entry)

    with open(docs_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Added version {new_version} to {docs_path}")


if __name__ == "__main__":
    main()
