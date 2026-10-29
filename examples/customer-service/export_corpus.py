#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Export the knowledge articles as markdown files for corpus upload.

Phase 1 grounds generation on the uploaded corpus and Phase 2 adds the MCP
source, so end-to-end testing needs both pointed at the same domain. Exporting
a subset here produces that second source without a second dataset: run with
--subset to get corpus and MCP content that overlap partially, which is what
exercises the retrieval loop's dedupe-by-content-hash path.

    python export_corpus.py                      # all 89 articles
    python export_corpus.py --subset service     # service articles only
    python export_corpus.py --subset categories  # per-category articles only
    python export_corpus.py --out /tmp/corpus
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from store_tools import _query, init_db  # noqa: E402

SUBSET_FILTERS = {
    "all": "",
    # Per-category policy, buying, and care articles.
    "categories": "WHERE category != ''",
    # Store-wide service articles (shipping, returns, payments, and so on).
    "service": "WHERE category = ''",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def export(out_dir: Path, subset: str) -> tuple[int, int]:
    init_db()
    rows = _query(
        "SELECT id, title, topic, category, content, source_uri "
        f"FROM knowledge_articles {SUBSET_FILTERS[subset]} ORDER BY id"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear articles from an earlier run first. The subsets are only useful
    # because they are disjoint, and leaving a wider run's files in place
    # silently merges them back together - exporting "service" over "all"
    # would hand you all 89 articles while reporting 29. Restricted to the
    # KB-*.md names this script writes, so nothing else in the directory
    # is touched.
    stale = sorted(out_dir.glob("KB-*.md"))
    for path in stale:
        path.unlink()
    for row in rows:
        path = out_dir / f"{row['id']}-{slugify(row['title'])}.md"
        front_matter = "\n".join(
            (
                "---",
                f"id: {row['id']}",
                f"title: {row['title']}",
                f"topic: {row['topic']}",
                f"category: {row['category']}",
                f"source_uri: {row['source_uri']}",
                "---",
            )
        )
        path.write_text(f"{front_matter}\n\n{row['content']}\n")
    return len(rows), len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "corpus")
    parser.add_argument("--subset", choices=sorted(SUBSET_FILTERS), default="all")
    args = parser.parse_args()
    count, removed = export(args.out, args.subset)
    cleared = f" (cleared {removed} from a previous export)" if removed else ""
    print(f"Wrote {count} '{args.subset}' articles to {args.out}{cleared}")


if __name__ == "__main__":
    main()
