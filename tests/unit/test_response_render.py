# SPDX-License-Identifier: Apache-2.0
"""Tests for rendering a custom_http response into assistant-turn text.

Covers reply extraction by dot-path, the shapes a response_extras path can
resolve to (one item, a list, a map, nested with a wildcard), hole substitution,
and the caps on items and total length.
"""

from __future__ import annotations

from typing import Any

import pytest

from arksim.simulation_engine.agent.response_render import (
    MAX_ITEMS_PER_SECTION,
    MAX_RENDERED_CHARS,
    ResponseRenderError,
    extract_reply_text,
    render_reply,
)

LINK_TEMPLATE = [
    {"path": "links", "label": "Links", "template": "{title}: https://{id}.com"}
]
BARE_TEMPLATE = [{"path": "links", "label": "Links", "template": "https://{}.com"}]
PAIR_TEMPLATE = [
    {"path": "links", "label": "Links", "template": "{key}: https://{value}.com"}
]


class TestReplyText:
    """response_path names the reply, and stays strict about resolving."""

    def test_top_level_key(self) -> None:
        assert render_reply({"raw_text": "hi"}, "raw_text") == "hi"

    def test_nested_path(self) -> None:
        assert render_reply({"data": {"reply": "hi"}}, "data.reply") == "hi"

    def test_list_index_segment(self) -> None:
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert render_reply(response, "choices.0.message.content") == "hi"

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ResponseRenderError, match="key 'nope' missing"):
            render_reply({"raw_text": "hi"}, "nope")

    def test_index_out_of_range_raises(self) -> None:
        with pytest.raises(ResponseRenderError, match="out of range"):
            render_reply({"choices": []}, "choices.0")

    def test_non_string_reply_raises(self) -> None:
        """A non-string reply would become an unreadable turn."""
        with pytest.raises(ResponseRenderError, match="did not resolve to a string"):
            render_reply({"raw_text": {"nested": "hi"}}, "raw_text")

    def test_extract_reply_text_is_the_same_walk(self) -> None:
        assert extract_reply_text({"data": {"reply": "hi"}}, "data.reply") == "hi"


class TestShapes:
    """The response decides the shape, so one config handles all of them."""

    @pytest.mark.parametrize(
        ("links", "extras", "expected"),
        [
            pytest.param(
                {"title": "Reset", "id": "a1"},
                LINK_TEMPLATE,
                "Links: Reset: https://a1.com",
                id="single object renders inline",
            ),
            pytest.param(
                [{"title": "Reset", "id": "a1"}, {"title": "2FA", "id": "d4"}],
                LINK_TEMPLATE,
                "Links:\n- Reset: https://a1.com\n- 2FA: https://d4.com",
                id="list of objects renders as bullets",
            ),
            pytest.param(
                "a1",
                BARE_TEMPLATE,
                "Links: https://a1.com",
                id="bare scalar",
            ),
            pytest.param(
                ["a1", "d4"],
                BARE_TEMPLATE,
                "Links:\n- https://a1.com\n- https://d4.com",
                id="list of scalars",
            ),
            pytest.param(
                {"reset": "a1", "twofa": "d4"},
                PAIR_TEMPLATE,
                "Links:\n- reset: https://a1.com\n- twofa: https://d4.com",
                id="map walked as pairs",
            ),
        ],
    )
    def test_shape(
        self,
        links: Any,  # noqa: ANN401
        extras: list[dict[str, Any]],
        expected: str,
    ) -> None:
        rendered = render_reply({"raw_text": "T.", "links": links}, "raw_text", extras)
        assert rendered == f"T.\n\n{expected}"

    @pytest.mark.parametrize(
        "links",
        [pytest.param(None, id="null"), pytest.param([], id="empty list")],
    )
    def test_empty_field_drops_the_section(self, links: Any) -> None:  # noqa: ANN401
        response = {"raw_text": "T.", "links": links}
        assert render_reply(response, "raw_text", LINK_TEMPLATE) == "T."

    def test_missing_field_drops_the_section(self) -> None:
        assert render_reply({"raw_text": "T."}, "raw_text", LINK_TEMPLATE) == "T."

    def test_wildcard_reaches_into_nested_lists(self) -> None:
        response = {
            "raw_text": "T.",
            "rc": {"rec": {"best": [{"itemId": "x1"}, {"itemId": "x2"}]}},
        }
        rendered = render_reply(
            response,
            "raw_text",
            [
                {
                    "path": "rc.rec.best[].itemId",
                    "label": "Recommended",
                    "template": "SKU-{}",
                }
            ],
        )
        assert rendered == "T.\n\nRecommended:\n- SKU-x1\n- SKU-x2"


class TestRendering:
    """Labels, templates, and defaults."""

    def test_no_template_renders_the_value(self) -> None:
        rendered = render_reply(
            {"raw_text": "T.", "links": ["a1", "d4"]},
            "raw_text",
            [{"path": "links", "label": "Links"}],
        )
        assert rendered == "T.\n\nLinks:\n- a1\n- d4"

    def test_no_template_flattens_an_object(self) -> None:
        rendered = render_reply(
            {"raw_text": "T.", "links": {"title": "Reset", "id": "a1"}},
            "raw_text",
            [{"path": "links"}],
        )
        assert rendered == "T.\n\ntitle: Reset, id: a1"

    def test_no_label_omits_the_header(self) -> None:
        rendered = render_reply(
            {"raw_text": "T.", "links": ["a1"]},
            "raw_text",
            [{"path": "links", "template": "https://{}.com"}],
        )
        assert rendered == "T.\n\nhttps://a1.com"

    def test_unresolved_hole_renders_empty(self) -> None:
        """Leniency per hole, so a heterogeneous list degrades per item."""
        rendered = render_reply(
            {"raw_text": "T.", "links": [{"id": "a1"}]},
            "raw_text",
            [{"path": "links", "template": "{nope}https://{id}.com"}],
        )
        assert rendered == "T.\n\nhttps://a1.com"

    def test_several_sections_keep_config_order(self) -> None:
        rendered = render_reply(
            {"raw_text": "T.", "links": ["a1"], "cites": ["c1", "c2"]},
            "raw_text",
            [
                {"path": "links", "label": "Links", "template": "https://{}.com"},
                {"path": "cites", "label": "Sources"},
            ],
        )
        assert rendered == "T.\n\nLinks: https://a1.com\n\nSources:\n- c1\n- c2"

    def test_holes_are_filled_by_substitution_only(self) -> None:
        """A template is data: braces that are not a path stay literal."""
        rendered = render_reply(
            {"raw_text": "T.", "links": ["a1"]},
            "raw_text",
            [{"path": "links", "template": "{:1000000000}{}"}],
        )
        assert rendered == "T.\n\n{:1000000000}a1"


class TestCaps:
    """Output is bounded by input, and both bounds truncate rather than raise."""

    def test_items_capped_with_a_count_of_the_rest(self) -> None:
        links = [f"a{i}" for i in range(MAX_ITEMS_PER_SECTION + 10)]
        rendered = render_reply(
            {"raw_text": "T.", "links": links},
            "raw_text",
            [{"path": "links", "label": "Links"}],
        )
        assert rendered.count("\n- ") == MAX_ITEMS_PER_SECTION + 1
        assert rendered.endswith("- ... and 10 more")

    def test_total_length_truncated(self) -> None:
        response = {"raw_text": "T.", "links": ["y" * 40_000] * 10}
        rendered = render_reply(
            {**response},
            "raw_text",
            [{"path": "links", "label": "Links"}],
        )
        assert len(rendered) <= MAX_RENDERED_CHARS + len("\n... (truncated)")
        assert rendered.endswith("... (truncated)")
