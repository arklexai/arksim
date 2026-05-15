# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration smokes for the responses provider against api.openai.com.

Skipped unless OPENAI_API_KEY is set in the environment. Two short calls
verify (1) alias wiring through the default constructor and (2) explicit
base_url + api_key end-to-end through the OpenAI SDK.

Override the model via ARKSIM_SMOKE_MODEL when gpt-4.1-mini is deprecated.

Cost: pennies per run on the happy path. OpenAILLM.call is wrapped in
`@retry(max_retries=5)`; auth or model-deprecation failures may make up
to 6 attempts before surfacing the error.
"""

from __future__ import annotations

import os

import pytest

from arksim.llms.chat.llm import LLM

_SMOKE_MODEL = os.environ.get("ARKSIM_SMOKE_MODEL", "gpt-4.1-mini")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; opt-in smoke test skipped",
)
def test_responses_provider_smoke_against_openai() -> None:
    llm = LLM(model=_SMOKE_MODEL, provider="responses")
    out = llm.call("Say only the word 'hello' and nothing else.")
    assert isinstance(out, str)
    assert "hello" in out.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; opt-in smoke test skipped",
)
def test_responses_provider_smoke_with_explicit_base_url() -> None:
    """Exercise the new base_url + api_key path against the real endpoint.

    Without an explicit base_url the provider falls back to env vars and
    we would not be testing the new code path. This call forces both
    the base_url and api_key plumbing to be exercised end-to-end.
    """
    llm = LLM(
        model=_SMOKE_MODEL,
        provider="responses",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    out = llm.call("Say only the word 'hello' and nothing else.")
    assert isinstance(out, str)
    assert "hello" in out.lower()
