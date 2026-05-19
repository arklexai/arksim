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
import time

import pytest

from arksim.llms.chat.llm import LLM

_SMOKE_MODEL = os.environ.get("ARKSIM_SMOKE_MODEL", "gpt-4.1-mini")
_CACHE_TEST_MODEL = os.environ.get("ARKSIM_CACHE_TEST_MODEL", "gpt-4o-mini")


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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; opt-in cache test skipped",
)
def test_responses_provider_cache_hit_on_repeat_prefix() -> None:
    """Empirically verify OpenAI's prompt cache fires for repeat prefixes.

    Per OpenAI's prompt-caching docs, prompts >= 1024 tokens with stable
    prefixes are cached automatically. This test sends several calls with
    the same large prefix and asserts at least one observes cached_tokens
    > 0. Multiple calls are needed because OpenAI load-balances across
    servers and a fresh prefix may not hit a warm cache on the first
    retry. The cumulative ``cached_input_tokens`` counter on the LLM
    captures any hit across the batch.

    Why this exists: PR #170 documents a 20-40% cost reduction from
    automatic prompt caching. Without this test, the claim is unverified.

    Model pin: this test defaults to gpt-4o-mini regardless of
    ``ARKSIM_SMOKE_MODEL`` because some newer models intermittently
    report zero cache hits even with valid stable prefixes (observed
    on gpt-4.1-mini and reported on gpt-5.x-nano in OpenAI community
    threads). Override via ``ARKSIM_CACHE_TEST_MODEL`` if your account
    has a different model with reliable caching.
    """
    llm = LLM(model=_CACHE_TEST_MODEL, provider="responses")

    # Build a stable prefix comfortably above the 1024-token cache floor.
    # Empirically the paragraph below tokenizes to ~35 tokens, so 60
    # repetitions yields ~2100 tokens.
    paragraph = (
        "You are a friendly customer service simulator. The user is "
        "trying to resolve a billing issue with their subscription. "
        "Be patient, ask clarifying questions, and stay in character. "
    )
    stable_prefix = paragraph * 60  # ~2100 tokens, comfortably above 1024

    # Send several calls with a small inter-call delay. The first primes
    # the cache; subsequent calls should hit it. OpenAI load-balances
    # across cache shards, so we look for at least one hit across the
    # batch rather than asserting per-call.
    num_calls = 4
    for i in range(num_calls):
        llm.call(stable_prefix + f"Reply with only the number {i}.")
        if i < num_calls - 1:
            time.sleep(2)

    stats = llm.cache_stats()
    assert stats["cached_input_tokens"] > 0, (
        f"Expected at least one cache hit across {num_calls} calls with "
        f"a ~2100-token stable prefix on model {_CACHE_TEST_MODEL!r}, "
        f"got cached_input_tokens=0. Full stats: {stats}. "
        f"Possible causes: model does not support prompt caching, "
        f"OpenAI cache shard miss across all retries, "
        f"or the cached_tokens field is not populated by this backend."
    )
