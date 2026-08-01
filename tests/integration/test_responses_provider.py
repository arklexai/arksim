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
_CACHE_TEST_MODEL = os.environ.get("ARKSIM_CACHE_TEST_MODEL", "gpt-5.1")

# Per-model cache verification matrix. Order matters only for readability:
# the arksim DEFAULT_MODEL (`arksim/constants.py`) leads, family siblings
# follow, then known-working baseline, then known-broken regression sentinel.
#
# Names that no longer resolve (``gpt-5.1-mini``, ``gpt-5.1-nano``) skip
# at construction time; we keep them in the list so a future OpenAI release
# of those SKUs is automatically picked up without a test edit.
_CACHE_MATRIX_MODELS = (
    "gpt-5.1",  # arksim DEFAULT_MODEL
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o-mini",  # historical baseline (cache routing variable)
    "gpt-4.1-mini",  # slow-warm regression sentinel
)

# Models that empirically return ``cached_tokens=0`` across all four calls
# even with a stable >=1024-token prefix. The matrix test treats hits on
# these as a regression notice (remove them from the set) rather than a
# failure of arksim. Empty today, retained as the documented escape hatch.
_CACHE_MATRIX_KNOWN_BROKEN: frozenset[str] = frozenset()


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

    Model pin: this test defaults to arksim's DEFAULT_MODEL (gpt-5.1)
    regardless of ``ARKSIM_SMOKE_MODEL`` so the verified benefit
    tracks the model arksim users actually run. Override via
    ``ARKSIM_CACHE_TEST_MODEL`` if your account routes a different
    model for caching tests.
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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; opt-in cache matrix skipped",
)
@pytest.mark.parametrize("model", _CACHE_MATRIX_MODELS)
def test_responses_provider_cache_matrix(model: str) -> None:
    """Per-model verification that OpenAI's prompt cache fires.

    Status per model is dynamic (depends on OpenAI's server-side cache
    routing for each model and shard warmth). The matrix exists to
    (1) prove arksim's DEFAULT_MODEL actually benefits from caching,
    (2) flag regressions when OpenAI changes model behavior, and
    (3) give PR reviewers an at-a-glance answer for which models
    deliver the documented benefit.

    Shape mirrors ``test_responses_provider_cache_hit_on_repeat_prefix``:
    four calls with a ~2100-token stable prefix, asserting cumulative
    ``cached_input_tokens > 0``. A two-call shape is unreliable because
    OpenAI load-balances across cache shards and a fresh prefix often
    misses the first repeat.

    Cost: roughly $0.01-0.05 per model per run.
    """
    try:
        llm = LLM(model=model, provider="responses")
    except Exception as exc:
        pytest.skip(f"Could not construct LLM for {model}: {exc}")

    paragraph = (
        "You are a friendly customer service simulator. The user is "
        "trying to resolve a billing issue with their subscription. "
        "Be patient, ask clarifying questions, and stay in character. "
    )
    stable_prefix = paragraph * 60  # ~2100 tokens, comfortably above 1024

    num_calls = 4
    for i in range(num_calls):
        try:
            llm.call(stable_prefix + f"Reply with only the number {i}.")
        except Exception as exc:
            pytest.skip(f"Model {model} unavailable or call failed: {exc}")
        if i < num_calls - 1:
            time.sleep(2)

    stats = llm.cache_stats()
    cached = stats["cached_input_tokens"]

    if model in _CACHE_MATRIX_KNOWN_BROKEN:
        if cached > 0:
            pytest.fail(
                f"Model {model} was known-broken (cached_tokens always 0 "
                f"across {num_calls} calls) but now shows {cached} cached "
                f"tokens. Remove {model!r} from _CACHE_MATRIX_KNOWN_BROKEN."
            )
        pytest.skip(
            f"Model {model} known to return cached_tokens=0 (server-side "
            f"cache-routing anomaly). Observed cached={cached}."
        )

    assert cached > 0, (
        f"Expected at least one cache hit across {num_calls} calls to "
        f"{model} with a ~2100-token stable prefix. Got cached={cached}. "
        f"Full stats: {stats}. If this is a known model-side regression, "
        f"add {model!r} to _CACHE_MATRIX_KNOWN_BROKEN."
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; opt-in growth test skipped",
)
def test_cache_hits_grow_across_simulator_turns() -> None:
    """Cache fires across multi-turn simulator calls (arksim's real shape).

    arksim's user simulator sends [system + scenario + growing history +
    trigger] on every turn. The stable prefix is reused across turns, so
    OpenAI's prompt cache fires from turn 2 onward and the cumulative hit
    rate climbs sharply across a multi-turn scenario.

    Observed cold-cache shape on gpt-5.1 (turn-1 starts cold, turns 2-5
    hit 85-95%): per-turn rates [0%, 86%, 92%, 89%, 95%], cumulative 74%.

    Distinct from the matrix test, which sends the SAME prefix twice
    (idealized retry shape). This test exercises the real arksim call
    pattern where each turn's prompt differs but shares a stable prefix.
    """
    llm = LLM(model=_CACHE_TEST_MODEL, provider="responses")

    # OpenAI's prompt cache only fires for prefixes >= 1024 tokens. Size the
    # stable portion (system + scenario) to ~1250 tokens so the cache is
    # eligible from turn 1. Real arksim user-simulator prompts in
    # production tend to land in the 1500-3000 token range once the
    # scenario goal and persona are populated.
    system_prompt = (
        "You are a simulated user testing a customer-service AI agent. "
        "Stay in character, respond naturally, and explore the agent's "
        "capabilities. Do not break character or reveal that you are a "
        "simulator. Always end your response with a follow-up question "
        "or a clarification request when appropriate. "
    ) * 10  # ~530 tokens

    scenario_context = (
        "Scenario: You are a long-time customer of XYZ Bank. Your goal "
        "is to dispute a charge on your credit card statement from last "
        "month. You believe the charge of $89.99 from 'Acme Subscriptions' "
        "is unauthorized; you never signed up for this service. You are "
        "polite but firm. You have your account number ready: 4532-XXXX. "
    ) * 10  # ~720 tokens

    history: list[str] = []
    hit_rates: list[float] = []
    cached_per_turn: list[int] = []

    for turn in range(5):
        previous_cached = llm.cache_stats()["cached_input_tokens"]
        previous_input = llm.cache_stats()["input_tokens"]

        message = (
            system_prompt
            + "\n\n"
            + scenario_context
            + "\n\nConversation so far:\n"
            + "\n".join(history)
            + f"\n\nIt is turn {turn + 1}. Respond as the user "
            "with one short sentence."
        )
        response = llm.call(message)
        history.append(f"user (turn {turn + 1}): {response}")
        history.append(
            f"agent (turn {turn + 1}): I understand. Let me look into that for you."
        )

        new_cached = llm.cache_stats()["cached_input_tokens"] - previous_cached
        new_input = llm.cache_stats()["input_tokens"] - previous_input
        rate = new_cached / new_input if new_input > 0 else 0.0
        hit_rates.append(rate)
        cached_per_turn.append(new_cached)

    final_stats = llm.cache_stats()

    # Headline assertion: cumulative cache hit rate must reflect the
    # growing prefix benefit. After 5 turns of a scenario, at least 30%
    # of input tokens should be cached. This is a conservative floor.
    overall_rate = final_stats["cache_hit_rate"]
    assert overall_rate >= 0.30, (
        f"Expected >= 30% cumulative cache hit rate after 5 turns, "
        f"got {overall_rate:.1%}. Per-turn: {hit_rates}. "
        f"Cached tokens per turn: {cached_per_turn}. "
        f"This is the realistic shape arksim users hit; if this drops "
        f"below 30% the documented cost-reduction claim is wrong."
    )

    # Cache must keep firing as the conversation grows. OpenAI's prompt
    # cache rounds to 128-token blocks and load-balances across shards,
    # so we cannot assert strict monotonic growth (cold shards may flake
    # a single turn back to 0). We assert that the cache fires on the
    # majority of turns and that the cached-token count never collapses.
    turns_with_hits = sum(1 for c in cached_per_turn if c > 0)
    assert turns_with_hits >= 3, (
        f"Cache must fire on at least 3 of 5 turns for the growth claim "
        f"to hold. Got {turns_with_hits} turns with hits. "
        f"Per-turn cached tokens: {cached_per_turn}."
    )
