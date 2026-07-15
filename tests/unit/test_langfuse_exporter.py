# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Langfuse exporter.

These tests use a recording double for the Langfuse client, so they run
without the optional ``langfuse`` package installed. The double mirrors the
real SDK contract (verified separately against langfuse 4.x): keyword-only
methods, context-manager observations, and ``score`` / ``score_trace`` /
``update`` on the yielded span object.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from arksim.evaluator.base_metric import QualResult, QuantResult
from arksim.evaluator.entities import (
    ConversationEvaluation,
    Evaluation,
    TurnEvaluation,
)
from arksim.integrations.langfuse import LangfuseExporter
from arksim.scenario.entities import (
    ExpectedToolCall,
    KnowledgeItem,
    Scenario,
    Scenarios,
    ToolCallsAssertion,
)
from arksim.simulation_engine.entities import (
    Conversation,
    Message,
    SimulatedUserPrompt,
    Simulation,
)
from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource

# ── Recording double for the Langfuse client ────────────────────────────


class FakeSpan:
    def __init__(self, recorder: FakeLangfuse, as_type: str, name: str) -> None:
        self._rec = recorder
        self.as_type = as_type
        self.name = name
        self.scores: list[dict[str, Any]] = []
        self.trace_scores: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.id = f"obs-{len(recorder.spans)}"

    def score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def score_trace(self, **kwargs: Any) -> None:
        self.trace_scores.append(kwargs)

    def update(self, **kwargs: Any) -> Any:
        self.updates.append(kwargs)
        return self


class FakeDatasetClient:
    def __init__(self, recorder: FakeLangfuse, name: str) -> None:
        self._rec = recorder
        self.name = name

    def run_experiment(self, *, task: Any, **kwargs: Any) -> Any:
        self._rec.run_experiment_calls.append(kwargs)
        items = [i for i in self._rec.items if i["dataset_name"] == self.name]
        for item_kwargs in items:
            item = _FakeItem(item_kwargs)
            output = task(item=item)
            self._rec.task_outputs.append((item.id, output))
        return _FakeResult(len(items))


class _FakeItem:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.id = kwargs.get("id")
        self.input = kwargs.get("input")
        self.expected_output = kwargs.get("expected_output")
        self.metadata = kwargs.get("metadata")


class _FakeResult:
    def __init__(self, n: int) -> None:
        self.n = n

    def format(self) -> str:
        return f"ran {self.n} items"


class FakeLangfuse:
    def __init__(self) -> None:
        self.datasets: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.spans: list[FakeSpan] = []
        self.run_experiment_calls: list[dict[str, Any]] = []
        self.task_outputs: list[tuple[Any, Any]] = []
        self.flushed = 0

    def create_dataset(self, **kwargs: Any) -> None:
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs)

    def get_dataset(self, name: str, **_: Any) -> FakeDatasetClient:
        return FakeDatasetClient(self, name)

    @contextmanager
    def start_as_current_observation(self, *, as_type: str, name: str, **_: Any) -> Any:
        span = FakeSpan(self, as_type, name)
        self.spans.append(span)
        yield span

    def flush(self) -> None:
        self.flushed += 1


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def scenarios() -> Scenarios:
    return Scenarios(
        schema_version="v1",
        scenarios=[
            Scenario(
                scenario_id="s-tool",
                user_id="u1",
                goal="Cancel my order",
                agent_context="You are a support agent.",
                knowledge=[KnowledgeItem(content="Order 42 is cancellable.")],
                user_profile="Impatient customer.",
                assertions=[
                    ToolCallsAssertion(
                        type="tool_calls",
                        expected=[
                            ExpectedToolCall(name="cancel_order", arguments={"id": 42})
                        ],
                        match_mode="contains",
                    )
                ],
            ),
            Scenario(
                scenario_id="s-plain",
                user_id="u2",
                goal="Ask store hours",
                agent_context="You are a support agent.",
                user_profile="Polite customer.",
            ),
        ],
    )


def _conversation(
    conversation_id: str, scenario_id: str, with_tool: bool
) -> Conversation:
    tool_calls = (
        [
            ToolCall(
                id="tc1",
                name="cancel_order",
                arguments={"id": 42},
                result="ok",
                source=ToolCallSource.OTEL_TRACE,
            )
        ]
        if with_tool
        else None
    )
    return Conversation(
        conversation_id=conversation_id,
        scenario_id=scenario_id,
        conversation_history=[
            Message(turn_id=0, role="simulated_user", content="Hi"),
            Message(turn_id=0, role="assistant", content="Hello!", tool_calls=None),
            Message(turn_id=1, role="simulated_user", content="Cancel order 42"),
            Message(
                turn_id=1, role="assistant", content="Done.", tool_calls=tool_calls
            ),
        ],
        simulated_user_prompt=SimulatedUserPrompt(
            simulated_user_prompt_template="t", variables={}
        ),
    )


@pytest.fixture
def simulation() -> Simulation:
    return Simulation(
        schema_version="v1.1",
        simulator_version="v1",
        conversations=[
            _conversation("c-tool", "s-tool", with_tool=True),
            _conversation("c-plain", "s-plain", with_tool=False),
        ],
    )


def _convo_eval(conversation_id: str) -> ConversationEvaluation:
    return ConversationEvaluation(
        conversation_id=conversation_id,
        goal_completion_score=0.8,
        goal_completion_reason="mostly done",
        turn_success_ratio=1.0,
        overall_agent_score=0.9,
        evaluation_status="partial_failure",
        turn_scores=[
            TurnEvaluation(
                turn_id=0,
                scores=[QuantResult(name="helpfulness", value=4.0, reason="good")],
                turn_score=4.0,
                turn_behavior_failure="skipped_good_performance",
                turn_behavior_failure_reason="",
            ),
            TurnEvaluation(
                turn_id=1,
                scores=[QuantResult(name="coherence", value=5.0, reason="clear")],
                turn_score=5.0,
                turn_behavior_failure="incorrect_tool_usage",
                turn_behavior_failure_reason="",
                qual_scores=[QualResult(name="tone", value="professional")],
            ),
        ],
    )


@pytest.fixture
def evaluation() -> Evaluation:
    return Evaluation(
        schema_version="v1.1",
        generated_at="2026-07-07T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="e1",
        simulation_id="sim1",
        conversations=[_convo_eval("c-tool"), _convo_eval("c-plain")],
        unique_errors=[],
    )


# ── Tests ───────────────────────────────────────────────────────────────


def test_creates_one_dataset_item_per_scenario(
    scenarios: Scenarios, simulation: Simulation
) -> None:
    fake = FakeLangfuse()
    LangfuseExporter(client=fake).export(scenarios, simulation, dataset_name="ds")

    assert len(fake.datasets) == 1
    assert fake.datasets[0]["name"] == "ds"

    ids = sorted(i["id"] for i in fake.items)
    assert ids == ["s-plain", "s-tool"]

    by_id = {i["id"]: i for i in fake.items}
    # expected_output derived from the tool_calls assertion
    assert by_id["s-tool"]["expected_output"]["tool_calls"][0]["name"] == "cancel_order"
    assert by_id["s-plain"]["expected_output"] is None
    # scenario input carries goal / profile / context / knowledge
    assert by_id["s-tool"]["input"]["goal"] == "Cancel my order"
    assert by_id["s-tool"]["input"]["knowledge"] == ["Order 42 is cancellable."]


def test_runs_dataset_experiment_and_flushes(
    scenarios: Scenarios, simulation: Simulation
) -> None:
    fake = FakeLangfuse()
    result = LangfuseExporter(client=fake).export(
        scenarios, simulation, dataset_name="ds", run_name="my-run"
    )

    assert len(fake.run_experiment_calls) == 1
    assert fake.run_experiment_calls[0]["name"] == "my-run"
    assert fake.flushed == 1
    assert result.format() == "ran 2 items"
    # task produced an output string per item
    assert dict(fake.task_outputs)["s-tool"] == "Done."


def test_emits_turn_and_tool_spans(
    scenarios: Scenarios, simulation: Simulation
) -> None:
    fake = FakeLangfuse()
    LangfuseExporter(client=fake).export(scenarios, simulation, dataset_name="ds")

    names = [s.name for s in fake.spans]
    # one scenario span + one conversation span per scenario
    assert "scenario:s-tool" in names
    assert "conversation:c-tool" in names
    # two turns per conversation as generations
    assert names.count("turn:0") == 2
    assert names.count("turn:1") == 2
    # the tool call in c-tool becomes a tool span
    tool_spans = [s for s in fake.spans if s.as_type == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "tool:cancel_order"


def test_attaches_scores(
    scenarios: Scenarios, simulation: Simulation, evaluation: Evaluation
) -> None:
    fake = FakeLangfuse()
    LangfuseExporter(client=fake).export(
        scenarios, simulation, evaluation, dataset_name="ds"
    )

    # turn-level scores land on the matching generation spans
    turn0 = [s for s in fake.spans if s.name == "turn:0"]
    turn1 = [s for s in fake.spans if s.name == "turn:1"]
    assert any(
        sc["name"] == "helpfulness"
        and sc["value"] == 4.0
        and sc["data_type"] == "NUMERIC"
        for s in turn0
        for sc in s.scores
    )
    assert any(
        sc["name"] == "tone" and sc["data_type"] == "CATEGORICAL"
        for s in turn1
        for sc in s.scores
    )

    # sentinel behavior-failure labels (SKIP_OUTCOMES) are not exported
    assert not any(
        sc["name"] == "turn_behavior_failure" for s in turn0 for sc in s.scores
    )
    # real failure labels are exported as categorical scores
    assert any(
        sc["name"] == "turn_behavior_failure"
        and sc["value"] == "incorrect_tool_usage"
        and sc["data_type"] == "CATEGORICAL"
        for s in turn1
        for sc in s.scores
    )

    # conversation-level scores land at the trace level (1 conversation/scenario)
    conv_spans = [s for s in fake.spans if s.name.startswith("conversation:")]
    trace_score_names = {sc["name"] for s in conv_spans for sc in s.trace_scores}
    assert {
        "overall_agent_score",
        "goal_completion",
        "turn_success_ratio",
        "evaluation_status",
    } <= trace_score_names

    # numeric vs categorical typing on trace scores
    for s in conv_spans:
        for sc in s.trace_scores:
            if sc["name"] == "evaluation_status":
                assert sc["data_type"] == "CATEGORICAL"
            if sc["name"] == "overall_agent_score":
                assert sc["data_type"] == "NUMERIC"


def test_no_evaluation_still_traces(
    scenarios: Scenarios, simulation: Simulation
) -> None:
    fake = FakeLangfuse()
    LangfuseExporter(client=fake).export(scenarios, simulation, dataset_name="ds")
    # spans exist, but no scores attached anywhere
    assert fake.spans
    assert all(not s.scores and not s.trace_scores for s in fake.spans)


def test_missing_langfuse_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("no langfuse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"arksim\[langfuse\]"):
        LangfuseExporter()
