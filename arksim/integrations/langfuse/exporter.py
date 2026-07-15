# SPDX-License-Identifier: Apache-2.0
"""Export arksim simulations and evaluations to a Langfuse instance.

The mapping is:

* each :class:`~arksim.scenario.Scenario`            -> a Langfuse *dataset item*
* each :class:`~arksim.simulation_engine.Conversation` -> a Langfuse *trace*
  (with one nested generation per turn and one nested tool span per tool call)
* the whole batch                                    -> a Langfuse *dataset run*
  (created via the SDK's ``dataset.run_experiment`` API)
* each :class:`~arksim.evaluator.entities.ConversationEvaluation`
  and its per-turn scores                            -> Langfuse *scores* on the trace

Works against any Langfuse instance (including self-hosted) using only the
public ``langfuse`` SDK. No hosted-only connector is required.

``num_conversations_per_scenario`` note
---------------------------------------
A Langfuse dataset run links exactly one trace to each dataset item. arksim
can produce several conversations per scenario. When a scenario has a single
conversation (the common case, and what the bundled example uses) the mapping
is a clean 1:1 (scenario = item = trace) and conversation-level scores are
attached at the trace level so they drive the dataset-run comparison view.
When a scenario has multiple conversations, each conversation is emitted as a
nested span subtree under the one item trace, per-conversation scores are
attached to those conversation spans, and the mean ``overall_agent_score`` is
attached at the trace level.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from arksim.evaluator.utils.constants import SKIP_OUTCOMES
from arksim.scenario.entities import AssertionType, Scenario, Scenarios
from arksim.simulation_engine.entities import Conversation, Message, Simulation

if TYPE_CHECKING:
    from arksim.evaluator.entities import (
        ConversationEvaluation,
        Evaluation,
        TurnEvaluation,
    )

logger = logging.getLogger(__name__)


def _require_langfuse() -> Any:
    """Import and return the ``langfuse`` module, or raise a helpful error."""
    try:
        import langfuse
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "The Langfuse integration requires the 'langfuse' package. "
            "Install it with:\n\n    pip install 'arksim[langfuse]'\n\n"
            "or\n\n    pip install 'langfuse>=4,<5'"
        ) from exc
    return langfuse


class LangfuseExporter:
    """Send arksim simulation transcripts and evaluation scores to Langfuse.

    Args:
        client: An already-configured ``langfuse.Langfuse`` client. If given,
            the credential keyword arguments are ignored.
        public_key: Langfuse public key. Falls back to ``LANGFUSE_PUBLIC_KEY``.
        secret_key: Langfuse secret key. Falls back to ``LANGFUSE_SECRET_KEY``.
        host: Langfuse base URL (e.g. ``http://localhost:3000`` for a
            self-hosted instance). Falls back to ``LANGFUSE_HOST``.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            langfuse = _require_langfuse()
            kwargs: dict[str, Any] = {}
            if public_key is not None:
                kwargs["public_key"] = public_key
            if secret_key is not None:
                kwargs["secret_key"] = secret_key
            if host is not None:
                kwargs["host"] = host
            self._client = langfuse.Langfuse(**kwargs)

    @property
    def client(self) -> Any:
        """The underlying ``langfuse.Langfuse`` client."""
        return self._client

    # ── dataset creation ────────────────────────────────────────────────

    @staticmethod
    def _scenario_input(scenario: Scenario) -> dict[str, Any]:
        """Build the dataset-item input payload from a scenario."""
        return {
            "goal": scenario.goal,
            "user_profile": scenario.user_profile,
            "agent_context": scenario.agent_context,
            "knowledge": [k.content for k in scenario.knowledge],
        }

    @staticmethod
    def _expected_output(scenario: Scenario) -> dict[str, Any] | None:
        """Derive expected_output from a scenario's tool-call assertion, if any."""
        assertion = scenario.find_assertion(AssertionType.TOOL_CALLS)
        if assertion is None:
            return None
        return {
            "tool_calls": [
                {
                    "name": e.name,
                    "arguments": e.arguments,
                    "arg_match_mode": e.arg_match_mode,
                }
                for e in assertion.expected
            ],
            "match_mode": assertion.match_mode,
        }

    def _ensure_dataset(
        self,
        scenarios: Scenarios,
        dataset_name: str,
        dataset_description: str | None,
        create_items: bool,
    ) -> Any:
        """Create the dataset and (optionally) upsert one item per scenario.

        ``create_dataset`` and ``create_dataset_item`` are upserts keyed by
        name / id, so re-running against an existing dataset is safe.
        """
        self._client.create_dataset(
            name=dataset_name,
            description=dataset_description
            or "arksim scenarios (auto-generated by LangfuseExporter)",
            metadata={
                "source": "arksim",
                "schema_version": scenarios.schema_version,
            },
        )
        if create_items:
            for scenario in scenarios.scenarios:
                self._client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=scenario.scenario_id,
                    input=self._scenario_input(scenario),
                    expected_output=self._expected_output(scenario),
                    metadata={
                        "scenario_id": scenario.scenario_id,
                        "user_id": scenario.user_id,
                        "origin": scenario.origin,
                    },
                )
        return self._client.get_dataset(dataset_name)

    # ── trace / span emission ───────────────────────────────────────────

    def _emit_turn(
        self,
        assistant_msg: Message,
        user_input: str | None,
        turn_eval: TurnEvaluation | None,
    ) -> None:
        """Emit one generation span for a turn, its tool spans, and turn scores."""
        with self._client.start_as_current_observation(
            as_type="generation",
            name=f"turn:{assistant_msg.turn_id}",
            input=user_input,
            output=assistant_msg.content,
            metadata={"turn_id": assistant_msg.turn_id},
        ) as turn_span:
            for tc in assistant_msg.tool_calls or []:
                with self._client.start_as_current_observation(
                    as_type="tool",
                    name=f"tool:{tc.name}",
                    input=tc.arguments,
                    output=tc.result,
                    level="ERROR" if tc.error else "DEFAULT",
                    status_message=tc.error,
                    metadata={
                        "tool_call_id": tc.id,
                        "source": tc.source.value if tc.source else None,
                    },
                ):
                    pass

            if turn_eval is None:
                return

            for q in turn_eval.scores:
                turn_span.score(
                    name=q.name,
                    value=q.value,
                    data_type="NUMERIC",
                    comment=q.reason,
                )
            # turn_behavior_failure is always a non-empty string; sentinel
            # values in SKIP_OUTCOMES mean the turn had no actionable failure.
            if (
                turn_eval.turn_behavior_failure
                and turn_eval.turn_behavior_failure not in SKIP_OUTCOMES
            ):
                turn_span.score(
                    name="turn_behavior_failure",
                    value=turn_eval.turn_behavior_failure,
                    data_type="CATEGORICAL",
                    comment=turn_eval.turn_behavior_failure_reason or None,
                )
            for ql in turn_eval.qual_scores:
                turn_span.score(
                    name=ql.name,
                    value=ql.value,
                    data_type="CATEGORICAL",
                    comment=ql.reason,
                )

    @staticmethod
    def _attach_convo_scores(
        scorer: Any,
        convo_eval: ConversationEvaluation,
        to_trace: bool,
    ) -> None:
        """Attach conversation-level scores via ``score_trace`` or ``score``."""
        emit = scorer.score_trace if to_trace else scorer.score
        emit(
            name="overall_agent_score",
            value=convo_eval.overall_agent_score,
            data_type="NUMERIC",
        )
        emit(
            name="goal_completion",
            value=convo_eval.goal_completion_score,
            data_type="NUMERIC",
            comment=convo_eval.goal_completion_reason or None,
        )
        emit(
            name="turn_success_ratio",
            value=convo_eval.turn_success_ratio,
            data_type="NUMERIC",
        )
        emit(
            name="evaluation_status",
            value=convo_eval.evaluation_status,
            data_type="CATEGORICAL",
        )
        for q in convo_eval.convo_scores:
            emit(name=q.name, value=q.value, data_type="NUMERIC", comment=q.reason)
        for ql in convo_eval.convo_qual_scores:
            emit(
                name=ql.name,
                value=ql.value,
                data_type="CATEGORICAL",
                comment=ql.reason,
            )

    def _emit_conversation(
        self,
        conversation: Conversation,
        convo_eval: ConversationEvaluation | None,
        to_trace: bool,
    ) -> str:
        """Emit a conversation as a span subtree; return its final agent output."""
        turn_eval_by_id: dict[int, TurnEvaluation] = {}
        if convo_eval is not None:
            turn_eval_by_id = {t.turn_id: t for t in convo_eval.turn_scores}

        with self._client.start_as_current_observation(
            as_type="span",
            name=f"conversation:{conversation.conversation_id}",
            metadata={
                "conversation_id": conversation.conversation_id,
                "scenario_id": conversation.scenario_id,
            },
        ) as conv_span:
            last_user: str | None = None
            final_output = ""
            for msg in conversation.conversation_history:
                if msg.role == "simulated_user":
                    last_user = msg.content
                elif msg.role == "assistant":
                    final_output = msg.content
                    self._emit_turn(msg, last_user, turn_eval_by_id.get(msg.turn_id))
                    last_user = None

            if convo_eval is not None:
                self._attach_convo_scores(conv_span, convo_eval, to_trace)
            conv_span.update(output=final_output)
        return final_output

    # ── public API ──────────────────────────────────────────────────────

    def export(
        self,
        scenarios: Scenarios,
        simulation: Simulation,
        evaluation: Evaluation | None = None,
        *,
        dataset_name: str,
        dataset_description: str | None = None,
        run_name: str | None = None,
        run_description: str | None = None,
        run_metadata: dict[str, Any] | None = None,
        create_items: bool = True,
        evaluators: list[Any] | None = None,
        run_evaluators: list[Any] | None = None,
        max_concurrency: int | None = None,
    ) -> Any:
        """Export a simulation (and optional evaluation) as a Langfuse dataset run.

        Args:
            scenarios: The scenarios that were simulated (become dataset items).
            simulation: The simulation output (conversations become traces).
            evaluation: Optional evaluation output (becomes scores on the traces).
            dataset_name: Langfuse dataset name (created/upserted).
            dataset_description: Optional dataset description.
            run_name: Name for the dataset run. Defaults to the simulation id.
            run_description: Optional dataset-run description.
            run_metadata: Optional metadata attached to the dataset run.
            create_items: When True (default), upsert one dataset item per
                scenario before running. Set False to reuse existing items.
            evaluators: Optional per-item Langfuse evaluator callables passed
                through to ``run_experiment`` (in addition to the arksim scores
                this exporter already attaches).
            run_evaluators: Optional run-level Langfuse evaluators.
            max_concurrency: Optional cap on concurrent task executions.

        Returns:
            The Langfuse experiment result object (has a ``.format()`` method).
        """
        dataset = self._ensure_dataset(
            scenarios, dataset_name, dataset_description, create_items
        )

        conv_by_scenario: dict[str, list[Conversation]] = defaultdict(list)
        for conv in simulation.conversations:
            conv_by_scenario[conv.scenario_id].append(conv)

        eval_by_conv: dict[str, ConversationEvaluation] = {}
        if evaluation is not None:
            eval_by_conv = {c.conversation_id: c for c in evaluation.conversations}

        def task(*, item: Any, **_: Any) -> Any:
            metadata = getattr(item, "metadata", None) or {}
            scenario_id = metadata.get("scenario_id") or getattr(item, "id", None)
            conversations = conv_by_scenario.get(scenario_id, [])
            total = len(conversations)

            with self._client.start_as_current_observation(
                as_type="span",
                name=f"scenario:{scenario_id}",
                input=getattr(item, "input", None),
                metadata={"scenario_id": scenario_id, "num_conversations": total},
            ) as root:
                outputs: list[str] = []
                overall: list[float] = []
                for conv in conversations:
                    convo_eval = eval_by_conv.get(conv.conversation_id)
                    outputs.append(
                        self._emit_conversation(conv, convo_eval, to_trace=total == 1)
                    )
                    if convo_eval is not None:
                        overall.append(convo_eval.overall_agent_score)

                # For multi-conversation scenarios, attach a trace-level summary
                # so the dataset-run comparison still has a headline number.
                if total > 1 and overall:
                    root.score_trace(
                        name="overall_agent_score_mean",
                        value=sum(overall) / len(overall),
                        data_type="NUMERIC",
                    )

            if not outputs:
                return ""
            return outputs[0] if len(outputs) == 1 else outputs

        experiment_kwargs: dict[str, Any] = {
            "name": run_name or f"arksim-{simulation.simulation_id}",
            "description": run_description
            or f"arksim simulation {simulation.simulation_id} "
            f"({len(simulation.conversations)} conversations)",
            "task": task,
            "evaluators": evaluators or [],
            "metadata": {
                "source": "arksim",
                "simulation_id": simulation.simulation_id,
                "simulator_version": simulation.simulator_version,
                **(run_metadata or {}),
            },
        }
        if run_evaluators:
            experiment_kwargs["run_evaluators"] = run_evaluators
        if max_concurrency is not None:
            experiment_kwargs["max_concurrency"] = max_concurrency

        result = dataset.run_experiment(**experiment_kwargs)
        self._client.flush()
        return result


def export_to_langfuse(
    scenarios: Scenarios,
    simulation: Simulation,
    evaluation: Evaluation | None = None,
    *,
    dataset_name: str,
    client: Any | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    **export_kwargs: Any,
) -> Any:
    """Convenience wrapper: build a :class:`LangfuseExporter` and call ``export``.

    Extra keyword arguments are forwarded to :meth:`LangfuseExporter.export`.
    """
    exporter = LangfuseExporter(
        client=client,
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    return exporter.export(
        scenarios,
        simulation,
        evaluation,
        dataset_name=dataset_name,
        **export_kwargs,
    )
