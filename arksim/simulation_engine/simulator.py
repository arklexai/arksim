# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from jinja2.sandbox import SandboxedEnvironment
from pydantic import ValidationError
from tqdm import tqdm

from arksim.config import AgentConfig, AgentType
from arksim.llms.chat import LLM
from arksim.llms.chat.base.usage import usage_label, usage_scope
from arksim.scenario import (
    KnowledgeItem,
    Scenarios,
)
from arksim.utils.concurrency import resolve_num_workers
from arksim.utils.output import save_json_file_async

if TYPE_CHECKING:
    from arksim.tracing import TraceReceiver

from arksim.tracing.context import _clear_trace_context, _set_trace_context

from .agent.factory import create_agent
from .core import TURN_KNOWLEDGE_FN
from .entities import (
    Conversation,
    ConversationState,
    Message,
    SimulatedUserPrompt,
    Simulation,
    SimulationInput,
    SimulationParams,
    TokenUsage,
)
from .tool_types import AgentResponse, ToolCall, ToolCallSource
from .utils.prompts import DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE
from .utils.utils import flip_hist

logger = logging.getLogger(__name__)


# v1.1: Message.tool_calls field added (optional, backward-compatible with v1)
SIMULATION_SCHEMA_VERSION = "v1.1"
SIMULATOR_VERSION = "v1"


STOP_SIGNAL = "###STOP###"


_KNOWN_TOOL_CALL_SOURCES = {s.value for s in ToolCallSource}


def _tool_call_from_dict(tc: dict[str, Any]) -> ToolCall | None:
    """Build a ToolCall from a serialized dict, tolerating schema drift.

    Old simulation snapshots may carry ``source`` values that are no longer
    in ``ToolCallSource`` (dropped to None with a warning), or be missing
    required fields like ``name`` (returns None with a warning so the rest
    of the snapshot still loads).

    Extra fields on the dict are silently ignored by Pydantic per the
    ``extra="ignore"`` config on ``ToolCall``.
    """
    source = tc.get("source")
    if source is not None and source not in _KNOWN_TOOL_CALL_SOURCES:
        logger.warning(
            "Ignoring unknown tool_call source %r when loading snapshot", source
        )
        tc = {**tc, "source": None}
    try:
        return ToolCall(**tc)
    except ValidationError as e:
        logger.warning(
            "Skipping malformed tool call in snapshot (id=%r): %s",
            tc.get("id"),
            e,
        )
        return None


class Simulator:
    def __init__(
        self,
        agent_config: AgentConfig,
        simulator_params: SimulationParams,
        llm: LLM,
        trace_receiver: TraceReceiver | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.simulator_params = simulator_params
        self.llm = llm
        self.trace_receiver = trace_receiver
        self.simulation: Simulation | None = None

    def _render_simulated_user_prompt(
        self,
        simulated_user_prompt_template: str,
        profile: str,
        goal: str,
        knowledge: list[KnowledgeItem],
        agent_context: str,
    ) -> str:
        """Render the simulated-user system prompt via Jinja2."""
        template = SandboxedEnvironment().from_string(simulated_user_prompt_template)
        return template.render(
            scenario={
                "agent_context": agent_context,
                "goal": goal,
                "knowledge": knowledge,
                "user_profile": profile,
            },
        )

    async def _generate_simulated_user_output(
        self,
        history: list[dict[str, Any]],
        is_multi_knowledge: bool,
        knowledge_content: list[str],
        turn_state: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Generate the simulated user's next message.

        Returns the LLM output and the (possibly updated)
        turn state for multi-knowledge scenarios.
        """
        if is_multi_knowledge:
            current_turn_knowledge, turn_state = await TURN_KNOWLEDGE_FN(
                self.llm,
                history,
                knowledge_content,
                turn_state,
            )
            if current_turn_knowledge:
                knowledge_msg = (
                    f"[Reference knowledge for this turn]:\n{current_turn_knowledge}"
                )
                augmented = list(history) + [{"role": "user", "content": knowledge_msg}]
                return await self.llm.call_async(augmented), turn_state
        return await self.llm.call_async(history), turn_state

    async def _run_single_conversation(
        self,
        profile: str,
        goal: str,
        knowledge: KnowledgeItem | list[KnowledgeItem],
        agent_context: str,
        max_turns: int,
        scenario_id: str = "",
        on_turn_complete: Callable[[], None] | None = None,
        on_turn_display: Callable[[str, str, str, int], None] | None = None,
    ) -> ConversationState | None:
        """Run a single conversation and return its state."""
        knowledge_content: list[str] = [
            kc.content if hasattr(kc, "content") else kc.get("content", "")
            for kc in knowledge
        ]
        is_multi_knowledge = len(knowledge_content) > 1 and any(knowledge_content)

        simulated_user_prompt_template = (
            self.simulator_params.simulated_user_prompt_template
            or DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE
        )
        instructional_prompt = self._render_simulated_user_prompt(
            simulated_user_prompt_template, profile, goal, knowledge, agent_context
        )

        agent = create_agent(self.agent_config)
        try:
            conversation_id = await agent.get_chat_id()
            logger.info(f"Starting conversation {conversation_id} with goal: {goal}")

            history: list[dict[str, Any]] = [
                {"role": "system", "content": instructional_prompt}
            ]
            metadata: dict[str, Any] = {
                "chat_id": conversation_id,
                "user_goal": goal,
                "knowledge": knowledge,
            }

            turn_state: dict[str, Any] = {}
            with usage_label(
                component="conversation",
                conversation_id=conversation_id,
            ):
                for turn in range(max_turns):
                    output, turn_state = await self._generate_simulated_user_output(
                        history,
                        is_multi_knowledge,
                        knowledge_content,
                        turn_state,
                    )

                    history.append({"role": "assistant", "content": output})
                    if on_turn_complete:
                        on_turn_complete()

                    if STOP_SIGNAL in output:
                        logger.info(
                            f"Conversation {conversation_id} stopped "
                            f"at turn {turn + 1} (STOP signal)"
                        )
                        break

                    metadata["turn_id"] = turn

                    # Set trace routing context so processors can read it
                    # without the agent passing it explicitly.
                    if self.trace_receiver is not None:
                        _set_trace_context(conversation_id, turn, self.trace_receiver)

                    try:
                        result = await agent.execute(
                            user_query=output,
                            metadata=metadata,
                        )
                    finally:
                        if self.trace_receiver is not None:
                            self.trace_receiver.signal_turn_complete(
                                conversation_id, turn
                            )
                            _clear_trace_context()

                    # Normalize response
                    if isinstance(result, AgentResponse):
                        answer = result.content
                        turn_tool_calls = result.tool_calls or []
                    else:
                        answer = result
                        turn_tool_calls = []

                    # Merge tool calls from trace receiver (if active).
                    # Dedup by ID first, then by (name, arguments) to handle
                    # cases where the trace receiver falls back to spanId while
                    # AgentResponse carries an SDK-assigned tool call ID.
                    if self.trace_receiver is not None:
                        traced = await self.trace_receiver.wait_for_traces(
                            conversation_id, turn
                        )
                        if traced:
                            existing_ids = {tc.id for tc in turn_tool_calls}
                            sig_to_idx: dict[tuple[str, str], int] = {}
                            for i, tc in enumerate(turn_tool_calls):
                                sig = (
                                    tc.name,
                                    json.dumps(tc.arguments, sort_keys=True),
                                )
                                sig_to_idx[sig] = i

                            for tc in traced:
                                sig = (
                                    tc.name,
                                    json.dumps(tc.arguments, sort_keys=True),
                                )
                                if tc.id not in existing_ids and sig not in sig_to_idx:
                                    turn_tool_calls.append(tc)

                    history.append(
                        {
                            "role": "user",
                            "content": answer,
                            **(
                                {
                                    "tool_calls": [
                                        tc.model_dump() for tc in turn_tool_calls
                                    ]
                                }
                                if turn_tool_calls
                                else {}
                            ),
                        }
                    )

                    if on_turn_display:
                        on_turn_display(conversation_id, output, answer, turn + 1)

            logger.info(
                f"Conversation {conversation_id} completed "
                f"with {len(history) - 1} messages"
            )

            return ConversationState(
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                conversation_history=history,
                simulated_user_prompt_template=simulated_user_prompt_template,
                simulated_user_profile=profile,
                user_goal=goal,
                knowledge=knowledge_content,
                agent_context=agent_context,
            )
        except Exception as e:
            logger.error(f"Error simulating conversation: {str(e)}")
            return None
        finally:
            await agent.close()

    def _to_conversation_output(self, state: ConversationState) -> Conversation:
        """Convert internal state to an output-format conversation."""
        flipped = flip_hist(state.conversation_history)

        messages: list[Message] = []
        turn_id = 0
        for msg in flipped:
            role = msg.get("role", "")
            if role == "user":
                messages.append(
                    Message(
                        turn_id=turn_id,
                        role="simulated_user",
                        content=msg.get("content", ""),
                    )
                )
            elif role == "assistant":
                raw_tcs = msg.get("tool_calls")
                if raw_tcs:
                    parsed = [_tool_call_from_dict(tc) for tc in raw_tcs]
                    tool_calls = [tc for tc in parsed if tc is not None] or None
                else:
                    tool_calls = None
                messages.append(
                    Message(
                        turn_id=turn_id,
                        role="assistant",
                        content=msg.get("content", ""),
                        tool_calls=tool_calls,
                    )
                )
                turn_id += 1

        simulated_user_prompt = SimulatedUserPrompt(
            simulated_user_prompt_template=state.simulated_user_prompt_template,
            variables={
                "scenario.agent_context": state.agent_context,
                "scenario.goal": state.user_goal,
                "scenario.knowledge": state.knowledge,
                "scenario.user_profile": state.simulated_user_profile,
            },
        )

        return Conversation(
            conversation_id=state.conversation_id,
            scenario_id=state.scenario_id,
            conversation_history=messages,
            simulated_user_prompt=simulated_user_prompt,
        )

    async def simulate(
        self,
        scenarios: Scenarios,
        on_progress: Callable[[int, int], None] | None = None,
        verbose: bool = False,
    ) -> Simulation:
        num_convos = self.simulator_params.num_convos_per_scenario * len(
            scenarios.scenarios
        )
        max_turns = self.simulator_params.max_turns
        estimated_total_turns = num_convos * max_turns

        num_workers = resolve_num_workers(self.simulator_params.num_workers, num_convos)
        logger.info(f"Preparing {len(scenarios.scenarios)} scenarios for simulation")

        pbar = tqdm(
            total=estimated_total_turns,
            desc="Simulating conversations",
            disable=verbose,
        )

        def _verbose_turn_display(
            convo_id: str, user_msg: str, agent_msg: str, turn: int
        ) -> None:
            logger.info(f"\n[{convo_id}] Turn {turn}")
            logger.info(f"  Simulated User: {user_msg}")
            logger.info(f"  Agent: {agent_msg}")

        on_turn_display = _verbose_turn_display if verbose else None

        def on_turn_complete() -> None:
            pbar.update(1)
            if on_progress:
                on_progress(pbar.n, estimated_total_turns)

        running_tasks = set()
        results: list[ConversationState] = []

        for scenario in scenarios.scenarios:
            for _ in range(self.simulator_params.num_convos_per_scenario):
                if len(running_tasks) >= num_workers:
                    done, running_tasks = await asyncio.wait(
                        running_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        try:
                            result = task.result()
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing conversation: {str(e)}")
                            logger.error(traceback.format_exc())

                coro = self._run_single_conversation(
                    scenario.user_profile,
                    scenario.goal,
                    scenario.knowledge,
                    scenario.agent_context,
                    max_turns,
                    scenario_id=scenario.scenario_id,
                    on_turn_complete=on_turn_complete,
                    on_turn_display=on_turn_display,
                )
                running_tasks.add(asyncio.create_task(coro))

        if running_tasks:
            done, _ = await asyncio.wait(running_tasks)
            for task in done:
                try:
                    result = task.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing conversation: {str(e)}")
                    logger.error(traceback.format_exc())

        if pbar.n < pbar.total:
            pbar.total = pbar.n
            pbar.refresh()
            if on_progress:
                on_progress(pbar.n, pbar.n)
        pbar.close()

        # Convert internal states to output conversations
        conversation_outputs = [
            self._to_conversation_output(state) for state in results
        ]
        conversation_outputs.sort(key=lambda c: c.conversation_id)

        self.simulation = Simulation(
            schema_version=SIMULATION_SCHEMA_VERSION,
            simulator_version=SIMULATOR_VERSION,
            conversations=conversation_outputs,
        )

        total_turns = sum(
            len([m for m in c.conversation_history if m.role == "assistant"])
            for c in conversation_outputs
        )
        logger.info(
            f"Simulation complete: {len(conversation_outputs)} conversations, "
            f"{total_turns} total turns"
        )
        return self.simulation

    async def save(self) -> None:
        if self.simulation is None:
            logger.error("Attempted to save simulation but output is not available")
            raise ValueError("Simulation data is not available")

        output_path = self.simulator_params.output_file_path
        logger.info(f"Saving conversations to {output_path}")

        await save_json_file_async(
            self.simulation.model_dump(), output_path, overwrite=True
        )
        logger.info("Simulation saved successfully")


async def run_simulation(
    settings: SimulationInput,
    scenarios: Scenarios | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> Simulation:
    if scenarios is None:
        scenarios = Scenarios.load(settings.scenario_file_path)

    if settings.agent_config is not None:
        # Inline agent config from YAML (already validated by pydantic)
        agent_config = settings.agent_config
    else:
        # Backward compat: load from separate JSON file
        agent_config = AgentConfig.load(settings.agent_config_file_path)

    llm = LLM(
        model=settings.model,
        provider=settings.provider,
    )

    simulation_params = SimulationParams(
        num_convos_per_scenario=settings.num_conversations_per_scenario,
        max_turns=settings.max_turns,
        num_workers=settings.num_workers,
        output_file_path=settings.output_file_path,
        simulated_user_prompt_template=settings.simulated_user_prompt_template,
    )

    # Start trace receiver if configured (lazy import to avoid circular deps)
    trace_cfg = settings.trace_receiver
    trace_receiver: TraceReceiver | None = None
    if trace_cfg and trace_cfg.enabled:
        from arksim.tracing import TraceReceiver as _TraceReceiver

        trace_receiver = _TraceReceiver(
            host=trace_cfg.host,
            port=trace_cfg.port,
            wait_timeout=trace_cfg.wait_timeout,
        )

        # Same-process agents use direct injection via contextvars,
        # so no HTTP listener is needed. Cross-process agents (Chat
        # Completions, A2A) need the HTTP endpoint for OTel export.
        is_same_process = agent_config.agent_type == AgentType.CUSTOM.value
        await trace_receiver.start(start_http=not is_same_process)

    try:
        simulator = Simulator(
            agent_config=agent_config,
            simulator_params=simulation_params,
            llm=llm,
            trace_receiver=trace_receiver,
        )

        with usage_scope() as tracker:
            simulation_output = await simulator.simulate(
                scenarios, on_progress=on_progress, verbose=verbose
            )

        if not simulation_output.conversations:
            raise RuntimeError(
                "Simulation failed: no conversations were completed successfully. "
                "Check the errors above for details."
            )

        simulation_output.usage = TokenUsage(
            total_input_tokens=tracker.total_input_tokens,
            total_output_tokens=tracker.total_output_tokens,
            total_cached_tokens=tracker.total_cached_tokens,
            total_reasoning_tokens=tracker.total_reasoning_tokens,
            by_model=tracker.summary(),
            breakdowns={
                "by_conversation": tracker.summary_by("conversation_id"),
            },
        )
        tracker.log_summary()

        await simulator.save()
        return simulation_output
    finally:
        if trace_receiver is not None:
            await trace_receiver.stop()
