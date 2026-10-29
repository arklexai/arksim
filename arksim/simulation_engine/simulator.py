# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx
from jinja2.sandbox import SandboxedEnvironment
from pydantic import ValidationError
from tqdm import tqdm

from arksim.config import AgentConfig, AgentType
from arksim.llms.chat import LLM
from arksim.scenario import Scenario, Scenarios
from arksim.utils.concurrency import resolve_num_workers
from arksim.utils.output import save_json_file_async

if TYPE_CHECKING:
    from arksim.tracing import TraceReceiver

from arksim.tracing.context import (
    _clear_trace_context,
    _set_trace_context,
    reset_request_headers,
    set_request_headers,
)

from .agent.factory import create_agent
from .core import ConversationLedger, generate_utterance, judge_turn
from .entities import (
    Conversation,
    ConversationExecutionError,
    ConversationState,
    Message,
    SimulatedUserPrompt,
    Simulation,
    SimulationInput,
    SimulationParams,
    TurnHooks,
)
from .tool_types import AgentResponse, ToolCall, ToolCallSource
from .utils.prompts import DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE
from .utils.utils import flip_hist

logger = logging.getLogger(__name__)


# v2: every conversation runs a judged turn loop (see run_single_conversation).
SIMULATION_SCHEMA_VERSION = "v2"
SIMULATOR_VERSION = "v2"


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
        *,
        agent_http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.simulator_params = simulator_params
        self.llm = llm
        self.trace_receiver = trace_receiver
        self.agent_http_client_factory = agent_http_client_factory
        self.simulation: Simulation | None = None

    def _render_simulated_user_prompt(
        self,
        simulated_user_prompt_template: str,
        scenario: Scenario,
    ) -> str:
        """Render the simulated-user system prompt via Jinja2."""
        template = SandboxedEnvironment().from_string(simulated_user_prompt_template)
        return template.render(
            scenario={
                "user_profile": scenario.user_profile,
            },
            web_search=self.simulator_params.web_search,
        )

    async def run_single_conversation(
        self,
        scenario: Scenario,
        max_turns: int,
        on_turn_complete: Callable[[], None] | None = None,
        on_turn_display: Callable[[str, str, str, int], None] | None = None,
        *,
        chat_id: str | None = None,
        turn_hooks: TurnHooks | None = None,
    ) -> ConversationState | None:
        """Run a single conversation and return its state.

        ``chat_id`` overrides ``agent.get_chat_id()`` as the conversation id.
        ``turn_hooks`` (see ``TurnHooks``) run before and after each
        ``agent.execute()``; headers returned by the before-hook are applied
        to that single agent request via the ``request_headers`` contextvar.

        The system prompt is fixed for the whole conversation and renders
        only ``user_profile`` - ``agent_context``/``goal`` never reach the
        simulated user (goal only reaches the generator decomposed into the
        directive's ``pending_checklist``); per-turn state (patience,
        pending checklist) rides on an ephemeral directive instead (see
        ``core.utterance.generate_utterance``).
        The simulated user never sees the scenario's ground-truth
        ``knowledge``. A turn judge updates a ``ConversationLedger`` after
        every agent reply, and termination is decided entirely in code
        (``user_satisfied``, ``gave_up``, ``blocked``, or ``max_turns``) -
        never by the generator. A scenario with an empty ``goal_checklist``
        (uncompiled) simply has no mandatory sub-goal to satisfy, so it
        runs to ``gave_up`` or ``max_turns``.
        """
        knowledge_content: list[str] = [kc.content for kc in scenario.knowledge]
        ledger = ConversationLedger.from_scenario(scenario)

        simulated_user_prompt_template = (
            self.simulator_params.simulated_user_prompt_template
            or DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE
        )
        instructional_prompt = self._render_simulated_user_prompt(
            simulated_user_prompt_template, scenario
        )

        agent = create_agent(
            self.agent_config, http_client_factory=self.agent_http_client_factory
        )
        try:
            conversation_id = chat_id or await agent.get_chat_id()
            logger.info(
                f"Starting conversation {conversation_id} with goal: {scenario.goal}"
            )

            history: list[dict[str, Any]] = [
                {"role": "system", "content": instructional_prompt}
            ]
            metadata: dict[str, Any] = {
                "chat_id": conversation_id,
                "user_goal": scenario.goal,
                "knowledge": scenario.knowledge,
            }

            for turn in range(max_turns):
                output = await generate_utterance(
                    self.llm,
                    scenario,
                    ledger,
                    history,
                    turn,
                    web_search=self.simulator_params.web_search,
                )

                history.append({"role": "assistant", "content": output})
                if on_turn_complete:
                    on_turn_complete()

                metadata["turn_id"] = turn

                # A before-hook failure propagates here, before the request.
                headers = (
                    await turn_hooks.before_agent_execute(turn)
                    if turn_hooks is not None
                    else None
                )

                error: BaseException | None = None
                headers_token = None
                try:
                    headers_token = set_request_headers(headers)
                    # Set trace routing context so processors can read it
                    # without the agent passing it explicitly.
                    if self.trace_receiver is not None:
                        _set_trace_context(conversation_id, turn, self.trace_receiver)
                    result = await agent.execute(
                        user_query=output,
                        metadata=metadata,
                    )
                except BaseException as e:
                    error = e
                    raise
                finally:
                    if headers_token is not None:
                        reset_request_headers(headers_token)
                    if self.trace_receiver is not None:
                        self.trace_receiver.signal_turn_complete(conversation_id, turn)
                        _clear_trace_context()
                    if turn_hooks is not None:
                        try:
                            await turn_hooks.after_agent_execute(turn, error)
                        except Exception:
                            # Never mask the in-flight exception (or fail a
                            # successful turn) because a hook failed.
                            logger.exception(
                                "after_agent_execute hook failed for turn %d", turn
                            )

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
                #
                # Empty-string ids fall through to signature-based dedup because
                # Gemini's native response format has no per-call id field, and
                # the A2A tool-call extension treats id as optional. Both
                # parsers fall back to "" by convention; the convention is
                # anchored by ``ToolCall.id: str`` (not nullable), so a missing
                # id always lands as "" and never as None.
                #
                # existing_ids and sig_to_idx are updated as traced calls are
                # appended so two same-id or same-signature traced calls
                # dedupe against each other within the loop.
                if self.trace_receiver is not None:
                    traced = await self.trace_receiver.wait_for_traces(
                        conversation_id, turn
                    )
                    if traced:
                        existing_ids = {tc.id for tc in turn_tool_calls if tc.id}
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
                                sig_to_idx[sig] = len(turn_tool_calls) - 1
                                if tc.id:
                                    existing_ids.add(tc.id)

                history.append(
                    {
                        "role": "user",
                        "content": answer,
                        **(
                            {"tool_calls": [tc.model_dump() for tc in turn_tool_calls]}
                            if turn_tool_calls
                            else {}
                        ),
                    }
                )

                if on_turn_display:
                    on_turn_display(conversation_id, output, answer, turn + 1)

                verdict = await judge_turn(
                    self.llm, scenario, ledger, history, answer, turn
                )
                if verdict is not None:
                    ledger.apply(verdict, turn)
                reason = ledger.should_terminate()
                if reason is not None:
                    ledger.termination_reason = reason
                    logger.info(
                        f"Conversation {conversation_id} terminated "
                        f"at turn {turn + 1} ({reason})"
                    )
                    break

            if ledger.termination_reason is None:
                ledger.termination_reason = "max_turns"

            logger.info(
                f"Conversation {conversation_id} completed "
                f"with {len(history) - 1} messages"
            )

            return ConversationState(
                conversation_id=conversation_id,
                scenario_id=scenario.scenario_id,
                conversation_history=history,
                simulated_user_prompt_template=simulated_user_prompt_template,
                simulated_user_profile=scenario.user_profile,
                user_goal=scenario.goal,
                knowledge=knowledge_content,
                agent_context=scenario.agent_context,
                ledger=ledger,
            )
        except Exception as e:
            # Surface the real error (e.g. the connector's "HTTP 429: ..."
            # RuntimeError) instead of a silent None, and carry any turns
            # completed before the failure so the caller can persist a partial
            # transcript. The batch runner in run_num_conversations catches this
            # like any Exception, so its behavior is unchanged.
            logger.exception("Error simulating conversation")
            partial_state: ConversationState | None = None
            try:
                if history and len(history) > 1:
                    partial_state = ConversationState(
                        conversation_id=conversation_id,
                        scenario_id=scenario.scenario_id,
                        conversation_history=history,
                        simulated_user_prompt_template=simulated_user_prompt_template,
                        simulated_user_profile=scenario.user_profile,
                        user_goal=scenario.goal,
                        knowledge=knowledge_content,
                        agent_context=scenario.agent_context,
                    )
            except NameError:
                # Failure occurred before history/conversation_id were assigned.
                partial_state = None
            raise ConversationExecutionError(str(e), partial_state) from e
        finally:
            await agent.close()

    def to_conversation_output(self, state: ConversationState) -> Conversation:
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

        # Only user_profile is an actual prompt variable now (see
        # run_single_conversation); the other ConversationState fields still
        # carry their values, just not here.
        simulated_user_prompt = SimulatedUserPrompt(
            simulated_user_prompt_template=state.simulated_user_prompt_template,
            variables={
                "scenario.user_profile": state.simulated_user_profile,
                "web_search": self.simulator_params.web_search,
            },
        )

        return Conversation(
            conversation_id=state.conversation_id,
            scenario_id=state.scenario_id,
            conversation_history=messages,
            simulated_user_prompt=simulated_user_prompt,
            ledger=state.ledger,
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

                coro = self.run_single_conversation(
                    scenario,
                    max_turns,
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
        conversation_outputs = [self.to_conversation_output(state) for state in results]
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

    web_search = settings.web_search
    if web_search and not llm.supports_web_search:
        logger.warning(
            "Provider '%s' has no hosted web fetch or search tool; the "
            "simulated user will run without it.",
            settings.provider,
        )
        web_search = False

    simulation_params = SimulationParams(
        num_convos_per_scenario=settings.num_conversations_per_scenario,
        max_turns=settings.max_turns,
        num_workers=settings.num_workers,
        output_file_path=settings.output_file_path,
        simulated_user_prompt_template=settings.simulated_user_prompt_template,
        web_search=web_search,
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

        simulation_output = await simulator.simulate(
            scenarios, on_progress=on_progress, verbose=verbose
        )

        if not simulation_output.conversations:
            raise RuntimeError(
                "Simulation failed: no conversations were completed successfully. "
                "Check the errors above for details."
            )

        await simulator.save()
        return simulation_output
    finally:
        if trace_receiver is not None:
            await trace_receiver.stop()
