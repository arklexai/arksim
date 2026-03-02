import asyncio
import logging
import os
import traceback
from collections.abc import Callable
from typing import Any

from jinja2.sandbox import SandboxedEnvironment
from tqdm import tqdm

from arksim.config import AgentConfig
from arksim.llms.chat import LLM
from arksim.scenario import KnowledgeItem, Scenario, Scenarios
from arksim.utils.concurrency import resolve_num_workers
from arksim.utils.output import save_json_file_async

from .agent.factory import create_agent
from .core import (
    TURN_KNOWLEDGE_FN,
    convert_attribute_to_profile_async,
    find_matched_attribute_async,
)
from .entities import (
    Conversation,
    ConversationState,
    Message,
    SimulatedUserPrompt,
    Simulation,
    SimulationInput,
    SimulationParams,
)
from .utils.prompts import DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE
from .utils.utils import flip_hist

logger = logging.getLogger(__name__)


SIMULATION_SCHEMA_VERSION = "v1"
SIMULATOR_VERSION = "v1"


STOP_SIGNAL = "###STOP###"


class Simulator:
    def __init__(
        self,
        agent_config: AgentConfig,
        simulator_params: SimulationParams,
        llm: LLM,
    ) -> None:
        self.agent_config = agent_config
        self.simulator_params = simulator_params
        self.llm = llm
        self.simulation: Simulation | None = None

    async def generate_profiles(
        self, scenario_output: Scenarios
    ) -> list[dict[str, Any]]:
        """Generate profiles for the scenario data."""

        scenarios = scenario_output.scenarios
        logger.info(f"Generating profiles for {len(scenarios)} scenarios")
        num_workers = resolve_num_workers(
            self.simulator_params.num_workers, len(scenarios)
        )
        semaphore = asyncio.Semaphore(num_workers)

        async def _generate_single_profile(scenario: Scenario) -> dict[str, Any]:
            async with semaphore:
                attribute_options = list(scenario.user_attributes.keys())
                label = await find_matched_attribute_async(
                    self.llm, scenario.goal, scenario.user_attributes, attribute_options
                )
                profile = await convert_attribute_to_profile_async(
                    self.llm,
                    scenario.user_attributes,
                    label,
                    scenario.agent_context,
                )
                return {
                    "scenario_id": scenario.scenario_id,
                    "goal": scenario.goal,
                    "knowledge": scenario.knowledge,
                    "profile": profile,
                    "attribute": scenario.user_attributes,
                    "agent_context": scenario.agent_context,
                }

        tasks = [_generate_single_profile(scenario) for scenario in scenarios]
        profiles = []
        with tqdm(total=len(scenarios), desc="Generating profiles") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    profiles.append(await coro)
                except Exception as e:
                    logger.error(f"Error generating profile: {str(e)}")
                finally:
                    pbar.update(1)

        return profiles

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
            },
            simulation={"profile": profile},
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
        attribute: dict[str, Any],
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
            metadata = {
                "chat_id": conversation_id,
                "user_attributes": attribute,
                "user_goal": goal,
                "knowledge": knowledge,
            }

            turn_state: dict[str, Any] = {}
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

                answer = await agent.execute(
                    user_query=output,
                    metadata=metadata,
                )
                history.append({"role": "user", "content": answer})

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
                messages.append(
                    Message(
                        turn_id=turn_id,
                        role="assistant",
                        content=msg.get("content", ""),
                    )
                )
                turn_id += 1

        simulated_user_prompt = SimulatedUserPrompt(
            simulated_user_prompt_template=state.simulated_user_prompt_template,
            variables={
                "scenario.agent_context": state.agent_context,
                "scenario.goal": state.user_goal,
                "scenario.knowledge": state.knowledge,
                "simulation.profile": state.simulated_user_profile,
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

        profiles = await self.generate_profiles(scenarios)
        num_workers = resolve_num_workers(self.simulator_params.num_workers, num_convos)

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

        for scenario in profiles:
            profile = scenario["profile"]
            goal = scenario["goal"]
            knowledge = scenario["knowledge"]
            attribute = scenario["attribute"]
            agent_context = scenario["agent_context"]
            scenario_id = scenario["scenario_id"]
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
                profile,
                goal,
                knowledge,
                agent_context,
                attribute,
                max_turns,
                scenario_id=scenario_id,
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
    scenario_output: Scenarios | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> Simulation:
    if scenario_output is None:
        scenario_output = Scenarios.load(settings.scenario_file_path)

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

    simulator = Simulator(
        agent_config=agent_config,
        simulator_params=simulation_params,
        llm=llm,
    )
    simulation_output = await simulator.simulate(
        scenario_output, on_progress=on_progress, verbose=verbose
    )
    await simulator.save()
    return simulation_output
