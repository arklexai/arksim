# arksim Claude Code Integration - E2E Test Report

**Date:** 2026-04-09 17:37 UTC
**Python:** 3.11.13
**Transport:** MCP stdio (official SDK client)
**Result:** 19 passed, 0 failed

### Test Environment

- Working directory: `/private/tmp/arksim-smoke-test`
- Python: `/opt/homebrew/opt/python@3.11/bin/python3.11` (3.11.13)
- OPENAI_API_KEY: set
- ANTHROPIC_API_KEY: set

### 1. MCP Server Initialization

MCP session initialized successfully via stdio transport.

Registered tools: `evaluate, init_project, launch_ui, list_results, read_result, simulate_evaluate`
Expected tools:   `evaluate, init_project, launch_ui, list_results, read_result, simulate_evaluate`
Match: **PASS**

### 2. User Journey 1: First-Time Setup (`/arksim test` hero flow)

Simulating: User types `/arksim test` with no existing config.
The skill would detect no config and call `init_project`.

**`init_project`** (PASS)

```json
// Input
{
  "agent_type": "custom",
  "directory": "/private/tmp/arksim-smoke-test"
}
```

```json
// Output
{
  "status": "success",
  "output": "",
  "message": "Project initialized with agent type 'custom'."
}
```

Scaffolded files: config.yaml: exists, scenarios.json: exists, my_agent.py: exists

Scenarios file loaded: 4 scenarios (schema_version: v1)

**Running real simulation + evaluation** (this calls the actual arksim CLI with LLM API)...
Using gpt-4o-mini with 1 conversation per scenario and 2 max turns for speed.

**`simulate_evaluate`** (PASS)

```json
// Input
{
  "config_path": "/private/tmp/arksim-smoke-test/config.yaml",
  "cli_overrides": {
    "model": "gpt-4o-mini",
    "num_conversations_per_scenario": "1",
    "max_turns": "2"
  }
}
```

```json
// Output
{
  "status": "success",
  "output": "",
  "stderr": "2026-04-09 10:37:20 - arksim - INFO - \nSimulation configuration:\n2026-04-09 10:37:20 - arksim - INFO -   agent_config: {'agent_name': 'my-agent', 'agent_type': 'custom', 'api_config': None, 'custom_config': {'module_path': '/private/tmp/arksim-smoke-test/my_agent.py', 'class_name': None, 'agent_class': None}}\n2026-04-09 10:37:20 - arksim - INFO -   agent_config_file_path: None\n2026-04-09 10:37:20 - arksim - INFO -   max_turns: 2\n2026-04-09 10:37:20 - arksim - INFO -   model: gpt-4o-mini\n2026-04-09 10:37:20 - arksim - INFO -   num_conversations_per_scenario: 1\n2026-04-09 10:37:20 - arksim - INFO -   num_workers: 50\n2026-04-09 10:37:20 - arksim - INFO -   output_file_path: /private/tmp/arksim-smoke-test/simulation.json\n2026-04-09 10:37:20 - arksim - INFO -   provider: openai\n2026-04-09 10:37:20 - arksim - INFO -   scenario_file_path: /private/tmp/arksim-smoke-test/scenarios.json\n2026-04-09 10:37:20 - arksim - INFO -   simulated_user_prompt_template: None\n2026-04-09 10:37:20 - arksim - INFO -   trace_receiver: None\n2026-04-09 10:37:20 - arksim - INFO - \n2026-04-09 10:37:20 - arksim - INFO - \nEvaluation configuration:\n2026-04-09 10:37:20 - arksim - INFO -   custom_metrics_file_paths: []\n2026-04-09 10:37:20 - arksim - INFO -   generate_html_report: True\n2026-04-09 10:37:20 - arksim - INFO -   metrics_to_run: ['faithfulness', 'helpfulness', 'coherence', 'verbosity', 'relevance', 'goal_completion', 'agent_behavior_failure']\n2026-04-09 10:37:20 - arksim - INFO -   model: gpt-4o-mini\n2026-04-09 10:37:20 - arksim - INFO -   num_workers: 50\n2026-04-09 10:37:20 - arksim - INFO -   numeric_thresholds: None\n2026-04-09 10:37:20 - arksim - INFO -   output_dir: /private/tmp/arksim-smoke-test/results\n2026-04-09 10:37:20 - arksim - INFO -   provider: openai\n2026-04-09 10:37:20 - arksim - INFO -   qualitative_failure_labels: None\n2026-04-09 10:37:20 - arksim - INFO -   scenario_file_path: /private/tmp/arksim-smoke-test/scenarios.json\n2026-04-09 10:37:20 - arksim - INFO -   score_threshold: None\n2026-04-09 10:37:20 - arksim - INFO -   simulation_file_path: None\n2026-04-09 10:37:20 - arksim - INFO - \n2026-04-09 10:37:20 - arksim.simulation_engine.simulator - INFO - Preparing 4 scenarios for simulation\n\nSimulating conversations:   0%|          | 0/8 [00:00<?, ?it/s]2026-04-09 10:37:21 - arksim.simulation_engine.simulator - INFO - Starting conversation 57b9811d-149b-41df-88c7-4609de1564dd with goal: You have a straightforward question that the agent should be able to answer. Ask clearly, provide any details the agent requests, and evaluate whether the response is helpful and accurate.\n2026-04-09 10:37:21 - arksim.simulation_engine.simulator - INFO - Starting conversation 2cb373d8-98df-4209-990c-8b339dd4953a with goal: You want help with something the agent is not designed to handle. Ask about an unrelated topic (e.g. medical advice, legal questions, or something clearly outside the agent's domain). A good agent should politely decline or redirect you rather than making something up.\n2026-04-09 10:37:21 - arksim.simulation_engine.simulator - INFO - Starting conversation 4dd7c451-5281-454d-986a-19a06aa0432a with goal: You have a request that could be interpreted multiple ways. State it vaguely on purpose. If the agent picks one interpretation and runs with it, ask about the other interpretation. A good agent should ask for clarification before committing to an answer.\n2026-04-09 10:37:21 - arksim.simulation_engine.simulator - INFO - Starting conversation 2792f1d1-b1d5-4d72-9091-382c29b1ca3c with goal: You have a goal that requires multiple steps to accomplish. Start by explaining what you want at a high level. When the agent responds, ask a follow-up question to clarify one of the details. Then change your mind about one aspect and ask the agent to adjust. A good agent should track context across turns and handle the revision gracefully.\n\nSimulating conversations:  12%|\u2588\u258e        | 1/8 [00:01<00:08,  1.24s/it]\nSimulating conversations:  50%|\u2588\u2588\u2588\u2588\u2588     | 4/8 [00:01<00:01,  2.76it/s]\nSimulating conversations:  62%|\u2588\u2588\u2588\u2588\u2588\u2588\u258e   | 5/8 [00:02<00:01,  2.46it/s]2026-04-09 10:37:23 - arksim.simulation_engine.simulator - INFO - Conversation 4dd7c451-5281-454d-986a-19a06aa0432a completed with 4 messages\n2026-04-09 10:37:23 - arksim.simulation_engine.simulator - INFO - Conversation 2cb373d8-98df-4209-990c-8b339dd4953a completed with 4 messages\n\nSimulating conversations:  88%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258a | 7/8 [00:02<00:00,  3.02it/s]2026-04-09 10:37:23 - arksim.simulation_engine.simulator - INFO - Conversation 2792f1d1-b1d5-4d72-9091-382c29b1ca3c completed with 4 messages\n\nSimulating conversations: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 8/8 [00:03<00:00,  2.88it/s]2026-04-09 10:37:24 - arksim.simulation_engine.simulator - INFO - Conversation 57b9811d-149b-41df-88c7-4609de1564dd completed with 4 messages\n\nSimulating conversations: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 8/8 [00:03<00:00,  2.58it/s]\n2026-04-09 10:37:24 - arksim.simulation_engine.simulator - INFO - Simulation complete: 4 conversations, 8 total turns\n2026-04-09 10:37:24 - arksim.simulation_engine.simulator - INFO - Saving conversations to /private/tmp/arksim-smoke-test/simulation.json\n2026-04-09 10:37:24 - arksim.simulation_engine.simulator - INFO - Simulation saved successfully\n2026-04-09 10:37:24 - arksim - INFO - Simulation completed in 3.34 seconds\n2026-04-09 10:37:24 - arksim.evaluator.evaluator - INFO - Evaluator initialized: num_workers=50\n2026-04-09 10:37:24 - arksim.evaluator.evaluator - INFO - Starting evaluation of 4 conversations\n2026-04-09 10:37:24 - arksim.evaluator.evaluator - INFO - Preprocessing complete: 8 total turns to evaluate\n\nEvaluating...:   0%|          | 0/13 [00:00<?, ?step/s]\nEvaluating...:   8%|\u258a         | 1/13 [00:04<00:55,  4.64s/step]\nEvaluating...:  31%|\u2588\u2588\u2588       | 4/13 [00:04<00:08,  1.10step/s]\nEvaluating...:  46%|\u2588\u2588\u2588\u2588\u258c     | 6/13 [00:04<00:03,  1.77step/s]\nEvaluating...:  62%|\u2588\u2588\u2588\u2588\u2588\u2588\u258f   | 8/13 [00:05<00:01,  2.59step/s]\nEvaluating...:  77%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258b  | 10/13 [00:07<00:01,  1.75step/s]\nDetecting agent errors:  92%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258f| 12/13 [00:07<00:00,  1.75step/s]2026-04-09 10:37:31 - arksim.evaluator.evaluator - INFO - Detecting agent errors\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Detected 5 unique errors\n\nDetecting agent errors: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:14<00:00,  1.35s/step]\nEvaluation complete: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:14<00:00,  1.35s/step]   \nEvaluation complete: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:14<00:00,  1.09s/step]\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Post-processing complete\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Evaluation complete: 4 conversations, 8 turns, 5 unique errors\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Saving evaluation results to /private/tmp/arksim-smoke-test/results/evaluation.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Evaluation results saved successfully\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Generated 5 focus file(s) in /private/tmp/arksim-smoke-test/results/focus/\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Displaying evaluation summary\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n============================================================\nEVALUATION SUMMARY\n============================================================\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Conversations: 4\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Total Turns: 8\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Average Turns per Conversation: 2.0\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nTURN-BY-TURN EVALUATION:\n------------------------\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Note: Scores range from 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance.\n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Conversation 1:\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant repeated the user's question back instead of providing...), Coherence: 1.0 (The response is completely incoherent, as it simply repeats the...), Helpfulness: 1.0 (The response completely missed the essence of the user's query...), Faithfulness: 1.0 (The assistant's response does not address the user's question about...), Relevance: 1.0 (The AI assistant's response is not relevant at all to...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: repetition (The assistant repeated the same information from the previous turn...), Relevance: 1.0 (The assistant's last message simply repeats the user's previous message...), Faithfulness: 1.0 (The assistant's response is completely in conflict with the knowledge,...), Helpfulness: 1.0 (The response is a verbatim repetition of the user's message,...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Conversation 2:\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant simply repeated the user's statement without providing any...), Relevance: 1.0 (The response simply repeats the user's query without providing any...), Helpfulness: 1.0 (The response does not provide any recommendations or information regarding...), Faithfulness: 1.0 (The assistant's response merely repeats the user's question without providing...), Coherence: 1.0 (The response simply repeats the user's query without providing any...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: disobey user request (The agent repeated the user's original statement instead of addressing...), Coherence: 1.0 (The response is completely incoherent, as it repeats the user's...), Verbosity: 1.0 (The response is excessively verbose, repeating the user's previous message...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Conversation 3:\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: failure to ask for clarification (The user's request is vague, mentioning 'that thing about the...), Relevance: 1.0 (The assistant's response simply repeats the user's request without providing...), Helpfulness: 1.0 (The assistant's response simply repeats the user's query without offering...), Coherence: 1.0 (The AI assistant's response is completely incoherent and provides no...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: failure to ask for clarification (The user's request was vague, and the assistant did not...), Verbosity: 1.0 (The last message from the AI assistant is excessively lengthy...), Relevance: 1.0 (The assistant's response simply repeated the user's previous message without...), Coherence: 1.0 (The AI assistant's last message repeats the user's question verbatim,...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Conversation 4:\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant repeated the user's question without addressing the request...), Helpfulness: 1.0 (The response does not provide any helpful information or address...), Coherence: 1.0 (The assistant's response is completely incoherent and merely repeats the...), Faithfulness: 1.0 (The assistant's response does not provide any information related to...), Relevance: 1.0 (The response from the AI assistant does not address the...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: disobey user request (The assistant repeated the user's statement instead of properly acknowledging...), Relevance: 1.0 (The assistant's response does not address the user's question about...), Helpfulness: 1.0 (The assistant's responses are not helpful as they do not...)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nCONVERSATION-LEVEL METRICS:\n---------------------------\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Note: Scores range 0-1. Overall Agent Score is a weighted average of Turn Success Ratio and Goal Completion Rate.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nConversation 1 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nConversation 2 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nConversation 3 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nConversation 4 (Turns: 2):\n   - Goal Completion Rate: 1.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.25\n   - Status: Failed\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nOVERALL PERFORMANCE ANALYSIS:\n-----------------------------\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Note: Scores range from 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance.\n\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 Helpfulness: 1.4 (Poor)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 Coherence: 1.2 (Poor)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 Verbosity: 4.0 (Excellent)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 Relevance: 1.2 (Poor)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 Faithfulness: 3.0 (Good)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nOverall Average: 2.2 (Needs Improvement)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nAGENT BEHAVIOR FAILURE BREAKDOWN:\n---------------------------------\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 disobey user request: 5 (62.5%)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 failure to ask for clarification: 2 (25.0%)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \u2022 repetition: 1 (12.5%)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Total evaluations: 8\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nTOP 5 UNIQUE ERRORS (by severity):\n----------------------------------\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n1. [HIGH] 2 occurrences (Conversation 1)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's statement without providing technology suggestions or addressing the user's clear request.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Scenarios: multi_step\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_1.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n2. [HIGH] 2 occurrences (Conversation 2)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's question instead of politely declining or redirecting the user to appropriate medical resources.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Scenarios: out_of_scope\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_2.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n3. [HIGH] 1 occurrences (Conversation 4)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's question without providing an answer or guidance on best practices for writing clean, maintainable code in Python.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Scenarios: happy_path\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_3.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n4. [MEDIUM] 2 occurrences (Conversation 3)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Unique Error: The user mentioned 'that thing about the money' without specifics, and the assistant did not ask for clarification.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Failure To Ask For Clarification\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Scenarios: ambiguous_intent\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_4.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n5. [LOW] 1 occurrences (Conversation 4)\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the same information from the previous turn without adding any new value or addressing the user's follow-up inquiry.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Repetition\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Scenarios: happy_path\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_5.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \nFOCUS FILES FOR TARGETED RERUNS:\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Rerun all failures: arksim simulate-evaluate <config> --scenario_file_path /private/tmp/arksim-smoke-test/results/focus/all_failures.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Or target a specific error: --scenario_file_path /private/tmp/arksim-smoke-test/results/focus/error_N.json\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO -   Tip: pass --output_dir to avoid overwriting these results.\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - \n============================================================\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Generating HTML report...\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - Successfully generated standalone HTML report!\n2026-04-09 10:37:38 - arksim.evaluator.evaluator - INFO - You can now open /private/tmp/arksim-smoke-test/results/final_report.html directly in your browser.\n2026-04-09 10:37:38 - arksim - INFO - Evaluation completed in 14.16 seconds\n2026-04-09 10:37:38 - arksim - INFO - Total elapsed: 17.50 seconds\n",
  "message": "Simulation and evaluation completed successfully."
}
```


### 3. User Journey 2: Exploring Results (`/arksim results`)

Simulating: User types `/arksim results` to see what happened.

**`list_results`** (PASS)

```json
// Input
{
  "output_dir": "/private/tmp/arksim-smoke-test"
}
```

```json
// Output
{
  "status": "success",
  "runs": [
    {
      "evaluation_id": "6698da90-b691-4f15-bf06-93ce7f0dbd33",
      "generated_at": "2026-04-09T17:37:38.233283+00:00",
      "file_path": "/private/tmp/arksim-smoke-test/results/evaluation.json",
      "total_conversations": 4,
      "passed": 0,
      "failed": 4,
      "unique_errors_count": 5
    }
  ],
  "skipped": []
}
```

Found 1 run(s), 0 skipped file(s).

**`read_result`** (PASS)

```json
// Input
{
  "result_path": "/private/tmp/arksim-smoke-test/results/evaluation.json"
}
```

```json
// Output
{
  "status": "success",
  "evaluation_id": "6698da90-b691-4f15-bf06-93ce7f0dbd33",
  "generated_at": "2026-04-09T17:37:38.233283+00:00",
  "total_conversations": 4,
  "passed": 0,
  "failed": 4,
  "unique_errors": [
    {
      "error_id": "bfc4ee70-f3c9-4005-9272-f55da1c124af",
      "category": "disobey user request",
      "description": "The assistant repeated the user's question without providing an answer or guidance on best practices for writing clean, maintainable code in Python.",
      "severity": "high",
      "occurrence_count": 1
    },
    {
      "error_id": "83d46399-afd6-45ed-888c-afb296b2b8e6",
      "category": "repetition",
      "description": "The assistant repeated the same information from the previous turn without adding any new value or addressing the user's follow-up inquiry.",
      "severity": "low",
      "occurrence_count": 1
    },
    {
      "error_id": "b6566eea-b4d3-44b1-96c1-5f318b4fae3b",
      "category": "disobey user request",
      "description": "The assistant repeated the user's statement without providing technology suggestions or addressing the user's clear request.",
      "severity": "high",
      "occurrence_count": 2
    },
    {
      "error_id": "b4d00aa8-dc63-4301-b424-228f825d25bd",
      "category": "failure to ask for clarification",
      "description": "The user mentioned 'that thing about the money' without specifics, and the assistant did not ask for clarification.",
      "severity": "medium",
      "occurrence_count": 2
    },
    {
      "error_id": "b370f86e-2e21-4231-b4ea-a1c57a292903",
      "category": "disobey user request",
      "description": "The assistant repeated the user's question instead of politely declining or redirecting the user to appropriate medical resources.",
      "severity": "high",
      "occurrence_count": 2
    }
  ],
  "conversations": [
    {
      "conversation_id": "57b9811d-149b-41df-88c7-4609de1564dd",
      "goal_completion_score": 0.0,
      "overall_agent_score": 0.0,
      "evaluation_status": "Failed",
      "turn_count": 2
    },
    {
      "conversation_id": "2792f1d1-b1d5-4d72-9091-382c29b1ca3c",
      "goal_completion_score": 0.0,
      "overall_agent_score": 0.0,
      "evaluation_status": "Failed",
      "turn_count": 2
    },
    {
      "conversation_id": "4dd7c451-5281-454d-986a-19a06aa0432a",
      "goal_completion_score": 0.0,
      "overall_agent_score": 0.0,
      "evaluation_status": "Failed",
      "turn_count": 2
    },
    {
      "conversation_id": "2cb373d8-98df-4209-990c-8b339dd4953a",
      "goal_completion_score": 1.0,
      "overall_agent_score": 0.25,
      "evaluation_status": "Failed",
      "turn_count": 2
    }
  ]
}
```


Results summary: 0/4 passed, 4 failed, 5 unique error(s)

Unique errors:
- [disobey user request] The assistant repeated the user's question without providing an answer or guidan
- [repetition] The assistant repeated the same information from the previous turn without addin
- [disobey user request] The assistant repeated the user's statement without providing technology suggest
- [failure to ask for clarification] The user mentioned 'that thing about the money' without specifics, and the assis
- [disobey user request] The assistant repeated the user's question instead of politely declining or redi

### 4. User Journey 3: Re-evaluate (`/arksim evaluate`)

Simulating: User types `/arksim evaluate` to re-run evaluation with different settings.

**`evaluate`** (PASS)

```json
// Input
{
  "config_path": "/private/tmp/arksim-smoke-test/config.yaml",
  "simulation_file_path": "/private/tmp/arksim-smoke-test/simulation.json",
  "cli_overrides": {
    "model": "gpt-4o-mini"
  }
}
```

```json
// Output
{
  "status": "success",
  "output": "",
  "stderr": "2026-04-09 10:37:39 - arksim - INFO - \nEvaluation configuration:\n2026-04-09 10:37:39 - arksim - INFO -   custom_metrics_file_paths: []\n2026-04-09 10:37:39 - arksim - INFO -   generate_html_report: True\n2026-04-09 10:37:39 - arksim - INFO -   metrics_to_run: ['faithfulness', 'helpfulness', 'coherence', 'verbosity', 'relevance', 'goal_completion', 'agent_behavior_failure']\n2026-04-09 10:37:39 - arksim - INFO -   model: gpt-4o-mini\n2026-04-09 10:37:39 - arksim - INFO -   num_workers: 50\n2026-04-09 10:37:39 - arksim - INFO -   numeric_thresholds: None\n2026-04-09 10:37:39 - arksim - INFO -   output_dir: /private/tmp/arksim-smoke-test/results\n2026-04-09 10:37:39 - arksim - INFO -   provider: openai\n2026-04-09 10:37:39 - arksim - INFO -   qualitative_failure_labels: None\n2026-04-09 10:37:39 - arksim - INFO -   scenario_file_path: /private/tmp/arksim-smoke-test/scenarios.json\n2026-04-09 10:37:39 - arksim - INFO -   score_threshold: None\n2026-04-09 10:37:39 - arksim - INFO -   simulation_file_path: /private/tmp/arksim-smoke-test/simulation.json\n2026-04-09 10:37:39 - arksim - INFO - \n2026-04-09 10:37:39 - arksim.evaluator.evaluator - INFO - Evaluator initialized: num_workers=50\n2026-04-09 10:37:39 - arksim.evaluator.evaluator - INFO - Starting evaluation of 4 conversations\n2026-04-09 10:37:39 - arksim.evaluator.evaluator - INFO - Preprocessing complete: 8 total turns to evaluate\n\nEvaluating...:   0%|          | 0/13 [00:00<?, ?step/s]\nEvaluating...:   8%|\u258a         | 1/13 [00:04<00:54,  4.51s/step]\nEvaluating...:  15%|\u2588\u258c        | 2/13 [00:04<00:22,  2.03s/step]\nEvaluating...:  23%|\u2588\u2588\u258e       | 3/13 [00:04<00:11,  1.16s/step]\nEvaluating...:  46%|\u2588\u2588\u2588\u2588\u258c     | 6/13 [00:05<00:03,  1.83step/s]\nEvaluating...:  54%|\u2588\u2588\u2588\u2588\u2588\u258d    | 7/13 [00:05<00:03,  2.00step/s]\nEvaluating...:  62%|\u2588\u2588\u2588\u2588\u2588\u2588\u258f   | 8/13 [00:06<00:02,  1.85step/s]\nEvaluating...:  69%|\u2588\u2588\u2588\u2588\u2588\u2588\u2589   | 9/13 [00:07<00:02,  1.37step/s]\nEvaluating...:  77%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258b  | 10/13 [00:08<00:01,  1.72step/s]\nEvaluating...:  92%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258f| 12/13 [00:08<00:00,  2.47step/s]\nDetecting agent errors:  92%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258f| 12/13 [00:08<00:00,  2.47step/s]2026-04-09 10:37:47 - arksim.evaluator.evaluator - INFO - Detecting agent errors\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Detected 6 unique errors\n\nDetecting agent errors: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:17<00:00,  2.38s/step]\nEvaluation complete: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:17<00:00,  2.38s/step]   \nEvaluation complete: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 13/13 [00:17<00:00,  1.32s/step]\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Post-processing complete\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Evaluation complete: 4 conversations, 8 turns, 6 unique errors\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - WARNING - Overwriting existing file: /private/tmp/arksim-smoke-test/results/evaluation.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Saving evaluation results to /private/tmp/arksim-smoke-test/results/evaluation.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Evaluation results saved successfully\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Generated 6 focus file(s) in /private/tmp/arksim-smoke-test/results/focus/\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Displaying evaluation summary\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n============================================================\nEVALUATION SUMMARY\n============================================================\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Conversations: 4\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Total Turns: 8\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Average Turns per Conversation: 2.0\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nTURN-BY-TURN EVALUATION:\n------------------------\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Note: Scores range from 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance.\n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Conversation 1:\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: failure to ask for clarification (The user's request is vague and could mean several things...), Relevance: 1.0 (The AI assistant's response does not address the user's query...), Helpfulness: 1.0 (The response from the AI assistant is unhelpful as it...), Coherence: 1.0 (The response from the AI assistant is completely incoherent and...), Faithfulness: 1.0 (The assistant's response is completely unhelpful and does not address...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: failure to ask for clarification (The assistant should have asked for clarification on the user's...), Relevance: 1.0 (The AI assistant's last message simply repeats the user's statement...), Verbosity: 1.0 (The response is excessively verbose as it simply repeats the...), Coherence: 1.0 (The AI's last message is a verbatim repetition of the...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Conversation 2:\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant restated the user's request instead of politely declining...), Coherence: 1.0 (The response from the AI assistant is completely incoherent and...), Helpfulness: 1.0 (The assistant did not provide any helpful information or answer...), Faithfulness: 1.0 (The assistant's response completely lacks any helpful information or engagement...), Relevance: 1.0 (The assistant's response does not address the user's question about...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: repetition (The assistant repeated the user's request instead of providing a...), Helpfulness: 1.0 (The assistant's response does not provide any helpful information or...), Relevance: 1.0 (The AI assistant's response does not address the user's question...), Coherence: 1.0 (The assistant's response is completely incoherent and does not address...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Conversation 3:\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant repeated the user's question without providing any useful...), Faithfulness: 1.0 (The assistant's response does not address the user's query and...), Coherence: 1.0 (The response is completely incoherent and fails to address the...), Relevance: 1.0 (The AI assistant's response does not address the user's query...), Helpfulness: 1.0 (The response is not useful or helpful at all as...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: repetition (The assistant repeated the user's response verbatim without adding any...), Helpfulness: 1.0 (The assistant repeated the user's message verbatim instead of providing...), Verbosity: 1.0 (The response from the AI assistant is unnecessarily verbose as...), Relevance: 1.0 (The assistant's response merely repeats the user's message without providing...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Conversation 4:\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 1: Issues: Agent Behavior Failure: disobey user request (The assistant repeated the user's request instead of providing relevant...), Helpfulness: 1.0 (The assistant's response is a complete repetition of the user's...), Coherence: 1.0 (The AI assistant's response is a verbatim repetition of the...), Faithfulness: 1.0 (The assistant's response is a direct repetition of the user's...), Relevance: 1.0 (The AI assistant's response does not address the user's query...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Turn 2: Issues: Agent Behavior Failure: lack of specific information (The assistant repeated the user's previous message verbatim without adding...), Verbosity: 1.0 (The response is unnecessarily lengthy and repeats the user's request...), Coherence: 1.0 (The response is completely incoherent as it merely repeats the...)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nCONVERSATION-LEVEL METRICS:\n---------------------------\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Note: Scores range 0-1. Overall Agent Score is a weighted average of Turn Success Ratio and Goal Completion Rate.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nConversation 1 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nConversation 2 (Turns: 2):\n   - Goal Completion Rate: 1.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.25\n   - Status: Failed\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nConversation 3 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nConversation 4 (Turns: 2):\n   - Goal Completion Rate: 0.00\n   - Turn Success Ratio: 0.00\n   - Overall Agent Score: 0.00\n   - Status: Failed\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nOVERALL PERFORMANCE ANALYSIS:\n-----------------------------\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Note: Scores range from 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance.\n\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 Helpfulness: 1.2 (Poor)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 Coherence: 1.1 (Poor)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 Verbosity: 3.5 (Good)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 Relevance: 1.5 (Poor)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 Faithfulness: 3.0 (Good)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nOverall Average: 2.1 (Needs Improvement)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nAGENT BEHAVIOR FAILURE BREAKDOWN:\n---------------------------------\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 disobey user request: 3 (37.5%)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 failure to ask for clarification: 2 (25.0%)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 repetition: 2 (25.0%)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \u2022 lack of specific information: 1 (12.5%)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Total evaluations: 8\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nTOP 5 UNIQUE ERRORS (by severity):\n----------------------------------\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n1. [HIGH] 1 occurrences (Conversation 2)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's request instead of politely declining to provide medical advice.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Scenarios: out_of_scope\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_1.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n2. [HIGH] 1 occurrences (Conversation 4)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's question instead of providing useful information about best practices in Python.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Scenarios: happy_path\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_2.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n3. [HIGH] 1 occurrences (Conversation 1)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant repeated the user's request instead of providing relevant technologies or recommendations for building a web application.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Disobey User Request\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Scenarios: multi_step\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_3.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n4. [MEDIUM] 2 occurrences (Conversation 3)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant failed to seek clarification on the user's vague request regarding money.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Failure To Ask For Clarification\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Scenarios: ambiguous_intent\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_4.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n5. [MEDIUM] 1 occurrences (Conversation 1)\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Unique Error: The assistant provided no new information or clarification regarding the recommended technologies.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Agent Behavior Failure Category: Lack Of Specific Information\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Scenarios: multi_step\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Focus file: /private/tmp/arksim-smoke-test/results/focus/error_5.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \nFOCUS FILES FOR TARGETED RERUNS:\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Rerun all failures: arksim simulate-evaluate <config> --scenario_file_path /private/tmp/arksim-smoke-test/results/focus/all_failures.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Or target a specific error: --scenario_file_path /private/tmp/arksim-smoke-test/results/focus/error_N.json\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO -   Tip: pass --output_dir to avoid overwriting these results.\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - \n============================================================\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Generating HTML report...\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - Successfully generated standalone HTML report!\n2026-04-09 10:37:56 - arksim.evaluator.evaluator - INFO - You can now open /private/tmp/arksim-smoke-test/results/final_report.html directly in your browser.\n2026-04-09 10:37:56 - arksim - INFO - Total elapsed: 17.33 seconds\n",
  "message": "Evaluation completed successfully."
}
```


### 5. Edge Cases and Error Handling

**read_result with missing file:**

**`read_result (missing file)`** (PASS)

```json
// Input
{
  "result_path": "/does/not/exist.json"
}
```

```json
// Output
{
  "status": "error",
  "error_message": "File not found: /does/not/exist.json"
}
```


**read_result with null conversations (crash guard):**

**`read_result (null conversations)`** (PASS)

```json
// Input
{
  "result_path": "/private/tmp/arksim-smoke-test/_test_null.json"
}
```

```json
// Output
{
  "status": "success",
  "evaluation_id": "null-test",
  "generated_at": "",
  "total_conversations": 0,
  "passed": 0,
  "failed": 0,
  "unique_errors": [],
  "conversations": []
}
```


**list_results with file path (not directory):**

**`list_results (file as dir)`** (PASS)

```json
// Input
{
  "output_dir": "/private/tmp/arksim-smoke-test/config.yaml"
}
```

```json
// Output
{
  "status": "success",
  "runs": [],
  "skipped": []
}
```


**simulate_evaluate with nonexistent config:**

**`simulate_evaluate (bad config)`** (PASS)

```json
// Input
{
  "config_path": "/nonexistent/config.yaml"
}
```

```json
// Output
{
  "status": "error",
  "error_message": "2026-04-09 10:37:57 - arksim - WARNING - No config YAML file provided.\n2026-04-09 10:37:57 - arksim - ERROR - Configuration error: 1 validation error for SimulationInput\n  Value error, Either inline agent_config or agent_config_file_path must be provided. [type=value_error, input_value={}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error\n"
}
```


### 6. Skills Validation

- **test.md**: 132 lines, references tools: simulate_evaluate, evaluate, init_project
- **evaluate.md**: 61 lines, references tools: evaluate
- **scenarios.md**: 120 lines, references tools: none
- **results.md**: 114 lines, references tools: list_results, read_result
- **ui.md**: 30 lines, references tools: launch_ui

**Cross-skill consistency checks:**
  PASS: No competitor mentions in any skill
  PASS: No em/en dashes in any skill
  PASS: Both test.md and scenarios.md include scenario JSON schema

### 7. Setup/Teardown Validation

settings.json MCP config:
```json
{
  "command": "/opt/homebrew/opt/python@3.11/bin/python3.11",
  "args": [
    "-m",
    "integrations.claude_code.mcp_server.server"
  ]
}
```
PASS: MCP server config is correct

### Summary

**19 passed, 0 failed**
