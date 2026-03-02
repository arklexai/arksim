# Agent Evaluation Example - Insurance Company

This directory gives an example of running Arksim with an example customer service agent for the **insurance company use case**. You can follow the example to evaluate your own agent.

> This example includes two types of agents:
>
> - **Option 1**: OpenAI agent that directly uses the OpenAI API to interact with the user simulator.
> - **Option 2**: Customized in-house agent exposed through A2A Protocol or Chat Completions-compatible interface to interact with the user simulator.

## Option 1: OpenAI Agent

Steps to run:

1. Set the following environment variables:
   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   ```

   If using the Chat Completions agent (`agent_config_chat_completions.json`), also set:

   ```bash
   export AGENT_API_KEY="<YOUR_AGENT_API_KEY>"
   ```

2. The default `agent_config.json` is already configured for OpenAI. No changes needed.
3. Review `config.yaml` for this example (the default configuration is sufficient to get started).
4. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

## Option 2: In-house Agent (`agent_server`)

In the `./examples/bank-insurance/agent_server` folder, we provide a sample RAG-based agent implemented with LangGraph that can be exposed with A2A Protocol or through Chat Completions interface.

Steps to run:

1. Choose an agent configuration and rename it to `agent_config.json` (if `agent_config.json` already exists, rename or remove it first).
   - For **A2A (recommended)**: `agent_config_a2a.json` → `agent_config.json`
   - For **Chat Completions**: `agent_config_chat_completions.json` → `agent_config.json`

   The agent configs support environment variable substitution using `${ENV_VAR_NAME}` syntax (for example, `${A2A_CLIENT_CREDENTIAL}` or `${OPENAI_API_KEY}`).

2. Install the sample agent dependencies:
   - Create a virtual environment (Python 3.11 recommended) and install dependencies:
     1. Create and activate the environment:
        ```bash
        conda create -n bank_venv python=3.11
        conda activate bank_venv
        ```
     2. Navigate to the sample agent directory and install requirements:
        ```bash
        cd examples/bank-insurance/agent_server
        pip install -r requirements.txt
        ```

3. Start one of the sample agents:
   - **3.1 A2A agent server**

     This exposes an A2A-compatible agent on port `9999`.

     ```bash
     export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
     cd ../../../
     python -m examples.bank-insurance.agent_server.a2a.server
     ```

   - **3.2 Chat Completions wrapper**

     This exposes an OpenAI Chat Completions-compatible endpoint on port `8888` at `/chat/completions`.

     ```bash
     export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
     cd ../../../
     python -m examples.bank-insurance.agent_server.chat_completions.server
     ```

     For the chat completions server, if `enable_metadata` is set to `true` in the request body, the following metadata will be sent together with the user message:
     - `chat_id`: The chat ID of the conversation.
     - `user_attributes`: The user attributes of the conversation.
     - `user_goal`: The user goal of the conversation.
     - `knowledge`: The knowledge of the conversation.

   You can also adapt these servers to call your own backend agent by following the comments in `agent_server/chat_completions/server.py` (for Chat Completions) or by implementing your own A2A-compatible executor in `agent_server/a2a`.

4. Review `config.yaml` for this example.

5. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```
