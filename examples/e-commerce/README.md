# Agent Evaluation Example - E-commerce

This directory gives an example of running ArkSim with an example shopping assistant agent for the **e-commerce use case**. You can follow the example to evaluate your own agent.

> This example includes two types of agents:
>
> - **Option 1**: OpenAI agent that directly uses the OpenAI API to interact with the user simulator.
> - **Option 2**: Customized in-house agent exposed through A2A Protocol, Chat Completions-compatible interface, or loaded directly as a Python class.

## Option 1: OpenAI Agent

Steps to run:

1. Set the following environment variables:
   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   ```

2. Review `config.yaml` for this example (the default configuration is sufficient to get started).

3. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

## Option 2: In-house Agent (`agent_server`)

In the `./examples/e-commerce/agent_server` folder, we provide a sample RAG-based agent implemented with OpenAI Agents SDK that can be exposed with A2A Protocol or through Chat Completions interface.

Steps to run:

1. Install the sample agent dependencies:
   - Create a virtual environment (Python 3.11 recommended) and install dependencies:
     1. Create and activate the environment:
        ```bash
        conda create -n ecommerce_venv python=3.11
        conda activate ecommerce_venv
        ```
     2. Navigate to the sample agent directory and install requirements:
        ```bash
        cd examples/e-commerce/agent_server
        pip install -r requirements.txt
        ```

2. Start one of the sample agents (run all commands below from the **parent directory of the `examples` folder**):
   - **2.1 A2A agent server**

     This exposes an A2A-compatible agent on port `9999`.

     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     export A2A_API_KEY=1234-4567-8910
     python -m examples.e-commerce.agent_server.a2a.server
     ```

   - **2.2 Chat Completions wrapper**

     This exposes an OpenAI Chat Completions-compatible endpoint on port `8888` at `/chat/completions`.

     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     export AGENT_API_KEY=123456
     python -m examples.e-commerce.agent_server.chat_completions.server
     ```

   - **2.3 Custom agent connector**

     This loads the agent directly as a Python class — no HTTP server needed. See [`custom_agent.py`](custom_agent.py) for the `BaseAgent` subclass implementation.

   You can also adapt these servers to call your own backend agent by following the comments in `agent_server/chat_completions/server.py` (for Chat Completions) or by implementing your own A2A-compatible executor in `agent_server/a2a`.

3. From this example directory, run with the appropriate config:
   - **A2A agent**:
     ```bash
     export A2A_API_KEY=1234-4567-8910
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     arksim simulate-evaluate config_a2a.yaml
     ```
   - **Chat Completions agent**:
     ```bash
     export AGENT_API_KEY=123456
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     arksim simulate-evaluate config_chat_completions.yaml
     ```
   - **Custom agent (CLI)**:
     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     arksim simulate-evaluate config_custom.yaml
     ```
   - **Custom agent (Python script)**:
     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     python run_pipeline.py
     ```
