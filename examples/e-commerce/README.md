# Agent Evaluation Example - E-commerce

This directory gives an example of running Arksim with an example shopping assistant agent for the **e-commerce use case**. You can follow the example to evaluate your own agent.

> This example includes two types of agents:
>
> - **Option 1**: OpenAI agent that directly uses the OpenAI API to interact with the user simulator.
> - **Option 2**: Customized in-house agent exposed through a Chat Completions-compatible interface to interact with the user simulator.

## Option 1: OpenAI Agent

Steps to run:

1. Set your API keys as environment variables:

   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   ```

   The agent config uses `${OPENAI_API_KEY}` which will be automatically substituted at runtime.

   If using the Chat Completions agent (`agent_config_chat_completions.json`), also set:

   ```bash
   export AGENT_API_KEY="<YOUR_AGENT_API_KEY>"
   ```

2. The default `agent_config.json` is already configured for OpenAI. No changes needed.

3. Review `config.yaml` for this example.

4. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

## Option 2: In-house Agent

Steps to run:

1. Rename `agent_config_chat_completions.json` to `agent_config.json`. If `agent_config.json` already exists, rename or remove it first.

2. Start the agent:

   The user simulator expects agent responses in the Chat Completions format. You can either use the sample agent provided in the `./examples/e-commerce/chat_completions_server` folder, or integrate your own agent.

   **2.1 Run the Sample Agent**

   In the `./examples/e-commerce/chat_completions_server` folder, we provide a RAG-based agent implemented with LangGraph in `agent.py`. Follow the steps below to start the agent:
   - Create a virtual environment (Python 3.11 recommended) and install dependencies:
     1. Create and activate the environment:
        ```bash
        conda create -n ecommerce_venv python=3.11
        conda activate ecommerce_venv
        ```
     2. Navigate to the sample agent directory and install requirements:
        ```bash
        cd examples/e-commerce/chat_completions_server
        pip install -r requirements.txt
        ```

   - Set your OpenAI API key via the `OPENAI_API_KEY` environment variable:

     ```bash
     export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
     ```

   - Start the chat completion wrapper. This starts the agent on port 8080:

     ```bash
     python chat_completion_wrapper.py
     ```

   - Verify the servers are running:
     ```bash
     lsof -i -P -n | grep 8080
     ```
     You should see output similar to:
     ```bash
     python3.11  <process_id> <username>   10u  IPv4 0x6eaae5951a5c469b      0t0  TCP 127.0.0.1:8080 (LISTEN)
     ```

   **2.2 Your own agent**
   - If you have your own agent, wrap your agent in OPENAI chat completion request and response format. You can follow the `#TODO` comments in `chat_completion_wrapper.py` to use it.
   - Then, from the repository root, start the chat completion wrapper:
     ```bash
     python -m examples.e-commerce.chat_completions_server.chat_completion_wrapper
     ```

3. Review `config.yaml` for this example.

4. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```
