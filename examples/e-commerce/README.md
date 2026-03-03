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

2. Review `config.yaml` for this example (the default configuration is sufficient to get started).

3. From this example directory, run:
   ```bash
   arksim simulate-evaluate path/to/examples/e-commerce/config_simulate.yaml
   ```

## Option 2: In-house Agent (`agent_server`)

In the `./examples/e-commerce/agent_server` folder, we provide a sample RAG-based agent implemented with OpenAI Agents SDK that can be exposed through a Chat Completions-compatible interface.

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

2. Start the sample agent (run all commands below from the **parent directory of the `examples` folder**):
   - **2.1 Chat Completions wrapper**

     This exposes an OpenAI Chat Completions-compatible endpoint on port `8888` at `/chat/completions`.

     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     export AGENT_API_KEY="<YOUR_AGENT_API_KEY>"
     python -m examples.e-commerce.agent_server.chat_completions.server
     ```

   - **2.2 Your own agent**

     If you have your own agent, wrap it in the OpenAI Chat Completions request and response format. You can follow the `#TODO` comments in `agent_server/chat_completions/server.py` to integrate it, then start the server:

     ```bash
     export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
     export AGENT_API_KEY="<YOUR_AGENT_API_KEY>"
     python -m examples.e-commerce.agent_server.chat_completions.server
     ```

3. From this example directory, run:
   ```bash
   export AGENT_API_KEY="<YOUR_AGENT_API_KEY>"
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   arksim simulate-evaluate config_chat_completions.yaml
   ```
