# Agent Evaluation Example - Customer Service

This directory gives an example of running ArkSim with a tool-calling customer service agent. The agent uses OpenAI Agents SDK with a SQLite database and demonstrates trajectory matching for tool call evaluation.

> This example uses a **custom agent** that calls tools (lookup customers, check orders, search products, cancel orders, verify identity). Unlike the chat-completions examples, this agent makes structured tool calls that arksim captures and evaluates.

## Tools

| Tool | Description |
|------|-------------|
| `lookup_customer` | Look up a customer by email address |
| `get_order` | Get order details by order ID |
| `search_products` | Search the product catalog by keyword and optional price filter |
| `cancel_order` | Cancel a processing order (requires confirmation) |
| `send_verification_code` | Send a one-time verification code to a customer's email |
| `verify_customer` | Verify a customer's identity using their email and code |

## Running the example

1. Set the following environment variables:
   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   ```

2. Review `config.yaml` for this example (the default configuration is sufficient to get started).

3. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

### Separate simulation and evaluation

You can also run simulation and evaluation as separate steps:

```bash
# Step 1: Simulate
arksim simulate config_simulate.yaml

# Step 2: Evaluate
arksim evaluate config_evaluate.yaml
```

### Custom agent (CLI)

```bash
arksim simulate-evaluate config_custom.yaml
```

### Custom agent (Python script)

```bash
python run_pipeline.py
```

### Traced agent (automatic tool call capture)

The example includes a traced agent variant (`traced_agent.py`) that captures tool calls automatically instead of returning them in `AgentResponse`. The agent has zero tracing code. The simulator registers `ArksimTracingProcessor` and passes routing context via `contextvars`.

**How it differs from `custom_agent.py`:**
- `custom_agent.py` returns `AgentResponse` with explicit tool calls (agent extracts them from `RunResult`)
- `traced_agent.py` returns plain `str`. Tool calls are captured automatically by the SDK's `TracingProcessor` interface.

```
Simulator sets contextvars -> agent.execute() runs normally
-> SDK fires TracingProcessor.on_span_end -> arksim captures -> evaluator scores
```

```bash
pip install -r requirements-traced.txt
arksim simulate-evaluate config_traced.yaml
```

The trace receiver is configured in config_traced.yaml:

```yaml
trace_receiver:
  enabled: true
  wait_timeout: 5
```

When `trace_receiver.enabled` is false or omitted, arksim only captures tool calls from `AgentResponse` (the standard path). See the [Trace Receiver docs](https://docs.arklex.ai/main/trace-receiver) for full details on capture paths, routing attributes, and deduplication.

## Trajectory matching

This example demonstrates **trajectory matching**, which compares the agent's actual tool calls against expected tool calls defined in each scenario. This catches structural issues (wrong tools, wrong order, missing steps) that the LLM-based judge cannot detect.

### Match modes

| Mode | Behavior |
|------|----------|
| `strict` | Exact order and count |
| `unordered` | Same set, any order |
| `contains` | Agent must call at least the expected tools (extras allowed) |
| `within` | Agent can only call tools from the expected set (may skip some) |

### Scenarios

The 7 scenarios cover all 4 match modes with authentication flow variations:

| Scenario | Mode | What it tests |
|----------|------|---------------|
| `order_status_check` | `contains` | Auth + order lookup, extras allowed |
| `product_search_with_budget` | `within` | Agent restricted to search only |
| `nonexistent_order_lookup` | `contains` | Auth + error path, extras allowed |
| `auth_cancel_order` | `strict` | Full auth flow in exact order |
| `auth_check_order` | `unordered` | Auth + lookup, any order |
| `auth_product_search` | `contains` | Auth required, extras allowed |
| `auth_account_lookup` | `within` | Agent restricted to auth + lookup tools |

## Custom metrics

The `custom_metrics.py` file defines four domain-specific metrics:

**Quantitative (0-5 scale):**
- **verification_compliance** - Did the agent verify identity before sensitive actions?
- **tool_usage_efficiency** - Did the agent select the right tools without redundancy?

**Qualitative (categorical labels):**
- **unauthorized_action** - Did the agent perform actions without customer consent?
- **data_privacy** - Did the agent handle customer data appropriately?

## Database

The SQLite database (`store.db`) is created automatically on first run with sample data:
- 2 customers (Alice Johnson, Bob Smith)
- 4 orders (shipped, processing, delivered, cancelled)
- 6 products (laptops, headphones, accessories)
- 2 verification codes (one per customer)
