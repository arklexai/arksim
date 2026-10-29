# Agent Evaluation Example - Customer Service

This directory gives an example of running ArkSim with a tool-calling customer service agent, demonstrates trajectory matching for tool call evaluation, and provides a medium-sized MCP dataset for developing grounded scenario generation. Two agent framework variants are included, sharing the same tool logic and database.

> This example uses a **custom agent** that calls tools for customers, orders, catalog search, identity verification, returns, product guidance, and searchable store knowledge. Unlike the chat-completions examples, this agent makes structured tool calls that arksim captures and evaluates.

## Variants

| Variant | Framework | Folder |
|---------|-----------|--------|
| OpenAI Agents SDK | `openai-agents` | `openai_agents/` |
| Google ADK | `google-adk` | `google_adk/` |

Both variants wrap the same shared core in `store_tools.py` - the tool logic and SQLite setup are written once, framework-agnostic, and each variant's `*_tools.py` / `tools.py` is a thin wrapper over it. This mirrors a common production setup: an agent built on Google ADK that exposes its own existing tools rather than a from-scratch reimplementation.

## Tools

| Tool | Description |
|------|-------------|
| `lookup_customer` | Look up a customer by email address |
| `get_order` | Get order details by order ID |
| `get_delivery_status` | Get fulfillment method, promised date, and delivery milestones for an order |
| `search_products` | Search the product catalog by keyword, with price, category, stock, and paging filters |
| `list_product_categories` | Inspect catalog facets, stock counts, and price ranges before searching |
| `cancel_order` | Cancel a processing order (requires confirmation) |
| `send_verification_code` | Send a one-time verification code to a customer's email |
| `verify_customer` | Verify a customer's identity using their email and code |
| `get_return_policy` | Get the return policy for a product category (returns prose, not JSON) |
| `get_product_guide` | Get the setup/care guide for a product (returns prose, not JSON) |
| `search_store_knowledge` | Search policies, buying guides, care guides, delivery, warranty, pricing, installation, privacy, and other support content. Returns excerpts |
| `get_knowledge_article` | Fetch a complete citable article by the ID returned from knowledge search |
| `get_document` | Fetch a complete document by its `store://` source URI, for knowledge articles and catalog entries alike |

`get_return_policy` and `get_product_guide` are compatibility helpers that return prose directly. The search-then-read pair is `search_store_knowledge` plus `get_knowledge_article` or `get_document`: search returns a 240-character excerpt and the full article averages about 1,100 characters, so a retrieval loop that grounds only on search results gets a fraction of each policy and has to make the second call to see the deciding detail.

`get_delivery_status` is the live-data tool of the set, standing in for a fulfillment lookup.

## MCP grounding development fixture

The dataset is deliberately large enough that a model cannot rely on the handful of rows named in `scenarios.json`. It contains generated-but-deterministic distractors, overlapping terminology, stock and price variation, category-specific policy differences, and service articles that require semantic tool selection. Knowledge search runs on SQLite FTS5, so queries match on tokens rather than substrings and word order does not matter.

`mcp_grounding_cases.json` is a golden development matrix of eighteen scenario-generation requests with expected tool choices, stable evidence IDs, and facts that must be present. Each case declares a `grounding_shape` naming the retrieval behaviour it exercises: `search_then_read`, `structured_lookup`, `cross_tool`, `pagination`, `ambiguous`, and `no_evidence` (a request the corpus genuinely cannot answer, where grounding nothing is the correct outcome). Unlike `scenarios.json`, these cases do not prescribe the final simulated-user persona or tool trajectory.

### Protocol shapes exposed to the connector

Real MCP servers do not answer in one shape, so `google_adk/mcp_shapes.py` spreads the tools across the shapes the spec allows: `structuredContent` with a text fallback, one text block per passage, a single prose block, a non-text (image) block, a protocol-level `isError`, and an in-band error payload on an otherwise successful call. The last two are distinct on purpose - an empty search found nothing but succeeded, while a lookup of a document that does not exist failed.

Three tools are exposed over MCP but withheld from the agent, purely as connector targets: `export_catalog_snapshot` (about 70 KB, for size caps), `render_room_preview` (image output, the kind of tool a connector usually excludes), and `simulate_tool_failure` (fails on demand, in either error shape).

### Running it

Start the server and check it without an LLM. `smoke_client.py` asserts and exits non-zero on failure, so it works as a CI protocol check:

```bash
python -m google_adk.mcp_server
python google_adk/smoke_client.py
```

The local Streamable HTTP endpoint is `http://127.0.0.1:9997/mcp`. The platform connector intentionally rejects loopback and private-network endpoints, so expose the server through an HTTPS development tunnel or deploy it somewhere reachable before registering it:

```bash
MCP_HOST=0.0.0.0 MCP_PORT=9997 MCP_ALLOWED_HOSTS=your-public-host.example python -m google_adk.mcp_server
MCP_URL=https://your-public-host.example/mcp python google_adk/smoke_client.py
```

Credentials and failures are switched on by environment variable, so the connector's auth, health-check, timeout, and per-source failure paths have a target without a second server:

| Variable | Effect |
|----------|--------|
| `MCP_AUTH_TOKEN=secret` | Require `Authorization: Bearer secret`; 401 otherwise |
| `MCP_AUTH_HEADER='X-Api-Key: secret'` | Require a custom header; 401 otherwise |
| `MCP_FAULT=down` | Every request returns 503, as an unreachable source |
| `MCP_FAULT=slow:5000` | Delay every request by 5s, for per-call timeouts |
| `MCP_FAULT=fail_every:3` | Every third request returns 500, for retry behaviour |
| `MCP_FAULT_TOOLS=search_store_knowledge` | Named tools always return `isError` while the rest keep working |

Register the endpoint in the platform's MCP configuration, enable the read-only tools, and select it as a scenario-generation source. Useful development prompts include:

- Generate scenarios about matching a competitor's price, including exclusions.
- Generate scenarios for returning opened power tools and custom-tinted paint.
- Generate scenarios involving appliance delivery preparation and hidden damage.
- Generate scenarios for choosing electrical products where compatibility and safety matter.

### Using the corpus as a second source

Phase 1 grounds generation on the uploaded corpus and Phase 2 adds the MCP source, so testing both together needs them pointed at the same domain. `export_corpus.py` writes the knowledge articles out as markdown with front matter, and `--subset` produces disjoint halves so overlap between the two sources is controlled:

```bash
python export_corpus.py                      # all 89 articles
python export_corpus.py --subset service     # 29 store-wide service articles
python export_corpus.py --subset categories  # 60 per-category articles
```

Upload one subset as ordinary documents and serve the other over MCP to exercise multi-source scoping; export the same set to both to exercise dedupe by content hash.

### Checking the fixture

```bash
python store_tools.py           # print dataset counts and facets
python -m unittest test_store_tools    # dataset and tool checks, no dependencies
python -m unittest test_mcp_server     # MCP protocol shapes, in-memory transport
```

`test_mcp_server.py` covers the same ground as the smoke client without binding a port, and skips itself when the ADK variant's dependencies are not installed.

`store.db` is disposable. `init_db()` automatically rebuilds it when the dataset version changes, while preserving the canonical IDs and values used by existing trajectory assertions.

## Run matrix

| | Direct (Python class) | A2A server |
|---|---|---|
| **OpenAI Agents SDK** | `arksim simulate-evaluate config_custom.yaml` | `AGENT_BACKEND=openai python -m a2a_server.server` + `arksim simulate-evaluate config_a2a.yaml` |
| **Google ADK** | `arksim simulate-evaluate config_adk.yaml` | `AGENT_BACKEND=adk python -m a2a_server.server` + `arksim simulate-evaluate config_a2a.yaml` |

Plus, Google ADK-only:

| | Command |
|---|---|
| **MCP server** (exposes tools, not the full agent) | `python -m google_adk.mcp_server` |
| **MCP smoke test** | `python google_adk/smoke_client.py` |

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

### Quick manual testing (scratch scripts)

`try_it.py` (OpenAI variant) and `google_adk/try_it.py` (Google ADK variant) are lightweight, non-official scratch scripts for manually poking at an agent during development. Each asks the agent a single question and prints the answer plus which tools got called - no scenario file, no evaluation, no HTML report, just a fast sanity check that a tool wiring change works before running the full `arksim simulate-evaluate` pipeline.

```bash
python try_it.py "What is the return policy for laptops?"                    # OpenAI variant
python google_adk/try_it.py "What is the status of order ORD-1001?"          # Google ADK variant
```

These are not part of the example's official surface (no `arksim` command touches them) - safe to ignore if you only care about the simulate/evaluate flows above.

### Traced agent (automatic tool call capture)

The example includes a traced agent variant (`openai_agents/traced_agent.py`) that captures tool calls automatically instead of returning them in `AgentResponse`. The agent registers `ArksimTracingProcessor` once at module load. The simulator passes routing context automatically.

**How it differs from `custom_agent.py`:**
- `custom_agent.py` returns `AgentResponse` with explicit tool calls (agent extracts them from `RunResult`)
- `traced_agent.py` returns plain `str`. Tool calls are captured automatically by the SDK's `TracingProcessor` interface.

```
Simulator sets routing context -> agent.execute() runs normally
-> SDK fires TracingProcessor.on_span_end -> arksim captures -> evaluator scores
```

```bash
pip install -r openai_agents/requirements-traced.txt
arksim simulate-evaluate config_traced.yaml
```

The trace receiver is configured in config_traced.yaml:

```yaml
trace_receiver:
  enabled: true
  wait_timeout: 5
```

When `trace_receiver.enabled` is false or omitted, arksim only captures tool calls from `AgentResponse` (the standard path). See the [Tool Call Capture docs](https://docs.arklex.ai/main/tool-call-capture) for full details.

### Google ADK variant

The example also includes a Google ADK variant (`google_adk/`), built the way many production agents are (ADK `FunctionTool`s behind a conversational agent). It reuses the same `store_tools.py` core as the OpenAI variant - only the framework-specific wrapping differs.

```bash
pip install google-adk
export GOOGLE_API_KEY="<YOUR_GOOGLE_API_KEY>"
arksim simulate-evaluate config_adk.yaml
```

`google_adk/agent.py`'s `run_and_capture_tool_calls()` pairs ADK's per-event function calls with their responses by id (ADK emits these as separate stream events, unlike the OpenAI Agents SDK's single `RunResult`), producing the same `ToolCall` shape the OpenAI variant does.

**MCP server.** `google_adk/mcp_server.py` exposes the tools over MCP (streamable HTTP, port 9997), using the low-level `mcp.server.Server` directly - not the decorator-per-tool `MCPServer` - to demonstrate wrapping an existing `FunctionTool` set the way a team with existing tools would, rather than redeclaring them in a new framework's decorator style. Requires `mcp>=2.0.0`. Each tool carries MCP annotations (`readOnlyHint` / `destructiveHint`) from `google_adk/adk_tools.py`'s `TOOL_ANNOTATIONS` map; `send_verification_code` deliberately carries none, to exercise the fail-safe default (absent hint treated as unsafe, not safe) in the platform's MCP connector.

```bash
python -m google_adk.mcp_server       # terminal 1
python google_adk/smoke_client.py     # terminal 2 - lists tools, calls two of them
```

**A2A server, either backend.** `a2a_server/agent.py` picks the backend from `AGENT_BACKEND` (default `openai`):

```bash
AGENT_BACKEND=adk python -m a2a_server.server
```

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

The 7 scenarios cover authentication flow variations across three match modes:

| Scenario | Mode | What it tests |
|----------|------|---------------|
| `order_status_check` | `contains` | Auth + order lookup, extras allowed |
| `product_search_with_budget` | `within` | Agent restricted to the catalog and knowledge tools |
| `nonexistent_order_lookup` | `contains` | Auth + error path, extras allowed |
| `auth_cancel_order` | `strict` | Full auth flow in exact order |
| `auth_check_order` | `contains` | Auth + lookup, extras allowed |
| `auth_product_search` | `contains` | Auth required, extras allowed |
| `auth_account_lookup` | `within` | Agent restricted to auth + lookup tools |

`unordered` is not used here. It rejects any call outside the expected set while adding nothing over `contains` except that rejection, so a scenario wanting "these tools, order irrelevant" says `contains` and one wanting a closed set says `within`.

## Custom metrics

The `custom_metrics.py` file defines four domain-specific metrics:

**Quantitative (0-5 scale):**
- **verification_compliance** - Did the agent verify identity before sensitive actions?
- **tool_usage_efficiency** - Did the agent select the right tools without redundancy?

**Qualitative (categorical labels):**
- **unauthorized_action** - Did the agent perform actions without customer consent?
- **data_privacy** - Did the agent handle customer data appropriately?

## Database

The SQLite database (`store.db`) is created automatically with:

- 108 customers across standard, premium, and Pro tiers
- 500 orders spanning delivered, shipped, processing, pending, and cancelled states, each with an order date, a promised date, and one of four fulfillment methods (parcel, pickup, appointment, freight)
- 225 products across 25 categories, with descriptions, stock variation, and useful price distributions
- 89 citable knowledge articles across 19 topics, including returns, buying guides, product care, shipping, delivery, pricing, warranty, installation, accounts, payments, privacy, rentals, order tracking, store pickup, special orders, safety notices, and support. Each is a multi-section document averaging about 1,100 characters
- Verification codes for every customer

Dates are generated from a fixed anchor (2 March 2026), so every run produces the same data. An order's dates follow from its status rather than being drawn independently, so a `pending` order is always still inside its lead time and a `delivered` one is always past its promised date - an agent asked "where is my order" can answer from either field without contradicting itself.

`ORD-1001`-`ORD-1004`, `P-001`-`P-006`, and `C-001`/`C-002` are the original seed rows `scenarios.json`'s trajectory assertions and knowledge hints reference by exact ID and value. The deterministic generated rows provide corpus scale and distractors without changing those assertions.

`scenarios.json` is scenario schema v2: each scenario carries `behavior` (the simulated user's starting patience) and a `goal_checklist` whose items reference knowledge chunks by id. Without a checklist a conversation has no mandatory sub-goal to satisfy, so it can only end in `gave_up` or `max_turns` - the seven scenarios here can now reach `user_satisfied`.

Tool-call assertions use the mode that matches what each scenario is actually testing. `contains` where the point is that certain tools were reached and extra calls are the agent's business; `within` where the point is that the agent stayed inside a set, listing every tool it may legitimately reach; `strict` only for `auth_cancel_order`, whose subject is the exact verify-then-cancel order, and whose agent context therefore names the four tools and rules the rest out. Adding a tool to the agent means revisiting the `within` and `strict` scenarios, since an unexpected call fails both.
