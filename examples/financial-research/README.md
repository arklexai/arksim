# Financial Simulation Module Proposal for ArkSim


# 4/17 Financial Research Example for ArkSim

## Overview
This example adds a filing-grounded financial research agent on top of ArkSim.

The current version is intentionally simplified to use only fixed local sources, so the simulation is more stable and reproducible.

## Current scope
This example currently uses:
- local company filing PDFs
- offline document ingestion
- a local retrieval index built from those filings
- ArkSim simulation and evaluation

It does not currently depend on live web search or external financial data APIs.

## Why this version is simplified
Earlier iterations included external tools such as web/news retrieval and structured financial data adapters.
Those sources made behavior less stable across runs and introduced avoidable noise during evaluation.

To better match ArkSim’s simulation/evaluation workflow, this version keeps only fixed filing-based inputs.

## Current workflow
1. Put filing PDFs into `data/reports/`
2. Run offline ingestion to build the local index
3. Run ArkSim simulation/evaluation against the filing-grounded agent

## What this example is trying to show
- multi-turn filing-grounded financial Q&A
- scenario-based financial research simulation
- ArkSim-based evaluation of domain-specific agent behavior
- a more reproducible setup using fixed local evidence

## Current limitations
This is still a prototype example.

Known limitations:
- some turn-level answers are still too generic or incomplete
- some table-heavy questions need more deterministic extraction/calculation logic
- retrieval and synthesis can still be improved for detailed filing analysis

## Future improvement directions
Possible next steps:
- improve deterministic handling of filing tables and derived metrics
- add better section-aware retrieval
- optionally reintroduce structured financial data or earnings materials as fixed local sources
- improve domain-specific evaluation metrics

## Setup
1. Put report PDFs into `data/reports/`
2. Build the local index:
   `python scripts/ingest_docs.py`
3. Run ArkSim:
   `arksim simulate-evaluate config.yaml`

## Summary
This example does not modify ArkSim core.

It is a lightweight, filing-grounded financial research example that uses ArkSim as the simulation and evaluation backbone, while intentionally keeping the data sources fixed and local.





## What this does
This module adds a financial research simulation agent on top of ArkSim.

Inputs:
- company filing PDFs / text
- structured financial dataset
- Tavily web search

Outputs:
- role-conditioned market view
- bull / bear / risks / conclusion
- source-backed simulated answer

## Purpose

I plan to contribute to ArkSim longer term by adding a lightweight financial simulation module on top of the existing framework, without modifying ArkSim core.

The idea is to use ArkSim's existing:

* scenario definition
* simulation engine
* evaluation pipeline

and extend it with a domain-specific financial research agent that can use:

* user role / investor persona
* financial report PDFs
* structured financial datasets
* web/news retrieval (e.g. Tavily)

This would allow us to simulate analyst / PM-style conversations and evaluate how well an agent supports market analysis, thesis formation, and investment-style reasoning.

---

## High-level goal

Build a new example module under `examples/financial/` that enables:

* role-conditioned financial conversations
* PDF-grounded financial research
* optional financial dataset integration
* optional web/news retrieval
* ArkSim-based simulation + evaluation of multi-turn financial analysis behavior

This is intended as an **additive module**, not a core architecture change.

---

## Proposed scope

### In scope

* Add a new `examples/financial-research/` example
* Implement a custom `BaseAgent` financial agent
* Ingest company filings / reports into a local RAG index
* Support scenario-based personas (e.g. hedge fund analyst, long-only PM)
* Evaluate conversation quality using ArkSim's standard simulation/evaluation flow

### Out of scope for initial version

* Trading execution
* Personalized investment advice
* Modifying ArkSim core simulation/evaluation architecture
* Production-grade market data platform integration in v1

---

## Why this is useful

ArkSim is already good at:

* simulating multi-turn user behavior
* evaluating agent behavior and failures
* comparing scenarios systematically

A financial module would make ArkSim useful for a new class of evaluation problems:

* thesis-building conversations
* earnings / filings interpretation
* role-specific financial reasoning
* source-grounded investment research assistance

This is a good fit for ArkSim because the framework already handles the hard part of **simulation + evaluation**; the new work is mainly in **domain-specific agent behavior + data grounding**.

---

## Current architecture idea

```text
ArkSim Scenarios
    ↓
Simulated User
    ↓
Financial Custom Agent
    ├─ PDF RAG
    ├─ Financial Dataset Adapter
    └─ Web / News Retrieval (Tavily)
    ↓
Agent Response
    ↓
ArkSim Evaluation
    ↓
Simulation + Evaluation Reports
```

---
<!-- 
## Planned folder structure

```text
examples/financial/
  config.yaml
  scenarios.json
  README.md

  agent/
    financial_agent.py
    prompts.py
    schemas.py
    tools/
      rag_tool.py
      financial_data_tool.py
      tavily_tool.py

  data/
    reports/
    index.jsonl

  scripts/
    ingest_docs.py

  metrics/
    custom_metrics.py
```

---

## Current status

Prototype work is already in progress:

* custom financial agent scaffolded
* local PDF ingestion / retrieval tested
* ArkSim simulation + evaluation path already exercised end-to-end
* early results show the integration works, but agent behavior still needs refinement

Main issues found so far:

* responses are too template-driven
* grounding to PDFs/sources needs to be stricter
* web retrieval currently adds noise if local evidence is weak
* structured financial dataset integration is still shallow

---

## Next steps

1. Improve PDF grounding and source handling
2. Make response behavior query-aware instead of always returning a full memo
3. Temporarily reduce/noise-control Tavily usage during debugging
4. Strengthen scenario knowledge and evidence design
5. Add custom financial evaluation metrics once behavior is more stable

---

## Collaboration intent

I would like to contribute to this repo on a longer-term basis through this financial module direction.

The near-term goal is to get internal feedback on:

* whether this is a useful extension area for ArkSim
* whether the proposed module boundary is appropriate
* which parts should remain example-level vs. framework-level
* what evaluation directions would be most valuable internally

---

## Simple summary

This proposal does **not** change ArkSim core.
It adds a new domain-specific example that uses ArkSim as the simulation/evaluation backbone for financial research conversations.

In short:
**ArkSim provides the evaluation framework; the financial module provides the domain-specific agent and data grounding.**

## Setup
1. Copy `.env.example` to `.env`
2. Put reports into `data/reports/`
3. Build index:
   `python scripts/ingest_docs.py`
4. Test one scenario:
   `python scripts/run_single.py`
5. Run via ArkSim:
   `arksim simulate-evaluate config.yaml` -->
