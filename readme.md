This is a good MVP: you already have (1) workflow routing, (2) a Docker+pytest objective loop for coding, (3) strict RAG grounding checks, and (4) trace logging that’s ready to become training data.[1]

## README-ready project description

### Project title
**Hybrid Orchestrator: Self-improving Multi-Model / Multi-Agent AI System**

### One-liner
An orchestration layer that routes tasks to the best combination of models/tools, runs selective multi-agent debate, verifies outputs with objective checks (pytest/RAG grounding), and continuously learns from failures using preference-style training data.[1]

### What exists today (Beta)
- **Router → workflow dispatch**: JSON decision → `coding | ragqa | reasoning | science`. [1]
- **Coding workflow**: iterative “write main.py → run pytest in Docker → retry” loop (objective pass/fail).[1]
- **RAGQA workflow**: chunk selection + “answer with citations only from chosen chunks” + strict judge that rejects uncited/invalid claims.[1]
- **Reasoning workflow**: solver + optional critic (basic debate hook exists).[1]
- **Tracing**: structured events to `traces/*.json` (rounds, model outputs, chunk choices, pytest results).[1]

## Roadmap

### Phase 1 — Harden the MVP (1–2 weeks)
Goal: make runs repeatable, debuggable, and measurable.
- Add a unified **RunResult schema** across workflows (success, score, latency, token/cost estimate, artifacts).[1]
- Add “controls” defaults & validation (max rounds, debate on/off, budgets) to prevent router JSON from breaking runtime.[1]
- Improve reasoning/science judging from “structure-only” to at least **consistency checks** (e.g., require final + critic response; add “abstain if missing info” policy).[1]

Deliverable: stable CLI that can run `taskXXX` end-to-end and emit comparable trace JSON.

### Phase 2 — Selective multi-agent collaboration (2–4 weeks)
Goal: quality gains without huge latency.
- Implement **Top‑K selection**: router chooses 2–4 candidate workers rather than one.[1]
- Add **committee + merge** pattern:
  - multiple solvers answer in parallel
  - critic compares (or cross-critiques)
  - final “merger” produces the final answer
- Extend this beyond reasoning into:
  - **coding**: solver writes patch, verifier suggests fixes, optional “patch-critic”
  - **ragqa**: retriever proposes chunks, verifier audits chunk relevance, solver answers

Deliverable: measurable improvement vs single-solver baselines.

### Phase 3 — “Plan-first” / self-explanation routing (3–6 weeks)
Goal: route based on *how agents think*, not only on prompt features.
- Add a **plan-only step** (cheap): ask candidate agents for:
  - understanding of the task
  - intended approach
  - expected tools/tests
- Feed plans into the orchestrator to select top‑K and decide whether to debate, retrieve, verify, or escalate.

Deliverable: better routing decisions and fewer wasted calls.

### Phase 4 — Training data & self-improvement loop (6–10 weeks)
Goal: turn traces into learning signals.
- From existing judges you can already create preference pairs:
  - coding: passed trace > failed trace[1]
  - ragqa: fully grounded answer > answer with uncited/invalid claims[1]
- Add a dataset writer: `trace → (prompt, decision, outcome, preferred/nonpreferred)`.
- Start with lightweight learning:
  - train a small router classifier (or fine-tune a small LLM router) on “best workflow/worker” labels
  - then add preference-style fine-tuning for routing policies (cost/latency vs score)

Deliverable: router improves over time from your own task distribution.

### Phase 5 — Research-grade novelty features (10–16 weeks)
Goal: features suitable for a thesis/demo.
- **Ensemble of orchestrators**: multiple routers propose top‑K; vote/merge decisions; use disagreement as a “hardness” signal.
- **Multi-objective control**: reward/score combines correctness + cost + latency + grounding/trust.
- **Memory/RAG of past traces**: retrieve similar past tasks and reuse successful plans/decisions.
- **Trust & calibration**: per-agent reliability profiles learned from historical performance (per task type).

Deliverable: a full “self-improving orchestration stack” story with ablations.

## Updated feature list (current + planned)

### Current (implemented)
- Workflows: `coding`, `ragqa`, `reasoning`, `science`.[1]
- Router outputs structured JSON decision.[1]
- Coding: Docker sandbox + pytest verification + multi-round retries.[1]
- RAGQA: chunking + chunk selection + strict citation enforcement.[1]
- Tracing: structured event log for every run.[1]

### Next (near-term)
- Top‑K worker selection
- Committee/debate + merge
- Plan-first self-explanations (for routing)
- Better scoring for reasoning/science (beyond structure)
- Budget controls (token/cost/latency caps)

### Future (research/novelty)
- Ensemble orchestrators + disagreement routing
- Automatic preference dataset creation from judges
- Continuous training of router (and optionally weak workers) from winners vs losers
- Reliability/trust scoring per agent/model
- Trace-memory retrieval (“similar task succeeded with this plan”)

## Why this project is novel (for a prof/presentation)
- It treats orchestration as a **learnable controller** over models/tools, not just hard-coded routing.[1]
- It uses **objective/verifiable feedback** where possible (pytest; citation-grounded RAG) and logs everything as structured traces for future learning.[1]
- It aims for a full loop: **route → collaborate → verify → log → learn → improve**, which is closer to how production agent systems should evolve than static prompt pipelines.[1]

## Suggested README structure (copy/paste)

- **Overview**
- **Architecture**
  - Router
  - Workflows (coding/ragqa/reasoning/science)
  - Judge
  - Trace store
  - Sandbox runner
- **How to run**
  - create `runs/taskXXX/`
  - add `prompt.txt`, `tests/` or `context/`
  - run `python -m orchestrator.app taskXXX`
- **Trace format**
- **Roadmap**
- **Evaluation plan**
  - pass@k for coding
  - citation validity rate for RAG
  - latency/cost metrics
