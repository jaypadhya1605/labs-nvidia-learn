# Workshop: "Storm day: an AI agent, measured" (90 minutes, hands-on)

**For:** Tern Travel Group engineers: platform, ML and contact-center app developers (fictional company).
**Delivered by:** the NVIDIA solutions architect, with Tern's platform lead as co-host for the Q&A.
**Outcome:** every attendee has run the Disruption Concierge, counted its LLM calls, measured and priced its traffic,
broken its prefix cache with one line, and diagnosed a fault from symptoms. Laptops only; the GPU part is a demo.
Synthetic data only.

| Time | Segment | Format | Material |
|---|---|---|---|
| 0:00 | Why an agent is an inference problem: 40,000 travelers, 6.5 LLM calls each, ~96% of the prompt repeated | talk (10) | slides 1-5 |
| 0:10 | Exercise 1 - one disruption, step by step: predict the tool calls, run, compare with the policy oracle | notebook §1 (15) | [exercises.md](exercises.md) |
| 0:25 | How engines serve agents: batching, prefix caching, speculation, FP8; vLLM, SGLang, TensorRT-LLM, NIM | talk (10) | slides 6-10 |
| 0:35 | Exercise 2 - where the tokens go; measure and price it; break it with one timestamp | notebook §2-4 (15) | |
| 0:50 | Break | (5) | |
| 0:55 | Exercise 3 - mystery fault in pairs: diagnose from symptoms, then read the drill | notebook §5 (15) | [drills/README.md](../drills/README.md) |
| 1:10 | Demo - the same benchmark on an H100: four engines and the cost waterfall | live or recorded (10) | notebook §6, [docs/09](../docs/09-results-and-exec-readout.md) |
| 1:20 | Next steps: what the PoC will prove, survey | (10) | slides 14-15, [sales/poc-charter.md](../sales/poc-charter.md) |

## Before the day (facilitator)
- Build and run the notebook on the mock: `python workshop/build_notebook.py`, then execute it (about 2-3 minutes).
- Run it once on the GPU VM too (`TTG_BASE_URL=http://localhost:8000/v1` with vLLM up) and keep the ledger for §6.
- Record the GPU demo (3 minutes) in case the venue network blocks the VM.
- Decide the attendee setup: **mock on laptops** (default, zero cost), or a shared lab endpoint behind a key.
  Hosted endpoints (build.nvidia.com) get synthetic data only.
- Print the drill table from [drills/README.md](../drills/README.md) for the pairs exercise.
- Full checklist and timing cues: [facilitator-guide.md](facilitator-guide.md).

## Attendee prerequisites (send 3 days before)
- Python 3.11+, git, 4 GB free RAM. Windows: use WSL2.
- `git clone <repo> && cd ttg-agentic-genai-lab && pip install -r requirements-dev.txt`
- Smoke test: `uvicorn ttg.mock.fake_llm:app --port 8000` in one terminal, then
  `TTG_BASE_URL=http://localhost:8000/v1 TTG_MODEL=ttg-llm python -m ttg.agent.run --scenario S0005` in another.
  It should end with `1/1 policy-compliant`.
