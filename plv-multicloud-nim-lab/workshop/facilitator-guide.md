# Facilitator guide

## Room and timing rules
- Start the notebook segments with everyone on the **mock**; switch the confident half to a live endpoint.
- Every lab ends with one sentence the attendee can repeat to their manager (write it on the whiteboard):
  1. "NIM is the OpenAI API with NVIDIA's engine underneath - our SDK code does not change."
  2. "The adapter cut prompt tokens ~5x with the same output contract."
  3. "Tool output is untrusted data; we measure injection resistance like we measure latency."
  4. "We know our cost per 1,000 summaries on each cloud, from measurements."

## Common failures (and the 30-second fix)
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: plover_api` | Run Jupyter from the repo root or `workshop/` (the first cell finds the root) |
| 404 for `plover-style-v1` | Adapter not loaded on that endpoint - expected on hosted endpoints; skip the cell |
| Live endpoint 401 | `PLV_API_KEY` missing; hosted keys never go in the notebook file |
| Streaming cell hangs | Corporate proxy buffering SSE - use the mock or a non-proxied network |
| Agent returns text instead of tool calls | Server tool-call parser is off (NIM: `--enable-auto-tool-choice --tool-call-parser ...` via `llm.extraArgs`) |

## Q&A bank (short answers)
- **"Why not just call a hosted API?"** For prototypes, do. For PHI at scale: data stays in your VPC, cost is flat per
  GPU-hour, and the same container runs in the hospital's data center (M9). Show the M10 scorecard.
- **"What does NVIDIA AI Enterprise add?"** Support, security patching (production branches), and the license for
  production NIM use: $4,500/GPU/year or $1/GPU-hour through cloud marketplaces (2026 list).
- **"Can we use Nemotron 3.5?"** Lightning has NIM profiles (BF16/W4A16/NVFP4, with LoRA). Run the agentic suite
  against it before switching - the add-on module shows how in 30 minutes.
- **"Which cloud is cheapest?"** The one where your SLO is met at the lowest $/1M tokens - `bench/compare_clouds.py`
  answers it with your traffic, not with list prices.
