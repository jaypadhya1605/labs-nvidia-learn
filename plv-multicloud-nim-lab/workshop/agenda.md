# Workshop: "Ship one NIM product on any cloud" (2 hours, hands-on)

**For:** Plover Health AI engineers (platform, ML, app). **Delivered by:** NVIDIA SA (+ CSP partner SA for the cloud part).
**Outcome:** every attendee has called NIM, compared a LoRA adapter to a long prompt, broken and fixed an agent, and
can explain the cost of a summary on two clouds. Synthetic data only.

| Time | Segment | Format | Material |
|---|---|---|---|
| 0:00 | Why NIM, in one picture: same container, any GPU, any cloud | talk (10) | slides 1-4 |
| 0:10 | Physics in 5 minutes: decode = bytes/bandwidth, prefill = FLOPs (🔁 Lab 02) | talk + selector demo (10) | `selector/instance_selector.py --spot` |
| 0:20 | Lab 1 - the OpenAI API, streaming, TTFT | notebook §1-2 (15) | `plover-workshop.ipynb` |
| 0:35 | Lab 2 - long prompt vs LoRA adapter; structured output | notebook §3-4 (15) | |
| 0:50 | Break | | |
| 1:00 | Lab 3 - agents: tool calls + prompt injection (🔁 Lab 03) | notebook §5 (15) | `tests/agentic/harness.py` |
| 1:15 | Demo - same chart, two clouds: Argo CD, NIM Operator, canary by commit | live demo (15) | AKS + GKE, `mlops/promote.sh` |
| 1:30 | Lab 4 - measure and price: concurrency probe, $/1k summaries | notebook §6-7 (15) | `bench/compare_clouds.py` output |
| 1:45 | What's new: Nemotron 3.5 (released today) - what it means for Plover | talk (10) | `addon-nemotron35/workshop-segment.md` |
| 1:55 | Next steps + survey | (5) | mutual action plan |

## Before the day (facilitator)
- Run the notebook against the mock (`python3 workshop/build_notebook.py`, execute) and against AKS (`PLV_ENDPOINT`).
- Decide the attendee endpoint: **mock** (zero setup), **build.nvidia.com** trial key (prototyping terms, synthetic
  data), or a **shared lab endpoint** (Plover API behind an internal load balancer + key). Never PHI.
- Pre-record the GitOps demo (2 min video) in case the conference Wi-Fi blocks the clusters.
- Print the drill table (drills/README.md) for the hallway track.
