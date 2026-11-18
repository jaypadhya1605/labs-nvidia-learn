# Slide outline (15 slides, talk track in italics)

1. **Plover x NVIDIA: one product, any cloud** - *"By noon you will have run it, broken it, and priced it."*
2. **The problem** - nurse reviewers, 90 seconds per case, payer + hospital customers on different clouds and on-prem.
3. **NIM in one picture** - container = model + optimized engine + OpenAI API; NIM Operator + GitOps on AKS/GKE/EKS/OKE/RKE2.
4. **What "portable" really means** - same chart; the diff is storage, scheduling, precision (show `git diff clusters/aks clusters/gke`).
5. **Physics** - decode floor = weights / bandwidth (A100 ~8 ms, L4 ~53 ms for 8B BF16); *"we say no to a GPU before we buy it."*
6. **Precision per GPU** - BF16 (Ampere) / FP8 (Ada, Hopper) / NVFP4 (Blackwell); NIM profiles pick the engine per GPU.
7. **Lab 1-2 checkpoint** - TTFT you measured; prompt tokens long vs adapter.
8. **LoRA as a product feature** - adapter = OCI artifact, release = Git commit, canary by model name, eval gate.
9. **Agents and safety** - tool calls, untrusted tool output, injection resistance as a release gate (🔁 Lab 03).
10. **Measure** - AIPerf sweep, operating point, $/1M tokens per cloud (table from `compare_clouds.py`).
11. **Data center** - air-gapped RKE2: mirror, host-cached model, no NGC key in the pod.
12. **Managed vs self-hosted** - scorecard (M10): conformance, latency, agentic score, $/1k summaries, data residency.
13. **What's new: Nemotron 3.5** - hybrid Mamba-MoE: 6-8 KiB KV/token vs 128 KiB; what it changes for cost and context.
14. **Operating it together** - weekly sync pack, fleet report, what's-new pins, drills.
15. **Next steps** - mutual action plan, PoC exit criteria, owners and dates.
