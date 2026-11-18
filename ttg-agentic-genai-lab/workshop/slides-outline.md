# Slides outline (15 slides, ~30 minutes of talk across the 90 minutes)

One idea and one number per slide. Every number names its source, so nobody has to take a slide on trust. `[measured]`
= fill from your own GPU runs (`results/ledger/`) before the workshop; never show mock numbers as GPU results.

| # | Title | The one number | Speaker note | Source |
|---|---|---|---|---|
| 1 | Storm day: an AI agent, measured | - | Who we are, what we'll do in 90 minutes, synthetic data only | - |
| 2 | The storm-day math | ~7,330 agent-hours in 6 hours | 40,000 travelers x 11 minutes; ~1,220 agents at once; a 3-hour queue today | [docs/01](../docs/01-customer-brief.md) |
| 3 | What the Disruption Concierge does | 5 steps | check, apply Tern Care v3, rebook/book/issue, read the receipt, message the guest; > $300 goes to a human | IROPS diagram in docs/01 |
| 4 | One task is a conversation | 6.5 LLM calls per task (4-12) | each tool call is one more model call; 3 to 11 tools per traveler | `ttg.world` golden set |
| 5 | Where the tokens go | ~96% of prompt tokens repeat | ~15,100 prompt tokens per task; the ~1,840-token system prompt rides on every call | notebook §2 |
| 6 | The stack | 1 protocol, 4 engines | LangGraph state machine + 2 MCP servers + one OpenAI-compatible engine | [docs/02](../docs/02-architecture-and-adrs.md) |
| 7 | 🔁 The physics | 4.8 ms per token floor | decode = bytes / bandwidth (16 GB / 3.35 TB/s on an H100); prefill = FLOPs | [docs/00](../docs/00-recall.md) |
| 8 | Prefix caching | prefill only the new ~620 tokens per task | vLLM automatic prefix caching, SGLang RadixAttention, TensorRT-LLM KV block reuse | ADR-02 |
| 9 | Speculation and FP8 | [measured] | n-gram drafts suit tool calls (ids, amounts); FP8 halves weight bytes; both behind the eval gate | ADR-07, ADR-08 |
| 10 | Why a JSON action protocol | 0 parser differences | the same prompt and the same expected output on every engine; retries become a metric | ADR-01 |
| 11 | What breaks in production | 10 drills | a timestamp in the prompt, a context sized for chat, TP without P2P, 4K photos, verbose thoughts | [drills/README.md](../drills/README.md) |
| 12 | The unit the CFO reads | $[measured] per 1,000 disruptions | GPU $/hour / (tasks per second at the SLO x 3,600) x 1,000; the operating point, not the peak | [docs/09](../docs/09-results-and-exec-readout.md) |
| 13 | What each step bought | [measured] waterfall | naive server -> batching -> prefix caching -> FP8 -> speculation -> LoRA prompt baking | `results/ledger/waterfall.png` |
| 14 | What the PoC will prove | 4 criteria | >= 95% policy-compliant, p95 <= 12 s, <= $2.00 per 1,000, a storm-day GPU plan with N+1 | [sales/poc-charter.md](../sales/poc-charter.md) |
| 15 | Next steps + survey | 2 minutes | the repo, the notebook, the PoC dates; survey QR code | [sales/mutual-action-plan.md](../sales/mutual-action-plan.md) |

## Rules for the deck
- One picture of the stack (slide 6), reused, never redrawn differently.
- No vendor comparison slide with someone else's benchmark. Slide 13 shows Tern's traffic on Tern's policy.
- Every slide with a GPU number says which GPU, which engine version and which date (`env/pins.env`).
- The deck works without the demo: slides 12-13 carry the recorded result if the live demo fails.
