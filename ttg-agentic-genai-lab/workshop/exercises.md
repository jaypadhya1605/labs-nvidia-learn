# Workshop exercises (attendee handout)

Work in the notebook (`workshop/ttg-workshop.ipynb`). Write each prediction down **before** you run the cell: the gap
between your guess and the result is the lesson. Numbers on the mock engine show the shape of a result, not a GPU
result. Answers are at the bottom; don't scroll yet.

## Exercise 1 - One disruption, step by step (notebook §1, 15 minutes)
Three travelers from the golden set:

| Scenario | What the guest says (short) | Your guess: tool calls | Result | Oracle plan |
|---|---|---|---|---|
| S0003 | TN138 to JFK is delayed, traveling alone, receipt for food uploaded | | | |
| S0005 | TN745 is delayed, party of 2, receipt for food uploaded | | | |
| S0016 | TN531 was cancelled, party of 4, taxi receipt uploaded | | | |

1. Read the three messages in the notebook. `flight_status`, `loyalty_profile` and `policy_search` always come first.
   Use `data/policies/tern-care-v3.md` for the rest.
2. Run the agent on each one and read the trace: every `LLM` line is one model call, every `TOOL` line one MCP call.
3. Compare with the oracle plan. Two questions:
   - S0003 uploaded a receipt. Why is there no reimbursement?
   - S0016 needs a human approval. Which voucher triggers it?

## Exercise 2 - Where the tokens go, and what they cost (notebook §2-4, 15 minutes)
| Question | Your guess | Result |
|---|---|---|
| Share of S0016's prompt tokens that is the shared system prompt | | |
| Ideal prefix-cache hit ratio for S0016 | | |
| Measured cache hit ratio, `mock-base` | | |
| After the timestamp: success rate | | |
| After the timestamp: cache hit ratio | | |
| After the timestamp: tasks per second at concurrency 8, vs before | | |

1. §2 counts S0016's prompt tokens per call from the real prompt builder (4 characters per token, an estimate).
2. §3 benchmarks the agent on 16 golden tasks at concurrency 1 and 8 and prices it at $2.49 per GPU-hour.
3. §4 adds `Current time: ...` to the top of the system prompt (`TTG_BUST_CACHE=1`) and runs the same benchmark.
4. Discuss with your neighbor: on a GPU, which latency number moves most, TTFT or inter-token latency? Why?

## Exercise 3 - Mystery fault, in pairs (notebook §5, 15 minutes)
1. Person A runs `mystery()` and doesn't say which drill it picked.
2. Person B reads the two columns (healthy, mystery) and answers three questions out loud:
   - Which metric moved first, and by how much?
   - What is the most likely root cause?
   - Which gate would have caught it before production?
3. Run `reveal()`, read that drill in `drills/README.md`, and say its "Say it" line. Swap roles and repeat.

---

## Answers (after the exercises)
**Exercise 1.** S0003: 3 tool calls, 4 LLM calls. The delay is weather, not Tern's fault, so the receipt earns nothing
(policy section 7: reimbursement only for controllable causes). S0005: 5 tool calls (adds `read_receipt` and one
`issue_voucher`: a $55.60 reimbursement). S0016: 11 tool calls, 12 LLM calls: rebook, hotel (2 rooms for 4 people),
read the receipt, then meal, credit and reimbursement vouchers. The $800 credit (cancelled, controllable: $200 x 4)
is above $300, so it goes to a human.

**Exercise 2 (mock, measured when the notebook was built; yours will differ a little).** S0016 sends ~31,200 prompt
tokens over 12 calls; the shared system prompt is ~71% of them, and an ideal prefix cache would reuse ~96%.
`mock-base` measured ~93% cache hits. After the timestamp: success still 100%, cache hits 0%, and tasks per second at
concurrency 8 fell from ~3.6 to ~2.75 (~23% fewer). On a GPU, TTFT moves most: every call re-prefills ~1,840 system
tokens plus the history; decode speed per token barely changes.

**Exercise 3.** Laptop drills 1, 2, 8 and 10: the first metric to move is, in order, the cache hit ratio, retries per
task, the violations (with success dropping while the guest messages still promise everything), and output tokens or
retries. Gates: perf gate (1, 10), eval gate (2, 8, 10).
