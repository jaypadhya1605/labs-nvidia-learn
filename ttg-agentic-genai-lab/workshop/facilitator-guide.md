# Facilitator guide

You are teaching engineers who will run this agent on their storm day. Keep the talk short, let them predict before
every run, and never quote a mock number as a GPU result.

## One week before
- Agree the date, the room, the number of attendees (16-30 works best for pairs) and the co-host from Tern.
- Send the prerequisites from [agenda.md](agenda.md) and ask everyone to run the smoke test.
- Fill the `[measured]` slides from your GPU runs (`results/ledger/`) and note the GPU, engine versions and date.

## The day before
- `python workshop/build_notebook.py`, then execute it on the mock: about 2-3 minutes end to end.
- Execute it once against the GPU VM (`TTG_BASE_URL=http://localhost:8000/v1`, vLLM up) and keep that ledger for §6.
- Record the 3-minute GPU demo: `engines/launch.sh vllm base`, one bench run, the ledger and the waterfall.
- Check the venue network: GitHub and PyPI reachable? If not, bring a wheelhouse on USB
  (`pip download -r requirements-dev.txt -d wheels`).

## Running it (timing cues)
| Clock | You say / do | If you run late |
|---|---|---|
| 0:00 | Slides 1-5. Ask: "How many agent-hours does a storm day take?" before showing slide 2. | Skip slide 3; the diagram is in the handout. |
| 0:10 | Exercise 1. Walk the room. Ask pairs to read one trace aloud: thought, action, observation. | Run only S0016. |
| 0:25 | Slides 6-10. Draw the stack once; point at the shared prefix on slide 8. | Merge slides 9 and 10. |
| 0:35 | Exercise 2. Stop everyone after §3 and collect three guesses for §4 before anyone runs it. | Skip the discussion question. |
| 0:50 | Break. Restart any stuck kernel; the mock restarts itself. | Shorten to 3 minutes. |
| 0:55 | Exercise 3. Pairs, two rounds. Hand out the printed drill table only after round 1. | One round. |
| 1:10 | GPU demo, or the recording. Show the waterfall, then the $ per 1,000 disruptions. | Recording only. |
| 1:20 | Slides 14-15. Ask the room what would make them trust the agent on a storm day. Survey. | Survey by email. |

## Expected answers (mock; measured when the notebook was built)
- Exercise 1: S0003 3 tool calls, S0005 5, S0016 11 (12 LLM calls). S0003's receipt earns nothing: weather is
  uncontrollable. S0016's $800 credit triggers human approval.
- Exercise 2: S0016 ~31,200 prompt tokens over 12 calls; ~71% is the shared system prompt; ideal cache reuse ~96%.
  `mock-base`: ~93% cache hits, ~3.6 tasks/s at concurrency 8. With the timestamp: 100% success, 0% cache hits,
  ~2.75 tasks/s at concurrency 8.
- Exercise 3: drill 8 shows 3/12 success with 0 retries and missing write actions; drill 10 shows 0% success with
  3 retries per task; drill 2 shows 1 retry per task and one extra LLM call; drill 1 shows the cache hit ratio at 0%.

## Common problems
| Problem | Fix |
|---|---|
| Port 8000 already in use | `TTG_WORKSHOP_PORT=8010` before starting Jupyter, or stop the other process |
| The notebook can't find the repo | Start Jupyter from the repo, or open the notebook from `workshop/` |
| Corporate proxy breaks localhost calls | `export NO_PROXY=localhost,127.0.0.1` before starting Jupyter |
| `pip install` fails on the venue network | The wheelhouse: `pip install --no-index --find-links wheels -r requirements-dev.txt` |
| Windows | WSL2 with Ubuntu; native Windows is not tested |
| A cell hangs for minutes | Interrupt the kernel, rerun the setup cell: it restarts the mock if needed |

## Questions you will get (and short answers)
- **"Why not just call a hosted frontier model?"** "You can, for prototyping on synthetic data. Guest data stays in your
  tenant and EU data in the EU, and storm-day volume makes per-token pricing your biggest variable cost. We'll show
  you the measured cost per disruption on your own GPUs, then you decide."
- **"Will it give away money?"** "Every voucher amount is checked against the policy oracle on 200 scenarios, and
  anything above $300 waits for a human. The limit lives in the tool, not in the prompt."
- **"How do EU guests stay in the EU?"** "A second deployment of the same chart in the EU region. Routing decides
  residency, not a prompt."
- **"What if a GPU fails on the storm day?"** "The capacity plan is N+1, and the engine autoscales on queue depth, not
  GPU utilization."
- **"Why an 8B model?"** "The policy is narrow and fully checkable. If the 8B passes the golden set, a 70B is twice the
  GPUs for the same answers (ADR-06)."

## Survey (5 questions, 2 minutes)
1. How useful was the workshop for your job? (1-5)
2. Which exercise taught you the most? (1 / 2 / 3)
3. Can you explain why prefix caching matters for agents to a colleague? (yes / not yet)
4. What would you need to trust the agent on a storm day?
5. Who else at Tern should see this?

## Within 48 hours
Send the repo link, the notebook, the recording, the slide PDF and a one-paragraph summary of what the room asked.
Log the questions in the technical win plan ([sales/technical-win-plan.md](../sales/technical-win-plan.md)).
