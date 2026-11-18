# Collateral (M12)

| File | What it is | Audience |
|---|---|---|
| [blog.md](blog.md) | "Storm day: what accelerated inference does to the cost of an AI agent" (~1,100 words) | engineers and architects |

The blog is a draft until every `[measured]` placeholder holds a number from `results/ledger/` (with its GPU, engine
versions and date) and the checklist below is done.

## Before you publish
1. **Numbers.** Every number is either an estimate labeled as one (4 characters per token, oracle trajectories) or a
   measurement with its source file. Mock-engine numbers are labeled "on the mock engine" and never presented as GPU
   results.
2. **Fictional and synthetic.** Keep the statement that Tern Travel Group, its policy and its travelers are fictional
   and that all data is synthetic. No real customer, employer-confidential or partner information, ever.
3. **Your employer's policy.** Check your current employer's rules on public posts and side projects before
   publishing. If NVIDIA hires you, its communications review applies to anything you publish about its products.
4. **Licenses.** Llama 3.1 is under the Llama 3.1 Community License. If you publish anything derived from it (the M6
   LoRA adapter, for example), read the license and follow its attribution terms ("Built with Llama"). Name the engine
   and container versions you used (`env/pins.env`).
5. **No benchmark claims about other vendors.** The post compares configurations of one stack on one workload. Say
   which GPU and which date, and link the repo so anyone can rerun it.
6. **Links.** The repo link works without access to anything private, and the README explains the mock path.

## Where to post
Your blog or LinkedIn article first, then a short post with the waterfall chart (`results/ledger/waterfall.png`) and
the link. Reuse the workshop notebook as the "try it" call to action.

**Checkpoint:** a colleague who has not seen the lab can read the post in 6 minutes and tell you the three lessons.

**Say it:** "I measured an agent on four NVIDIA inference engines and wrote down what each optimization bought, in
dollars per decision."
