# M12 - Teach it: workshop, notebook, blog (2 hours)

**A solutions architect scales by teaching: one 90-minute workshop leaves 20 engineers who can explain why prefix
caching decides the cost of an agent.** This module packages Lab 05 as a hands-on workshop for Tern's engineers and a
public blog post.

| File | What it is |
|---|---|
| [agenda.md](agenda.md) | 90-minute agenda, prerequisites, before-the-day list |
| [slides-outline.md](slides-outline.md) | 15 slides: title, one number, speaker note, source |
| [exercises.md](exercises.md) | attendee handout: three exercises, predictions first, answers at the bottom |
| [facilitator-guide.md](facilitator-guide.md) | timing cues, expected answers, common problems, questions, survey |
| `build_notebook.py` -> `ttg-workshop.ipynb` | the attendee notebook (generated; starts the mock itself) |
| [collateral/blog.md](../collateral/blog.md) | blog draft: "Storm day: what accelerated inference does to the cost of an AI agent" |

## Steps
1. Build the notebook and run it end to end on the mock (about 2-3 minutes):
   ```bash
   python workshop/build_notebook.py
   jupyter-execute --timeout=600 workshop/ttg-workshop.ipynb      # nbclient's runner; outputs are not saved
   ```
2. Do the three exercises yourself, with predictions, and time each one. Cut whatever runs over 15 minutes.
3. Fill the `[measured]` slides from your M4-M10 GPU runs, then record the 3-minute GPU demo.
4. Rehearse the talk segments with a timer: 10 minutes each, no more.
5. Fill the blog's `[measured]` placeholders from `results/ledger/` and follow [collateral/README.md](../collateral/README.md)
   before publishing anything.

**Checkpoint:** the notebook runs clean on a fresh laptop, you can give the three talk segments in 30 minutes, and every
number on your slides has a source file.

**Say it:** "In 90 minutes your engineers will run the agent, measure it, break it and price it, so on storm day they
know which number to watch and why."
