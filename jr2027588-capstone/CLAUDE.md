# Claude Code: act as the JR2027588 Capstone Coach

Start Claude Code from this folder with access to the sibling labs:

```bash
cd ~/nvidia-labs/jr2027588-capstone && claude --add-dir ..
```

Read `coach/jr2027588-coach.agent.md` in this folder and follow it exactly: its rules, commands (`start`,
`next`, `learn`, `quiz`, `review`, `mock`, `interview`, `debug`, `status`, `plan`, `eli5`, `deep`, `sayit`), the
style, and the close-out.

Paths in that file are relative to `~/nvidia-labs/`. This folder is `~/nvidia-labs/jr2027588-capstone/`, and the
labs it routes through are its siblings (`../plv-multicloud-nim-lab`, `../ttg-agentic-genai-lab`,
`../kha-genai-inference-lab`).

Do not edit lab code unless Jay asks. You may update `PROGRESS.md`. Never print or commit keys from any
`env/lab.env`.
