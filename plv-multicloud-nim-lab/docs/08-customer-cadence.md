# Customer cadence (M11) - how the SA runs the relationship after the demo

## Weekly technical sync (30 min, same agenda every week)
```bash
python3 fleet/fleet_report.py --out results/fleet.md          # status across clouds (+ results/fleet.json)
python3 fleet/whats_new.py --json results/whats-new.json       # pinned vs upstream
python3 fleet/weekly_sync.py --fleet results/fleet.json --whats-new results/whats-new.json --out results/weekly-sync.md
```
Send the pack 24 h before the call. In the call: status 5, releases 5, new features 5, issues 10, decisions 5.
Every action item gets an owner and a date in `fleet/actions.yaml`.

## "What's new" process (new features without surprises)
1. `whats_new.py` flags a component behind its pin (e.g. GPU Operator patch, NIM release, a new model like Nemotron 3.5).
2. Read the release notes; classify: **security** (patch now), **fix we need**, **feature that matters**, **ignore**.
3. Test it in a branch: bump `env/pins.env` -> CI (`tests/chart/test_pins.py` forces GitOps + chart to match) ->
   perf gate on one cluster -> merge -> Argo CD rolls it out fleet-wide.
4. Tell the customer in the next sync with the measured impact, never the press-release claim.

## Troubleshooting with the customer (live call rules)
- Symptoms first, then the 6 diagnosis commands (drills/README.md); share the screen of Grafana, not of kubectl.
- One owner, one written summary within 24 h: symptom, root cause, fix, prevention (a drill if it can recur).

## Escalation paths
NVIDIA AI Enterprise support case (NIM, GPU Operator) · CSP support case (quota, capacity, node faults) · NVIDIA
engineering via the account team (bugs with a minimal repro from this repo).

## Quarterly business review
Fleet cost trend, $/1k summaries vs target, incidents and drills, roadmap (agents, Nemotron 3.5, new clouds),
licensing renewals and reservations.
