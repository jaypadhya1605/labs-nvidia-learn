# Performance baselines

Copy a known-good AIPerf export here to arm the CI gate (`bench/perf_gate.py`, `.github/workflows/perf-gate.yml`):

```bash
cp results/member-<run-id>/c16/profile_export_aiperf.json bench/baselines/member-c16.json
git add bench/baselines/member-c16.json && git commit -m "perf baseline: member @ c16, NIM <tag>, <gpu>"
```

Re-baseline deliberately (new GPU type, new NIM version you accepted), never automatically.
