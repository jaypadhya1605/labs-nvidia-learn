# Performance baselines (🔁 Lab 02)

Arm the CI gate (`bench/perf_gate.py`, `.github/workflows/perf-gate.yml`) with a known-good AIPerf export
PER CLUSTER - a baseline from AKS says nothing about GKE:

```bash
cp results/aks-synth-600-200-<ts>/c32/profile_export_aiperf.json bench/baselines/aks-synth-600-200-c32.json
cp results/gke-synth-600-200-<ts>/c32/profile_export_aiperf.json bench/baselines/gke-synth-600-200-c32.json
git add bench/baselines && git commit -m "perf baselines: NIM 2.0.13, AKS A100 bf16 / GKE RTX PRO 6000 nvfp4"
```

Re-baseline deliberately (new GPU type, new NIM version you accepted, new precision), never automatically.
