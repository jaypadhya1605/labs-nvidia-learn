# profiling/ - M5: profile and optimize latency, throughput, memory, I/O

The method, the experiments table and the Say-it lines are in [docs/08-profiling-playbook.md](../docs/08-profiling-playbook.md).

| Tool | Question | Needs |
|---|---|---|
| `torch_profile.sh [variant]` | What runs on the GPU during real agent traffic? (vLLM `--profiler-config`, 60 engine steps) | GPU VM, docker |
| `nsys_capture.sh [variant]` | The full timeline: kernels, NVTX, CUDA graphs, NCCL (Nsight Systems) | nsys on the host |
| `dcgm_watch.sh [seconds]` | Is the GPU busy, and on what? SM / tensor / DRAM activity (DCGM), nvidia-smi dmon | GPU VM |
| `analyze_trace.py <trace or dir>` | GPU busy %, time per kernel category, idle gaps, top-20 kernels (also `--nsys-csv`) | anywhere |
| `roofline_check.py` | Is decode at the memory roofline? floor vs measured ITL (🔁 Lab 01, archmath M2) | anywhere |
| `coldstart.py <log>` | How long until a new replica serves, and why? (download, load GiB/s, compile, graphs) | anywhere |

Everything that reads files runs on the laptop: copy `results/profiles/` and `results/logs/` back from the VM.
Traces can be opened in https://ui.perfetto.dev (torch) or the Nsight Systems GUI (`.nsys-rep`).
