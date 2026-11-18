"""profiling/ (FORK B): trace analyzer, nsys CSV, roofline, cold-start parser, DCGM summary, script dry runs.

Fixtures in tests/fixtures/profiling/: a two-step CUDA-graph decode trace (real kernel names, invented timings),
an nsys cuda_gpu_kern_sum CSV, vLLM / SGLang / trtllm start-up logs and nvidia-smi / dcgmi dmon output.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import importlib.util

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "profiling"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    """Load profiling/<name>.py by path: Python 3.15 adds a stdlib package called `profiling` (PEP 799)."""
    spec = importlib.util.spec_from_file_location(f"ttg_profiling_{name}", REPO / "profiling" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


at = _load("analyze_trace")
cs = _load("coldstart")
rc = _load("roofline_check")


# ---------------------------------------------------------------------------------------------- classify
@pytest.mark.parametrize("name,cat", [
    ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "GEMM"),
    ("nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNT", "GEMM"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>", "GEMM"),
    ("void vllm::cutlass_scaled_mm_sm90_fp8<...>", "GEMM"),                      # FP8 GEMM is still GEMM
    ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<...>>>", "attention"),
    ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16>", "attention"),
    ("void vllm::rotary_embedding_kernel<c10::BFloat16, true>", "attention"),
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "communication"),
    ("void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>", "communication"),
    ("void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>", "norm/act/elementwise"),
    ("void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel>", "norm/act/elementwise"),
    ("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0", "norm/act/elementwise"),
    ("void vllm::scaled_fp8_quant_kernel<c10::BFloat16, c10::Float8_e4m3fn>", "quantize"),
    ("void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>", "sampling"),
    ("void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float>", "sampling"),
    ("some_mystery_kernel", "other"),
])
def test_classify(name, cat):
    assert at.classify(name) == cat


def test_memcpy_by_trace_category():
    assert at.classify("anything", "gpu_memcpy") == "memcpy/memset"
    assert at.classify("Memset (Device)", "gpu_memset") == "memcpy/memset"


# --------------------------------------------------------------------------------------------- torch trace
@pytest.fixture(scope="module")
def summary():
    trace = at.load_trace(FIX / "torch_trace_decode.json")
    s = at.summarize(at.gpu_events(trace), top=20, n_gaps=10)
    s["calls"] = at.runtime_calls(trace)
    return s


def test_busy_time_is_a_union_not_a_sum(summary):
    # 2 steps x 994 us of back-to-back kernels (+ a 4 us memset hidden under a GEMM on another stream)
    assert summary["busy_ms"] == pytest.approx(1.988, abs=1e-6)
    assert summary["window_ms"] == pytest.approx(3.540, abs=1e-6)
    assert summary["busy_pct"] == pytest.approx(100 * 1988 / 3540, abs=0.01)
    assert summary["streams"] == 2 and summary["n_events"] == 29


def test_gaps(summary):
    b = {x["bucket"]: x for x in summary["gap_buckets"]}
    assert b["<10us"]["count"] == 26 and b[">1ms"]["count"] == 1
    assert summary["largest_gaps"][0]["gap_us"] == pytest.approx(1500.0)


def test_categories_and_top_kernels(summary):
    pct = {c["category"]: c["pct"] for c in summary["categories"]}
    assert pct["GEMM"] == pytest.approx(100 * 1640 / 1992, abs=0.01)
    assert pct["attention"] == pytest.approx(100 * 148 / 1992, abs=0.01)
    assert pct["sampling"] == pytest.approx(100 * 120 / 1992, abs=0.01)
    assert pct["memcpy/memset"] == pytest.approx(100 * 14 / 1992, abs=0.01)
    top = summary["top_kernels"][0]
    assert "tilesize256x128x64" in top["name"] and top["count"] == 2 and top["total_ms"] == pytest.approx(0.6)
    assert sum(c["pct"] for c in summary["categories"]) == pytest.approx(100.0)


def test_diagnosis_reads_the_numbers(summary):
    notes = " ".join(at.diagnose(summary, summary["calls"]))
    assert summary["calls"] == {"graph_launches": 2, "kernel_launches": 3}
    assert "GPU busy only 56%" in notes and "CUDA graphs are replaying" in notes and "GEMM is 82%" in notes


def test_eager_trace_is_flagged():
    ev = [{"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": i * 10.0, "dur": 2.0}
          for i in range(500)]
    ev += [{"ph": "X", "cat": "kernel", "name": "sm90_xmma_gemm_x", "ts": i * 10.0 + 3, "dur": 5.0,
            "args": {"stream": 7}} for i in range(500)]
    trace = {"traceEvents": ev}
    s = at.summarize(at.gpu_events(trace))
    notes = " ".join(at.diagnose(s, at.runtime_calls(trace)))
    assert "eager mode" in notes and "500 kernel launches" in notes


def test_directory_gz_and_cli(tmp_path):
    raw = (FIX / "torch_trace_decode.json").read_bytes()
    worker = tmp_path / "dp0_pp0_tp0_dcp0_ep0_rank0.1760000000.pt.trace.json.gz"
    worker.write_bytes(gzip.compress(raw))
    time.sleep(0.01)
    frontend = tmp_path / "host_1.async_llm.1760000001.pt.trace.json.gz"       # newer, but CPU only: skipped
    frontend.write_bytes(gzip.compress(json.dumps({"traceEvents": []}).encode()))
    assert at.find_trace(tmp_path) == worker
    out = tmp_path / "summary.json"
    r = subprocess.run([sys.executable, "profiling/analyze_trace.py", str(tmp_path), "--json", str(out)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "GPU busy 56.2%" in r.stdout and "Top 14 kernels" in r.stdout
    d = json.loads(out.read_text())
    assert d["source"].endswith(worker.name) and d["launch_calls"]["graph_launches"] == 2 and d["notes"]


def test_empty_trace_explains_itself():
    with pytest.raises(ValueError, match="no GPU events"):
        at.summarize([])


def test_nsys_kern_sum_csv():
    rows = at.parse_nsys_kern_sum((FIX / "nsys_kern_sum.csv").read_text())
    assert len(rows) == 10 and rows[0]["instances"] == 17505
    s = at.summarize_nsys(rows, top=5)
    pct = {c["category"]: c["pct"] for c in s["categories"]}
    assert pct["GEMM"] > 75 and pct["communication"] > 1 and pct["quantize"] > 0
    assert len(s["top_kernels"]) == 5 and s["busy_pct"] is None
    r = subprocess.run([sys.executable, "profiling/analyze_trace.py", "--nsys-csv", str(FIX / "nsys_kern_sum.csv")],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "no idle-gap analysis" in r.stdout
    with pytest.raises(ValueError):
        at.parse_nsys_kern_sum("not,a,report\n1,2,3\n")


# ------------------------------------------------------------------------------------------------ roofline
@pytest.fixture(scope="module")
def llama():
    return rc.dense_model(rc.load_config(None))


def test_llama_31_8b_sizes(llama):
    assert llama["total_params"] == 8_030_261_248
    assert llama["kv_bytes_per_token_bf16"] == 131_072                     # 128 KiB per token in BF16


def test_floor_bf16_vs_fp8(llama):
    bf16 = rc.step_floor(llama, "H100-SXM", "bf16", "bf16", batch=1, ctx=2600)
    assert bf16["bound"] == "memory"
    assert bf16["bytes_gb"] == pytest.approx(15.35, abs=0.01) and bf16["floor_ms"] == pytest.approx(4.58, abs=0.01)
    fp8 = rc.step_floor(llama, "H100-SXM", "fp8", "bf16", batch=1, ctx=2600)
    assert fp8["floor_ms"] == pytest.approx(2.50, abs=0.01)
    kv8 = rc.step_floor(llama, "H100-SXM", "bf16", "fp8", batch=64, ctx=2600)
    kv16 = rc.step_floor(llama, "H100-SXM", "bf16", "bf16", batch=64, ctx=2600)
    assert kv8["kv_gb"] == pytest.approx(kv16["kv_gb"] / 2)
    assert kv16["max_batch_kv"] == 167 and kv16["fits"]
    assert not rc.step_floor(llama, "H100-SXM", "bf16", "bf16", batch=256, ctx=2600)["fits"]


def test_compare_and_verdicts(llama):
    floor = rc.step_floor(llama, "H100-SXM", "bf16", "bf16", batch=1, ctx=2600)
    c = rc.compare(floor, 6.1, "H100-SXM")
    assert c["bw_fraction"] == pytest.approx(0.751, abs=0.002) and "typical" in c["verdict"]
    assert "at the roofline" in rc.compare(floor, 4.9, "H100-SXM")["verdict"]
    assert "overhead-bound" in rc.compare(floor, 12.0, "H100-SXM")["verdict"]
    assert "faster than the floor" in rc.compare(floor, 3.0, "H100-SXM")["verdict"]


def test_moe_configs_are_sent_to_archmath():
    with pytest.raises(ValueError, match="archmath"):
        rc.dense_model({"hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16,
                        "intermediate_size": 1024, "vocab_size": 1000, "num_local_experts": 8})


def test_matches_archmath_and_shares_its_gpu_table(llama):
    am = pytest.importorskip("archmath.archmath")
    assert rc.GPUS == am.GPUS, "profiling/roofline_check.py GPUS drifted from archmath/archmath.py"
    m = am.analyze(am.load_config("llama-3.1-8b"))
    for prec, batch in (("bf16", 1), ("bf16", 32), ("fp8", 16)):
        theirs = am.roofline(m, "H100-SXM", prec, batch, 2600)
        mine = rc.step_floor(llama, "H100-SXM", prec, "bf16", batch, 2600)
        assert mine["floor_ms"] == pytest.approx(theirs["step_ms"], rel=1e-3)


def test_aiperf_export(tmp_path, llama):
    run = tmp_path / "c16"
    run.mkdir()
    (run / "profile_export_aiperf.json").write_text(json.dumps({
        "inter_token_latency": {"unit": "ms", "avg": 9.1, "p50": 8.8, "p95": 12.0},
        "input_sequence_length": {"unit": "tokens", "avg": 2400.0},
        "output_sequence_length": {"unit": "tokens", "avg": 80.0},
        "input_config": {"benchmark": {"phases": [{"name": "warmup", "type": "concurrency", "concurrency": 4},
                                                  {"name": "profiling", "type": "concurrency", "concurrency": 16}]}}}))
    ai = rc.read_aiperf(str(tmp_path))
    assert ai["itl_ms"] == 8.8 and ai["concurrency"] == 16 and ai["isl"] == 2400.0
    r = subprocess.run([sys.executable, "profiling/roofline_check.py", "--gpu", "H100-SXM", "--aiperf", str(run),
                        "--json"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["floor"]["batch"] == 16 and d["floor"]["ctx"] == 2440 and d["measured"]["measured_itl_ms"] == 8.8


def test_sweep_cli():
    r = subprocess.run([sys.executable, "profiling/roofline_check.py", "--sweep", "1,64,256"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "NO - KV > free HBM" in r.stdout and "Prefix caching stores" in r.stdout


# ---------------------------------------------------------------------------------------------- cold start
def test_vllm_log():
    r = cs.parse_log((FIX / "vllm_startup.log").read_text())
    assert r["engine"] == "vllm"
    assert r["cold_start_s"] == pytest.approx(125.2, abs=0.01)
    assert r["download_s"] == pytest.approx(61.912345) and r["load_weights_s"] == 5.12
    assert r["weights_gib"] == 14.99 and r["load_gib_s"] == pytest.approx(14.99 / 5.12)
    assert r["kv_tokens"] == 444_096 and r["max_model_len"] == 16_384 and r["max_concurrency"] == 27.11
    assert r["graph_capture_s"] == 9.0 and r["engine_init_s"] == 45.88 and r["compile_s"] == 28.2
    tips = " ".join(cs.suggestions(r))
    assert "downloading weights" in tips and "torch.compile" in tips and "storage-bound" not in tips


def test_sglang_log_own_timestamps():
    r = cs.parse_log((FIX / "sglang_startup.log").read_text())
    assert r["engine"] == "sglang" and r["cold_start_s"] == 52.0
    assert r["load_weights_s"] == 4.86 and r["weights_gib"] == 15.06
    assert r["kv_tokens"] == 413_000 and r["kv_mem_gib"] == pytest.approx(50.42)
    assert r["graph_capture_s"] == 26.4
    assert any("CUDA graph capture" in t for t in cs.suggestions(r))


def test_trtllm_wall_time_only_and_slow_disk_hint():
    r = cs.parse_log((FIX / "trtllm_startup.log").read_text())
    assert r["engine"] == "trtllm" and r["cold_start_s"] == 90.5
    slow = cs.parse_log("2026-10-17T00:00:00Z x\n"
                        "2026-10-17T00:00:01Z INFO 10-17 00:00:01 [x.py:1] Loading weights took 30.00 seconds\n"
                        "2026-10-17T00:00:02Z INFO 10-17 00:00:02 [x.py:2] Model loading took 15.00 GiB memory and "
                        "30.5 seconds\n2026-10-17T00:01:00Z INFO:     Application startup complete.\n")
    assert slow["load_gib_s"] == pytest.approx(0.5)
    assert any("runai_streamer" in t for t in cs.suggestions(slow))
    assert any("No wall time" in t for t in cs.suggestions(cs.parse_log("no timestamps here\n", "vllm")))


def test_coldstart_cli_json():
    r = subprocess.run([sys.executable, "profiling/coldstart.py", *map(str, sorted(FIX.glob("*_startup.log"))),
                        "--json"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert {x["engine"] for x in json.loads(r.stdout)} == {"vllm", "sglang", "trtllm"}


# --------------------------------------------------------------------------------------------- shell scripts
def test_dcgm_summary_only(tmp_path):
    for f in ("nvidia-smi-dmon.txt", "dcgmi-dmon.txt"):
        shutil.copy(FIX / f, tmp_path / f)
    r = subprocess.run(["bash", "profiling/dcgm_watch.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "SUMMARY_ONLY": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "summary.txt").read_text()
    assert "dcgmi-dmon.txt: 3 samples" in text and "nvidia-smi-dmon.txt: 3 samples" in text
    assert "TENSO" in text and "DRAMA" in text and "sm " in text
    rows = (tmp_path / "dcgmi-dmon.csv").read_text().splitlines()
    assert rows[0].startswith("Entity,SMACT") and len(rows) == 4


def _dry(script: str, *args, **env) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items()
            if k not in ("EXTRA_ARGS", "EXTRA_DOCKER", "ENTRYPOINT_WRAPPER", "GPUS", "PROFILES", "NSYS")}
    full.update({"DRY_RUN": "1", **env})
    return subprocess.run(["bash", script, *args], cwd=REPO, env=full, capture_output=True, text=True, timeout=60)


def test_torch_profile_dry_run():
    r = _dry("profiling/torch_profile.sh", "eager")
    assert r.returncode == 0, r.stderr
    assert '"profiler":"torch"' in r.stdout and '"torch_profiler_dir":"/profiles"' in r.stdout
    assert '"max_iterations":60' in r.stdout and '"ignore_frontend":true' in r.stdout
    assert ":/profiles" in r.stdout and "--enforce-eager" in r.stdout and "PLAN:" in r.stdout


def test_nsys_capture_dry_run_mounts_the_host_install():
    r = _dry("profiling/nsys_capture.sh", "base", NSYS="/opt/nvidia/nsight-systems-cli/2026.1.1/target-linux-x64/nsys")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "-v /opt/nvidia/nsight-systems-cli/2026.1.1:/nsys:ro" in out
    assert "--entrypoint /nsys/target-linux-x64/nsys" in out
    for f in ("--capture-range=cudaProfilerApi", "--trace-fork-before-exec=true", "--cuda-graph-trace=node",
              '"profiler":"cuda"', "VLLM_WORKER_MULTIPROC_METHOD=spawn", "--init"):
        assert f in out, f
    assert "vllm serve meta-llama/Llama-3.1-8B-Instruct" in out


def test_bench_file_mode_estimates_itl_per_level(tmp_path, llama):
    # 2 tasks at concurrency 8: 7 calls, 420 output tokens, 30 ms TTFT, 3.0 s of LLM time out of 4.0 s per task
    task = {"llm_calls": 7, "completion_tokens": 420, "prompt_tokens": 7 * 2300, "first_ttft_s": 0.03,
            "llm_time_s": 3.0, "latency_s": 4.0, "concurrency": 8}
    bench = {"label": "vllm-fp8", "engine": "vllm", "variant": "fp8", "gpu": "NVIDIA H100 80GB HBM3", "gpus": 1,
             "levels": [{"concurrency": 8}], "tasks": [task, dict(task)]}
    f = tmp_path / "vllm-fp8.json"
    f.write_text(json.dumps(bench))
    est = rc.estimate_from_tasks(bench["tasks"], 8)
    assert est["itl_ms"] == pytest.approx((3.0 - 7 * 0.03) / (420 - 7) * 1e3)
    assert est["batch"] == 6 and est["ctx"] == 2330
    r = rc.from_bench(str(f), llama, None, None, None, 1)
    assert r["gpu"] == "H100-SXM" and r["precision"] == "fp8" and r["kv"] == "bf16"
    assert r["levels"][0]["floor_ms"] == pytest.approx(rc.step_floor(llama, "H100-SXM", "fp8", "bf16", 6, 2330)["floor_ms"])
    out = subprocess.run([sys.executable, "profiling/roofline_check.py", str(f)], cwd=REPO, capture_output=True,
                         text=True)
    assert out.returncode == 0 and "ESTIMATED" in out.stdout and " 6.76 " in out.stdout
    assert rc.guess_gpu("L40S") == "L40S" and rc.guess_gpu("A100-SXM4-80GB") == "A100-80GB-SXM"
