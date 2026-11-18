"""engines/ (FORK B): launcher dry runs, fail-fast combos, preflight, trtllm YAMLs, script hygiene, hf baseline.

Offline: DRY_RUN=1 never calls docker; the hf server runs with a fake backend (no torch / transformers).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted([*REPO.glob("engines/*.sh"), *REPO.glob("profiling/*.sh"), *REPO.glob("finetune/*.sh")])
SECRET = "SECRET-do-not-print-1234"


def launch(engine: str, variant: str, **env) -> subprocess.CompletedProcess:
    full = {**os.environ, "DRY_RUN": "1", "HF_TOKEN": "hf_" + SECRET, "NGC_API_KEY": "nvapi-" + SECRET}
    for k in ("GPUS", "MODEL", "FP8_MODE", "EXTRA_ARGS", "EXTRA_DOCKER", "ENTRYPOINT_WRAPPER", "NIM_MODEL_PROFILE",
              "BIND", "PORT"):
        full.pop(k, None)
    full.update(env)
    return subprocess.run(["bash", "engines/launch.sh", engine, variant], cwd=REPO, env=full,
                          capture_output=True, text=True, timeout=60)


def ok(engine: str, variant: str, **env) -> dict:
    r = launch(engine, variant, **env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    out = {"cmd": lines[0], "all": r.stdout + r.stderr}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip()] = v.strip()
    out["flags"] = out.get("ENGINE_FLAGS", "").split()
    return out


# ------------------------------------------------------------------------------------------- script hygiene
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_scripts_have_usage_header_and_strict_mode(path):
    text = path.read_text()
    head = text.splitlines()[:16]
    assert head[0] == "#!/usr/bin/env bash"
    assert head[1].startswith(f"# {path.relative_to(REPO)} - "), "line 2 names the script and says what it does"
    assert any("Usage" in ln for ln in head), "usage comment in the header"
    assert "set -euo pipefail" in text
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    r = subprocess.run(["shellcheck", *map(str, SCRIPTS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_no_bash4_only_syntax():
    """macOS ships bash 3.2: no associative arrays, mapfile/readarray, ${x,,} or ${x^^}."""
    for p in SCRIPTS:
        t = p.read_text()
        for bad in ("declare -A", "mapfile", "readarray", ",,}", "^^}", "|&"):
            assert bad not in t, f"{p.name}: {bad}"


# ------------------------------------------------------------------------------------------------- vLLM
def test_vllm_base_is_fair_and_observable():
    o = ok("vllm", "base")
    for f in ("--enable-prompt-tokens-details", "--enable-force-include-usage", "--enable-prefix-caching",
              "--served-model-name", "--max-model-len", "--gpu-memory-utilization"):
        assert f in o["flags"]
    assert "--served-model-name ttg-llm" in o["cmd"]
    assert "vllm/vllm-openai:v0.31.0" in o["cmd"]
    assert "-e VLLM_SERVER_DEV_MODE=1" in o["cmd"]                      # POST /reset_prefix_cache for the bench
    assert "-p 127.0.0.1:8000:8000" in o["cmd"]                         # dev endpoints stay local by default
    assert "/root/.cache/vllm" in o["cmd"]                              # torch.compile cache survives restarts
    assert "TTG_MODEL=ttg-llm" in o["AGENT"]
    assert "--metrics-url http://localhost:8000/metrics" in o["BENCH"]
    assert "--reset-url http://localhost:8000/reset_prefix_cache" in o["BENCH"]
    assert "--label vllm-base" in o["BENCH"]


def test_vllm_variants():
    assert "--no-enable-prefix-caching" in ok("vllm", "nocache")["flags"]
    assert "--enable-prefix-caching" not in ok("vllm", "nocache")["flags"]
    spec = ok("vllm", "spec")
    assert '"method":"ngram"' in spec["cmd"] and "--speculative-config" in spec["flags"]
    assert "--enforce-eager" in ok("vllm", "eager")["flags"]
    assert "--kv-cache-dtype fp8" in ok("vllm", "fp8kv")["cmd"]
    vlm = ok("vllm", "vlm")
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in vlm["cmd"] and "--limit-mm-per-prompt" in vlm["flags"]
    assert "--max-model-len 32768" in vlm["cmd"]


def test_vllm_fp8_checkpoint_vs_online():
    ck = ok("vllm", "fp8")
    assert "nvidia/Llama-3.1-8B-Instruct-FP8" in ck["cmd"] and "--quantization" not in ck["flags"]
    on = ok("vllm", "fp8", FP8_MODE="online")
    assert "meta-llama/Llama-3.1-8B-Instruct " in on["cmd"] and "--quantization fp8" in on["cmd"]


def test_vllm_lora_serves_the_adapter_by_name():
    o = ok("vllm", "lora")
    assert "--lora-modules ttg-short=/adapters/ttg-short" in o["cmd"]
    assert "--max-lora-rank 16" in o["cmd"] and "--enable-lora" in o["flags"]
    assert ":/adapters:ro" in o["cmd"]
    assert "TTG_MODEL=ttg-short" in o["AGENT"]


def test_parallel_variants_need_two_gpus():
    tp = ok("vllm", "tp2")
    assert "--tensor-parallel-size 2" in tp["cmd"] and '"device=0,1"' in tp["cmd"]
    assert "--gpus 2" in tp["BENCH"]
    assert "--tensor-parallel-size 4" in ok("vllm", "tp2", GPUS="0,1,2,3")["cmd"]
    assert "--pipeline-parallel-size 2" in ok("vllm", "pp2")["cmd"]
    assert "--data-parallel-size 2" in ok("vllm", "dp2")["cmd"]
    r = launch("vllm", "tp2", GPUS="0")
    assert r.returncode == 2 and "at least 2 GPUs" in r.stderr


def test_bind_and_port_overrides():
    o = ok("vllm", "base", BIND="0.0.0.0", PORT="9000")
    assert "-p 0.0.0.0:9000:8000" in o["cmd"]
    assert "TTG_BASE_URL=http://localhost:9000/v1" in o["AGENT"]


def test_entrypoint_wrapper_is_vllm_only_and_not_counted_as_engine_flags():
    o = ok("vllm", "base", ENTRYPOINT_WRAPPER="/nsys/bin/nsys profile --capture-range=cudaProfilerApi -o /p/x")
    assert "--entrypoint /nsys/bin/nsys" in o["cmd"]
    assert "profile --capture-range=cudaProfilerApi -o /p/x vllm serve" in o["cmd"]
    assert "--capture-range=cudaProfilerApi" not in o["flags"]
    r = launch("sglang", "base", ENTRYPOINT_WRAPPER="/nsys/bin/nsys profile")
    assert r.returncode == 2 and "vllm only" in r.stderr


def test_extra_args_and_docker_are_appended():
    o = ok("vllm", "base", EXTRA_ARGS='--profiler-config {"profiler":"torch"}', EXTRA_DOCKER="-v /tmp/p:/profiles")
    assert "-v /tmp/p:/profiles" in o["cmd"]
    assert """--profiler-config '{"profiler":"torch"}'""" in o["cmd"]


# ------------------------------------------------------------------------------------------- SGLang / TRT-LLM / NIM
def test_sglang():
    b = ok("sglang", "base")
    for f in ("--enable-metrics", "--enable-cache-report", "--model-path", "--context-length", "--mem-fraction-static"):
        assert f in b["flags"]
    assert "lmsysorg/sglang:v0.5.22" in b["cmd"]
    assert "--reset-url http://localhost:8000/flush_cache" in b["BENCH"]
    assert "--disable-radix-cache" in ok("sglang", "nocache")["flags"]
    assert "--speculative-algorithm NGRAM" in ok("sglang", "spec")["cmd"]
    assert "--kv-cache-dtype fp8_e4m3" in ok("sglang", "fp8kv")["cmd"]
    assert "--disable-cuda-graph" in ok("sglang", "eager")["flags"]
    lora = ok("sglang", "lora")
    assert "--lora-paths ttg-short=/adapters/ttg-short" in lora["cmd"]
    assert "TTG_MODEL=ttg-llm:ttg-short" in lora["AGENT"]                 # SGLang: base-model:adapter
    assert "--tp-size 2" in ok("sglang", "tp2")["cmd"]


def test_trtllm():
    b = ok("trtllm", "base")
    assert "--extra_llm_api_options /engines/trtllm/base.yaml" in b["cmd"]
    assert "--served_model_name ttg-llm" in b["cmd"]
    assert "trtllm-serve serve" in b["cmd"] and "--entrypoint bash" in b["cmd"]   # detects the serve subcommand
    assert "--reset-url" not in b["BENCH"]                                        # no reset endpoint
    assert "nvidia/Llama-3.1-8B-Instruct-FP8" in ok("trtllm", "fp8")["cmd"]
    assert "/engines/trtllm/nocache.yaml" in ok("trtllm", "nocache")["cmd"]
    assert "/engines/trtllm/eager.yaml" in ok("trtllm", "eager")["cmd"]
    kv8 = ok("trtllm", "fp8kv")["cmd"]
    assert "/engines/trtllm/fp8kv.yaml" in kv8 and "--kv_cache_dtype" not in kv8      # KvCacheConfig.dtype in YAML
    assert "--tp_size 2" in ok("trtllm", "tp2")["cmd"]
    r = launch("trtllm", "fp8", FP8_MODE="online")
    assert r.returncode == 2 and "pre-quantized" in r.stderr


def test_nim():
    b = ok("nim", "base")
    assert "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13" in b["cmd"]
    assert "-e NGC_API_KEY " in b["cmd"] + " "                       # passed by NAME
    assert "NIM_SERVED_MODEL_NAME=ttg-llm" in b["cmd"] and "NIM_ENABLE_KV_CACHE_REUSE=1" in b["cmd"]
    assert "--metrics-url http://localhost:8000/v1/metrics" in b["BENCH"]
    assert "NIM_ENABLE_KV_CACHE_REUSE=0" in ok("nim", "nocache")["cmd"]
    assert launch("nim", "fp8").returncode == 2
    assert "-e NIM_MODEL_PROFILE" in ok("nim", "fp8", NIM_MODEL_PROFILE="abc123")["cmd"]


def test_hf_baseline():
    o = ok("hf", "base")
    assert "/engines/hf_naive_server.py" in o["cmd"] and "--served-model-name ttg-llm" in o["cmd"]
    assert "--levels 1,4 --n 8" in o["BENCH"] and "--metrics-url" not in o["BENCH"]


@pytest.mark.parametrize("engine,variant,why", [
    ("trtllm", "spec", "verify the speculative_config"), ("trtllm", "lora", "vllm or sglang"),
    ("trtllm", "dp2", "load balancer"), ("trtllm", "vlm", "vllm vlm"), ("sglang", "vlm", "vllm vlm"),
    ("nim", "eager", "profile"), ("nim", "lora", "NIM_PEFT_SOURCE"), ("hf", "nocache", "base only")])
def test_unsupported_combinations_fail_fast_with_a_reason(engine, variant, why):
    r = launch(engine, variant)
    assert r.returncode == 2
    assert "is not supported" in r.stderr and why in r.stderr


def test_unknown_engine_variant_and_usage():
    assert launch("tgi", "base").returncode == 2
    assert launch("vllm", "turbo").returncode == 2
    r = subprocess.run(["bash", "engines/launch.sh"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2 and "Usage: engines/launch.sh <engine> <variant>" in r.stdout


def test_secrets_are_never_printed():
    for engine, variant in [("vllm", "base"), ("sglang", "lora"), ("trtllm", "fp8"), ("nim", "base"), ("hf", "base")]:
        r = launch(engine, variant)
        assert SECRET not in r.stdout + r.stderr


# --------------------------------------------------------------------------------------------- preflight
def test_preflight_list_covers_every_variant_flag():
    r = subprocess.run(["bash", "engines/preflight.sh", "--list", "vllm"], cwd=REPO, capture_output=True, text=True)
    flags = set(r.stdout.split())
    for f in ("--enable-prefix-caching", "--no-enable-prefix-caching", "--quantization", "--kv-cache-dtype",
              "--speculative-config", "--enforce-eager", "--enable-lora", "--lora-modules", "--max-lora-rank",
              "--tensor-parallel-size", "--pipeline-parallel-size", "--data-parallel-size", "--limit-mm-per-prompt",
              "--enable-prompt-tokens-details", "--enable-force-include-usage"):
        assert f in flags, f
    r = subprocess.run(["bash", "engines/preflight.sh", "--list", "sglang"], cwd=REPO, capture_output=True, text=True)
    assert {"--disable-radix-cache", "--speculative-algorithm", "--lora-paths", "--enable-metrics"} <= set(r.stdout.split())
    r = subprocess.run(["bash", "engines/preflight.sh", "--list", "trtllm"], cwd=REPO, capture_output=True, text=True)
    assert {"--extra_llm_api_options", "--served_model_name", "--tp_size"} <= set(r.stdout.split())
    assert "--kv_cache_dtype" not in r.stdout.split()                      # set in engines/trtllm/fp8kv.yaml instead


def test_trtllm_yamls():
    base = yaml.safe_load((REPO / "engines/trtllm/base.yaml").read_text())
    assert base["kv_cache_config"]["enable_block_reuse"] is True and base["enable_iter_perf_stats"] is True
    noc = yaml.safe_load((REPO / "engines/trtllm/nocache.yaml").read_text())
    assert noc["kv_cache_config"]["enable_block_reuse"] is False
    eager = yaml.safe_load((REPO / "engines/trtllm/eager.yaml").read_text())
    assert "cuda_graph_config" in eager and eager["cuda_graph_config"] is None
    kv8 = yaml.safe_load((REPO / "engines/trtllm/fp8kv.yaml").read_text())
    assert kv8["kv_cache_config"]["dtype"] == "fp8" and kv8["kv_cache_config"]["enable_block_reuse"] is True


# ------------------------------------------------------------------------------------------ hf baseline server
class FakeBackend:
    def __init__(self):
        self.last_completion_tokens = 0

    def prompt_tokens(self, messages):
        return sum(len(m.get("content") or "") for m in messages) // 4

    def stream(self, messages, max_tokens, temperature):
        pieces = ['{"thought": "", ', '"action": "flight_status", ', '"args": {"flight_no": "TN482"}}']
        self.last_completion_tokens = len(pieces)
        yield from pieces


@pytest.fixture
def hf_app():
    spec = importlib.util.spec_from_file_location("hf_naive_server", REPO / "engines" / "hf_naive_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hf_naive_server_endpoints(hf_app):
    from fastapi.testclient import TestClient

    c = TestClient(hf_app.app)
    assert c.get("/health").status_code == 503                        # still "loading"
    hf_app.app.state.backend = FakeBackend()
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").json()["data"][0]["id"] == "ttg-llm"
    assert c.post("/reset_prefix_cache").status_code == 200
    body = {"model": "ttg-llm", "messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 64}
    r = c.post("/v1/chat/completions", json=body).json()
    content = r["choices"][0]["message"]["content"]
    assert json.loads(content)["action"] == "flight_status"
    assert r["usage"]["prompt_tokens"] == 100 and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    s = c.post("/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}})
    events = [ln[6:] for ln in s.text.splitlines() if ln.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"])
    assert json.loads(text)["args"]["flight_no"] == "TN482"
    assert chunks[-1]["usage"]["completion_tokens"] == 3


def test_hf_naive_server_help_runs_without_torch():
    r = subprocess.run(["python3", "engines/hf_naive_server.py", "--help"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "--served-model-name" in r.stdout
