"""M7 parallelism math: formulas, the worked examples in docs/05, nccl-tests parsing, scaling report, script dry runs.

Run: python3 -m pytest -q tests/test_parallel.py   (offline, no GPU, no torch)
Documented numbers are checked with a +-3% tolerance so docs/05 and the code cannot drift apart silently.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from archmath import archmath as am  # noqa: E402
from parallel import compare_nccl as cn  # noqa: E402
from parallel import parallelism_calc as pc  # noqa: E402
from parallel import scaling_report as sr  # noqa: E402

TOL = 0.03
SAMPLE = REPO / "parallel" / "samples" / "all_reduce_perf_format_example.txt"


@pytest.fixture(scope="module")
def m8():
    return am.analyze(am.load_config("llama-3.1-8b"))


@pytest.fixture(scope="module")
def m70():
    return am.analyze(am.load_config("llama-3.1-70b"))


@pytest.fixture(scope="module")
def ds():
    return am.analyze(am.load_config("deepseek-v3"))


# ------------------------------------------------------------------------------------------------ collectives
def test_ring_allreduce_moves_2_n_minus_1_over_n():
    assert pc.ring_allreduce_bytes(1e9, 8) == pytest.approx(1.75e9)
    assert pc.ring_allreduce_bytes(1e9, 2) == pytest.approx(1.0e9)
    assert pc.ring_allreduce_bytes(1e9, 1) == 0.0
    # approaches 2x the message for large n - the reason DP all-reduce cost is ~flat in GPU count
    assert pc.ring_allreduce_bytes(1e9, 1024) == pytest.approx(2e9, rel=0.002)


def test_allgather_and_reduce_scatter_move_n_minus_1_over_n():
    assert pc.allgather_bytes(8e9, 8) == pytest.approx(7e9)
    assert pc.allgather_bytes(8e9, 1) == 0.0


def test_alpha_beta_model():
    # 1 GB over NVLink 4 at 80% efficiency + one 10 us message
    t = pc.comm_time_s(1e9, "nvlink4", messages=1)
    assert t == pytest.approx(10e-6 + 1e9 / (450e9 * 0.8))
    # tiny messages are all latency
    assert pc.comm_time_s(16e3, "nvlink4", messages=100) == pytest.approx(100 * 10e-6, rel=0.01)
    assert pc.comm_time_s(0, "nvlink4", messages=0) == 0.0


def test_links_are_ordered_by_bandwidth():
    gbs = {k: v["gbs"] for k, v in pc.LINKS.items()}
    assert gbs["nvlink5"] > gbs["nvlink4"] > gbs["nvlink3"] > gbs["pcie5"] > gbs["pcie4"]
    assert gbs["nvlink4"] / gbs["ib-ndr"] == pytest.approx(9.0)        # NVLink 4 vs one 400G NIC per GPU


# ------------------------------------------------------------------------------------------------ memory
@pytest.mark.parametrize("zero,bytes_per_param", [(0, 16.0), (1, 2 + 2 + 12 / 8), (2, 2 + 2 / 8 + 12 / 8), (3, 16 / 8)])
def test_zero_stages_on_8_gpus(zero, bytes_per_param):
    assert pc.model_state_bytes(8e9, dp=8, zero=zero) == pytest.approx(bytes_per_param * 8e9)


def test_tp_pp_divide_the_model_states():
    assert pc.model_state_bytes(70e9, tp=8, pp=4) == pytest.approx(16 * 70e9 / 32)


def test_moe_experts_shard_over_ep_and_zero_over_dp_over_ep():
    # 10B dense + 90B experts, EP 8 inside DP 8, ZeRO-1: experts / 8, optimizer of experts over dp/ep = 1
    total, experts = 100e9, 90e9
    got = pc.model_state_bytes(total, dp=8, zero=1, expert_params=experts, ep=8)
    dense = 10e9 * (2 + 2 + 12 / 8)
    exp = (90e9 / 8) * (2 + 2 + 12)
    assert got == pytest.approx(dense + exp)


def test_lora_params_llama_8b_rank_16(m8):
    # r x (in + out) for q,k,v,o,gate,up,down x 32 layers = 41,943,040 (PEFT prints the same number)
    assert pc.lora_params(m8, 16) == 41_943_040
    assert pc.lora_params(m8, 16, targets=("q", "v")) == 32 * 16 * ((4096 + 4096) + (4096 + 1024))


def test_lora_states_frozen_base_is_2_bytes_per_param(m8):
    lp = pc.lora_params(m8, 16)
    got = pc.model_state_bytes(m8["total_params"], lora_params=lp)
    assert got == pytest.approx(2 * m8["total_params"] + 16 * lp)


def test_korthikanti_activations(m8):
    s, b, h, L, a = 2048, 1, 4096, 32, 32
    sbh = s * b * h
    flash = pc.activation_bytes(s, b, h, L, a, flash=True)
    assert flash == pytest.approx(sbh * 34 * L)
    eager = pc.activation_bytes(s, b, h, L, a, flash=False)
    assert eager == pytest.approx(sbh * (34 + 5 * a * s / h) * L)
    # TP 8 with sequence parallel divides everything by t
    assert pc.activation_bytes(s, b, h, L, a, tp=8, sp=True) == pytest.approx(flash / 8)
    # TP 8 without SP: 10 + 24/t (LayerNorm + dropout inputs stay replicated)
    assert pc.activation_bytes(s, b, h, L, a, tp=8, sp=False) == pytest.approx(sbh * (10 + 24 / 8) * L)
    # full recompute keeps only the layer inputs (2sbh per layer) + one layer being rematerialized
    full = pc.activation_bytes(s, b, h, L, a, recompute="full")
    assert full == pytest.approx(2 * sbh * L + sbh * 34)
    # 1F1B: the first stage holds min(p, m) micro-batches of its L/p layers
    assert pc.activation_bytes(s, b, h, L, a, pp=4, microbatches=32) == pytest.approx(sbh * 34 * (L / 4) * 4)


# ------------------------------------------------------------------------------------------------ time
def test_pipeline_bubble():
    assert pc.pipeline_bubble(1, 1) == 0.0
    assert pc.pipeline_bubble(4, 32) == pytest.approx(3 / 35)
    assert pc.pipeline_bubble(8, 8) == pytest.approx(7 / 15)


def test_6nt_and_lora_4nt():
    assert pc.train_flops(8e9, 1e9) == pytest.approx(48e18)
    assert pc.train_flops(8e9, 1e9, lora=True) == pytest.approx(32e18)
    # 8B on 15T tokens, 1,024 H100 at 40% MFU
    days = pc.train_time_s(8e9, 15e12, 1024, 989, 0.4) / 86400
    assert days == pytest.approx(6 * 8e9 * 15e12 / (1024 * 989e12 * 0.4) / 86400)


def test_ep_alltoall_and_cp_ring_bytes(ds):
    one_way = 128 * 8 * 7168 * (7 / 8)
    assert pc.ep_alltoall_bytes(128, 7168, 8, 8, 58, 1, 2, backward=False) == pytest.approx(one_way * 3 * 58)
    assert pc.ep_alltoall_bytes(128, 7168, 8, 1, 58) == 0.0
    chunk = (8192 / 2) * 1 * 2 * 8 * 128 * 2
    assert pc.cp_ring_bytes(8192, 1, 2, 8, 128, 32, backward=False) == pytest.approx(32 * chunk)
    assert pc.cp_ring_bytes(8192, 1, 1, 8, 128, 32) == 0.0


def test_ep_must_divide_dp_times_cp():
    q3 = am.analyze(am.load_config("qwen3-30b-a3b"))
    with pytest.raises(ValueError, match="must divide"):
        pc.train_plan(q3, dp=6, ep=4)


def test_expert_parallel_shrinks_moe_states():
    q3 = am.analyze(am.load_config("qwen3-30b-a3b"))
    with_ep = pc.train_plan(q3, dp=8, ep=8, cp=2, zero=1, seq=4096)
    without = pc.train_plan(q3, dp=8, ep=1, cp=2, zero=1, seq=4096)
    assert with_ep["model_state_gb"] < 0.5 * without["model_state_gb"]
    assert with_ep["fits"] and not without["fits"]
    assert with_ep["comm"]["ep"]["bytes"] > 0 and with_ep["comm"]["cp"]["bytes"] > 0


def test_fsdp_with_lora_still_gathers_the_frozen_base(m8):
    ddp = pc.train_plan(m8, dp=8, zero=0, lora_rank=16, seq=2048, mbs=4, mfu=0.35)
    fsdp = pc.train_plan(m8, dp=8, zero=3, lora_rank=16, seq=2048, mbs=4, mfu=0.35)
    assert fsdp["model_state_gb"] < ddp["model_state_gb"] / 5            # 16.7 GB -> 2.7 GB per GPU
    assert fsdp["comm"]["dp"]["bytes"] >= 2 * (2 * m8["total_params"]) * 7 / 8
    assert ddp["comm"]["dp"]["s"] * 1e3 < 1.0                             # LoRA gradients: ~84 MB, sub-ms on NVLink
    assert fsdp["comm"]["dp"]["s"] * 1e3 == pytest.approx(80, rel=TOL)


# ------------------------------------------------------------------------------------------------ docs/05 numbers
def test_e1_lora_on_one_h100(m8):
    none = pc.train_plan(m8, lora_rank=16, seq=2048, mbs=4, recompute="none", tokens=1.5e6, mfu=0.35)
    full = pc.train_plan(m8, lora_rank=16, seq=2048, mbs=4, recompute="full", tokens=1.5e6, mfu=0.35)
    assert none["model_state_gb"] == pytest.approx(16.7, rel=TOL)
    assert none["total_gb"] == pytest.approx(53.2, rel=TOL)
    assert full["total_gb"] == pytest.approx(20.0, rel=TOL)
    assert none["train_hours"] * 60 == pytest.approx(2.3, rel=TOL)
    assert none["fits"] and full["fits"]


def test_e2_full_finetune_8b_needs_sharding(m8):
    plans = {z: pc.train_plan(m8, dp=8, zero=z, seq=4096, mbs=2, recompute="full", mfu=0.4) for z in range(4)}
    assert not plans[0]["fits"]                                   # 128 GB of states on an 80 GB card
    assert all(plans[z]["fits"] for z in (1, 2, 3))
    assert plans[3]["model_state_gb"] == pytest.approx(16.1, rel=TOL)
    # ZeRO-3 / FSDP moves 1.5x the bytes of an all-reduce (2 all-gathers + 1 reduce-scatter vs RS + AG)
    assert plans[3]["comm"]["dp"]["bytes"] == pytest.approx(1.5 * plans[2]["comm"]["dp"]["bytes"], rel=0.001)
    pcie = pc.train_plan(m8, dp=8, zero=3, seq=4096, mbs=2, recompute="full", mfu=0.4, dp_link="pcie4")
    assert pcie["comm"]["dp"]["s"] > pcie["compute_s_per_step"]   # PCIe makes FSDP communication-bound


def test_e3_70b_pretraining_plan(m70):
    p = pc.train_plan(m70, dp=64, tp=8, pp=4, zero=1, seq=8192, mbs=1, microbatches=32, recompute="selective",
                      tokens=15e12, mfu=0.4, tp_link="nvlink4", dp_link="ib-ndr", pp_link="ib-ndr")
    assert p["gpus"] == 2048
    assert p["fits"]
    assert p["bubble"] == pytest.approx(3 / 35)
    assert p["total_gb"] == pytest.approx(32.0, rel=TOL)
    assert p["gpu_hours"] / 1e6 == pytest.approx(4.88, rel=TOL)   # Meta reports 7.0M H100-hours for 3.1 70B
    assert p["train_days"] == pytest.approx(99, rel=TOL)
    assert p["comm"]["tp"]["s"] > p["comm"]["dp"]["s"]           # TP traffic is why TP stays inside NVLink


def test_e4_tp_decode_latency_floor(m70):
    tp2 = pc.tp_decode(m70, "fp8", 2, 1, 8192, "nvlink4")
    tp8 = pc.tp_decode(m70, "fp8", 8, 1, 8192, "nvlink4")
    assert tp2["roofline_ms"] == pytest.approx(10.93, rel=TOL)   # 🔁 mock round Q1 answer (10.0-11.5 ms)
    assert tp2["step_ms"] == pytest.approx(12.54, rel=TOL)
    assert tp8["step_ms"] == pytest.approx(4.35, rel=TOL)
    # 4x the GPUs buys 2.9x, not 4x: 160 all-reduces x alpha do not shrink with TP
    assert 2.5 < tp2["step_ms"] / tp8["step_ms"] < 3.2
    assert tp8["allreduce_ms"] / tp8["step_ms"] > 0.3
    assert pc.tp_decode(m70, "fp8", 8, 1, 8192, "pcie4")["allreduce_ms"] > 2 * tp8["allreduce_ms"]
    assert not pc.tp_decode(m70, "fp8", 1, 1, 8192, "nvlink4")["fits"]
    assert pc.tp_decode(m70, "fp8", 1, 1, 8192)["allreduce_ms"] == 0.0


def test_e5_moe_alltoall_nvl72_vs_infiniband(ds):
    def t(ep, link):
        nb = pc.ep_alltoall_bytes(128, ds["hidden"], ds["experts_per_token"], ep, ds["moe_layers"],
                                  dispatch_bytes=1, combine_bytes=2, backward=False)
        return nb, pc.comm_time_s(nb, link, 2 * ds["moe_layers"])
    nb64, t_ib = t(64, "ib-ndr")
    nb72, t_nvl = t(72, "nvlink5")
    assert nb64 / 1e9 == pytest.approx(1.26, rel=TOL)
    assert t_ib * 1e3 == pytest.approx(34.3, rel=TOL)
    assert t_nvl * 1e3 == pytest.approx(2.9, rel=TOL)
    assert t_ib / t_nvl > 10


def test_examples_render_all_five(capsys):
    assert pc.main(["--examples"]) == 0
    out = capsys.readouterr().out
    for tag in ("### E1", "### E2", "### E3", "### E4", "### E5"):
        assert tag in out
    assert "41,943,040" in out


def test_cli_train_infer_links(capsys):
    assert pc.main(["--links"]) == 0
    assert "nvlink4" in capsys.readouterr().out
    assert pc.main(["--train", "--model", "qwen3-30b-a3b", "--dp", "8", "--ep", "8", "--cp", "2", "--zero", "1"]) == 0
    out = capsys.readouterr().out
    assert "EP all-to-all" in out and "CP ring" in out
    assert pc.main(["--infer", "--model", "llama-3.1-70b", "--precision", "fp8", "--tp", "2,8", "--batch", "1",
                    "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["tp"] for r in data] == [2, 8]


# ------------------------------------------------------------------------------------------------ nccl-tests parsing
def test_parse_format_example():
    parsed = cn.parse(SAMPLE.read_text(), SAMPLE.name)
    assert parsed["ngpus"] == 8
    assert parsed["collective"] == "all_reduce"
    assert parsed["rows"][0]["size"] == 8 and parsed["rows"][-1]["size"] == 4 << 30
    s = cn.summarize(parsed, "nvlink4")
    assert s["peak_busbw_gbs"] == pytest.approx(359.7, rel=0.01)
    assert s["efficiency"] == pytest.approx(0.80, abs=0.01)
    assert s["small_message_us"] == pytest.approx(18.0, rel=0.05)
    assert s["bandwidth_regime_from_bytes"] == 32 << 20
    assert s["busbw_consistent"] is True


def test_parse_handles_both_column_layouts():
    old = ("# nThread 1 nGpus 2 minBytes 8 maxBytes 1024 step: 2(factor) warmup iters: 5 iters: 20\n"
           "#  Rank  0 Group  0 Pid  1 on host device  0 [0x1b] NVIDIA H100 80GB HBM3\n"
           "#  Rank  1 Group  0 Pid  1 on host device  1 [0x2c] NVIDIA H100 80GB HBM3\n"
           "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong\n"
           "        1024           256     float     sum      -1    10.00    0.10    0.10      0    10.00    0.10    0.10      0\n"
           "     1048576        262144     float     sum      -1    20.00   52.43   52.43      0    20.00   52.43   52.43      0\n"
           "# Avg bus bandwidth    : 26.27\n")
    parsed = cn.parse("all_reduce_perf\n" + old)
    assert parsed["ngpus"] == 2 and len(parsed["rows"]) == 2 and parsed["avg_busbw"] == pytest.approx(26.27)
    s = cn.summarize(parsed, "nvlink4")
    assert s["bus_factor"] == pytest.approx(1.0)       # 2(n-1)/n = 1 for n = 2
    with pytest.raises(ValueError):
        cn.summarize(cn.parse("no rows here"), "nvlink4")


def test_compare_nccl_expands_directories(tmp_path):
    run = tmp_path / "node-8gpu-20261017T120000Z"
    run.mkdir()
    for name in ("alltoall_perf.log", "all_reduce_perf.log"):
        (run / name).write_text(SAMPLE.read_text())
    files = cn.expand([str(tmp_path)])
    assert [os.path.basename(f) for f in files] == ["all_reduce_perf.log", "alltoall_perf.log"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no \\*_perf.log"):
        cn.expand([str(empty)])


def test_tools_run_as_plain_files():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script, args in (("parallel/compare_nccl.py", [str(SAMPLE)]), ("parallel/parallelism_calc.py", ["--links"]),
                         ("parallel/scaling_report.py", ["--help"])):
        out = subprocess.run([sys.executable, script, *args], cwd=REPO, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-500:]


def test_compare_nccl_cli(capsys):
    assert cn.main([str(SAMPLE), "--link", "nvlink4"]) == 0
    out = capsys.readouterr().out
    assert "EFFICIENCY = 0.80" in out and "alpha ~ 18 us" in out
    assert cn.main([str(SAMPLE), str(SAMPLE), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


# ------------------------------------------------------------------------------------------------ scaling report
def test_scaling_report(tmp_path, capsys):
    def write(name, sps, runtime, mib):
        (tmp_path / f"{name}.log").write_text(
            "step 10 loss 1.2\n"
            f"{{'train_runtime': {runtime}, 'train_samples_per_second': {sps}, 'train_steps_per_second': 0.7, "
            "'train_loss': 0.41, 'epoch': 1.0}\n"
            "***** train metrics *****\n"
            f"  train_runtime            = 0:01:{runtime - 60:05.2f}\n"
            f"  train_samples_per_second = {sps}\n")
        (tmp_path / f"{name}.mem.csv").write_text("".join(f"2026/10/17 10:00:0{i}, {i}, {mib}\n" for i in range(2)))
    write("single-1gpu", 10.0, 80.0, 30000)
    write("ddp-4gpu", 36.0, 82.0, 30500)
    write("fsdp2-4gpu", 34.0, 85.0, 21000)
    runs = {r["run"]: r for r in sr.report(str(tmp_path))}
    assert runs["single-1gpu"]["train_runtime"] == pytest.approx(80.0)      # H:MM:SS parsed back to seconds
    assert runs["ddp-4gpu"]["speedup"] == pytest.approx(3.6)
    assert runs["ddp-4gpu"]["efficiency"] == pytest.approx(0.9)
    assert runs["fsdp2-4gpu"]["peak_mem_gib"] == pytest.approx(21000 / 1024)
    assert sr.main([str(tmp_path)]) == 0
    assert "| ddp-4gpu | 4 | ddp | 36.00 | 9.00 | 3.60x | 90% |" in capsys.readouterr().out
    assert sr.to_number("1:02:03.5") == pytest.approx(3723.5)
    # a parent directory resolves to its newest run (train_scaling.sh writes results/scaling/<timestamp>/)
    parent = tmp_path.parent / (tmp_path.name + "-parent")
    run = parent / "20261017T120000Z"
    run.mkdir(parents=True)
    for f in tmp_path.iterdir():
        (run / f.name).write_text(f.read_text())
    assert sr.run_dir(str(parent)).rstrip("/") == str(run)
    assert {r["run"] for r in sr.report(str(parent))} == {"single-1gpu", "ddp-4gpu", "fsdp2-4gpu"}


# ------------------------------------------------------------------------------------------------ script dry runs
BASH = shutil.which("bash")


def dry(*args, env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run([BASH, *args, "--dry-run"], cwd=REPO, env=e, capture_output=True, text=True,
                          check=True).stdout


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_nccl_tests_dry_run():
    out = dry("parallel/nccl_tests.sh", "8", "--p2p-off")
    assert "NCCL_P2P_DISABLE=1" in out and "NCCL_DEBUG=INFO" in out
    assert "-b 8 -e 4G -f 2 -g" in out and "nvidia-smi topo -m" in out and "nvidia-smi nvlink -s" in out
    assert "nvcr.io/nvidia/pytorch:" in out
    assert "NCCL_NVLS_ENABLE=0" in dry("parallel/nccl_tests.sh", "4", "--nvls-off")


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_sweep_parallel_dry_run():
    out = dry("parallel/sweep_parallel.sh", "all")
    for flag in ("--tensor-parallel-size 8", "--pipeline-parallel-size 2", "--data-parallel-size 2",
                 "--served-model-name ttg-llm", "--enable-prefix-caching", "nvidia/Llama-3.1-70B-Instruct-FP8"):
        assert flag in out
    assert out.count("python3 -m ttg.bench.agent_bench --label vllm-") == 8


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_train_scaling_dry_run():
    out = dry("parallel/train_scaling.sh", "1", "2", env={"TRAIN_ARGS": "--max-steps 60"})
    assert "python3 finetune/train_lora.py --max-steps 60" in out
    assert "accelerate launch --multi_gpu --num_processes 2 finetune/train_lora.py" in out
    assert "accelerate launch --config_file finetune/accelerate_fsdp2.yaml --num_processes 2 finetune/train_lora.py" in out
