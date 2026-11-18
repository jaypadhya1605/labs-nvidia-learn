"""M2 - archmath must reproduce the numbers model cards and Labs 01-04 already proved (no GPU, no torch)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archmath import archmath as am  # noqa: E402


def model(name):
    return am.analyze(am.load_config(name))


def close(value, expected, tol=0.03):
    return abs(value - expected) <= tol * expected


def test_bundled_configs_have_sources():
    names = am.bundled_names()
    assert {"llama-3.1-8b", "llama-3.1-70b", "qwen3-30b-a3b", "deepseek-v3", "mixtral-8x7b",
            "nemotron-3.5-lightning-30b-a3b", "qwen2.5-vl-7b", "llama-3.2-11b-vision"} <= set(names)
    for n in names:
        cfg = json.loads((ROOT / "archmath" / "configs" / f"{n}.json").read_text())
        assert cfg.get("_source"), f"{n} must say where its numbers come from"


def test_llama_8b_kv_and_params():
    m = model("llama-3.1-8b")
    assert m["kv_bytes_per_token_bf16"] == 128 * 1024           # 🔁 Lab 02: 2 x 32 x 8 x 128 x 2 B
    assert close(m["total_params"] / 1e9, 8.03)
    assert m["active_params"] == m["total_params"]


def test_llama_70b_kv_and_params():
    m = model("llama-3.1-70b")
    assert m["kv_bytes_per_token_bf16"] == 320 * 1024           # 🔁 mock round Q1
    assert close(m["total_params"] / 1e9, 70.6)


def test_deepseek_v3_mla_moe():
    m = model("deepseek-v3")
    assert close(m["total_params"] / 1e9, 671)
    assert close(m["active_params"] / 1e9, 37)
    assert m["kv_bytes_per_token_bf16"] == 61 * (512 + 64) * 2  # one latent + one RoPE key per layer
    assert m["layer_counts"] == {"mla": 61, "mlp": 3, "moe": 58}


def test_mixtral_moe():
    m = model("mixtral-8x7b")
    assert close(m["total_params"] / 1e9, 46.7)
    assert close(m["active_params"] / 1e9, 12.9)


def test_qwen3_moe():
    m = model("qwen3-30b-a3b")
    assert close(m["total_params"] / 1e9, 30.5)
    assert close(m["active_params"] / 1e9, 3.3)
    assert m["kv_bytes_per_token_bf16"] == 2 * 48 * 4 * 128 * 2


def test_nemotron_lightning_matches_lab04_sizer():
    m = model("nemotron-3.5-lightning-30b-a3b")
    assert m["layer_counts"] == {"attn": 6, "moe": 23, "mamba": 23}
    assert m["kv_bytes_per_token_bf16"] == 6 * 1024             # only 6 attention layers x 2 KV heads
    assert abs(m["ssm_state_bytes_per_seq"] / 1e6 - 49.1) < 0.1  # Lab 04 models.yaml: 49.1 MB per sequence
    assert close((m["total_params"] + m["mtp_params"]) / 1e9, 32.9, tol=0.01)   # BF16 checkpoint 65.83 GB / 2


def test_qwen25_vl_includes_vision_tower():
    m = model("qwen2.5-vl-7b")
    assert close(m["total_params"] / 1e9, 8.29, tol=0.01)
    assert 0.6e9 < m["vision_params"] < 0.75e9
    assert m["kv_bytes_per_token_bf16"] == 56 * 1024


def test_cross_attention_vlm_kv():
    m = model("llama-3.2-11b-vision")
    assert m["layer_counts"]["xattn"] == 8
    assert m["kv_bytes_per_token_bf16"] == 128 * 1024           # 32 self-attention layers
    assert m["xattn_kv_bytes_per_vision_token_bf16"] == 2 * 8 * 8 * 128 * 2


def test_expert_coverage_formula():
    assert am.expert_coverage(128, 8, 1) == pytest.approx(8.0)
    assert am.expert_coverage(128, 8, 10 ** 6) == pytest.approx(128.0)
    assert am.expert_coverage(8, 2, 8) == pytest.approx(8 * (1 - 0.75 ** 8))
    assert am.expert_coverage(0, 0, 32) == 0.0


def test_moe_decode_bytes_grow_with_batch_dense_does_not():
    moe, dense = model("qwen3-30b-a3b"), model("llama-3.1-8b")
    assert am.decode_weight_bytes(moe, "bf16", 32) > 5 * am.decode_weight_bytes(moe, "bf16", 1)
    assert am.decode_weight_bytes(dense, "bf16", 32) == am.decode_weight_bytes(dense, "bf16", 1)


def test_roofline_llama8b_h100_memory_bound():
    r = am.roofline(model("llama-3.1-8b"), "H100-SXM", "bf16", batch=1, ctx=4096)
    assert r["bound"] == "memory"
    assert 4.0 < r["step_ms"] < 5.5                               # ~15 GB / 3.35 TB/s
    assert r["intensity_flop_per_byte"] < 2                       # decode at batch 1: ~1 FLOP per byte


def test_roofline_mock_round_q1_tp2_fp8():
    r = am.roofline(model("llama-3.1-70b"), "H100-SXM", "fp8", batch=1, ctx=8192, gpus=2)
    assert 10.0 < r["step_ms"] < 11.5                             # mock round: ~10.5 ms per token on TP2
    assert r["fits"]


def test_nvfp4_bytes_include_block_scales():
    assert am.PRECISION_BYTES["nvfp4"] == pytest.approx(0.5 + 1 / 16)


def test_attention_share_grows_with_context():
    m = model("llama-3.1-8b")
    shares = [am.attention_share(m, c, "decode") for c in (8192, 32768, 131072)]
    assert shares == sorted(shares) and shares[0] < 0.3 and shares[-1] > 0.75
    hybrid = model("nemotron-3.5-lightning-30b-a3b")
    assert am.attention_share(hybrid, 131072, "decode") < am.attention_share(m, 131072, "decode")


def test_prefill_flops_close_to_2nl_for_short_prompts():
    m = model("llama-3.1-8b")
    f = am.prefill_flops(m, 3000)
    assert close(f, 2 * am.matmul_active(m) * 3000, tol=0.08)


@pytest.mark.parametrize("args", [[], ["--roofline", "--batch", "1,32"], ["--coverage"], ["--attention-share"],
                                  ["--prefill", "3000"], ["--notes", "--model", "llama-3.1-8b"]])
def test_cli_markdown(args):
    out = subprocess.run([sys.executable, "-m", "archmath.archmath", *args], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    assert out.startswith(("|", "Decode", "MoE", "Share", "Prefill", "Assumptions"))


def test_cli_json():
    out = subprocess.run([sys.executable, "-m", "archmath.archmath", "--json", "--roofline", "--model",
                          "llama-3.1-8b"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    assert data["summary"][0]["model"] == "llama-3.1-8b" and data["roofline"]


def test_unknown_model_is_an_error():
    p = subprocess.run([sys.executable, "-m", "archmath.archmath", "--model", "gpt-9"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode != 0 and "unknown model" in p.stderr


def test_external_config_path(tmp_path):
    cfg = json.loads((ROOT / "archmath" / "configs" / "llama-3.1-8b.json").read_text())
    cfg["_name"] = "my-llama"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    m = am.analyze(am.load_config(str(path)))
    assert m["name"] == "my-llama" and m["kv_bytes_per_token_bf16"] == 131072
    assert os.path.exists(path)
