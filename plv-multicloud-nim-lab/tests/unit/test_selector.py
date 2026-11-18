"""The selector must reproduce Lab 02's physics numbers - if these fail, the model is wrong, not the GPU."""
import json
from pathlib import Path

import instance_selector as sel
import model_sizer

FIX = Path(__file__).resolve().parents[1] / "fixtures"
LLAMA = {"params_total_b": 8.03, "params_active_b": 8.03, "kv_layers": 32, "kv_heads": 8, "head_dim": 128,
         "mamba_state_mb_per_seq": 0, "moe_experts": None, "moe_active_experts": None}


def test_llama_kv_is_128_kib_per_token():
    assert sel.kv_bytes_per_token(LLAMA) == 131072


def test_a100_decode_floor_matches_lab02():
    gpu = sel.load("gpus.yaml")["A100-80GB-PCIe"]
    r = sel.evaluate(LLAMA, gpu, "bf16", 600, 200, 1000, 1000)
    # Lab 02: 16.06 GB / 1935 GB/s = 8.3 ms at 100% bandwidth; the selector assumes 80% + 2 ms overhead
    assert 11 < r["decode_floor_ms"] < 13


def test_l4_bf16_is_vetoed_for_a_40ms_itl_slo():
    gpu = sel.load("gpus.yaml")["L4"]
    r = sel.evaluate(LLAMA, gpu, "bf16", 600, 200, 40, 1000)
    assert r["max_batch_slo"] == 0 and "physics veto" in r["note"]
    r8 = sel.evaluate(LLAMA, gpu, "fp8", 600, 200, 40, 1000)
    assert r8["max_batch_slo"] > 0, "FP8 halves the bytes per token and rescues the L4"


def test_precision_support_is_enforced():
    gpus = sel.load("gpus.yaml")
    assert sel.evaluate(LLAMA, gpus["A100-80GB-PCIe"], "fp8", 600, 200, 40, 1000)["fits"] is False
    assert sel.evaluate(LLAMA, gpus["L40S"], "nvfp4", 600, 200, 40, 1000)["fits"] is False
    assert sel.evaluate(LLAMA, gpus["A100-80GB-PCIe"], "w4a16", 600, 200, 40, 1000)["fits"] is True


def test_verified_is_per_model():
    llama = sel.load("models.yaml")["llama-3.1-8b"]
    assert sel.verified_note(llama, "A100-80GB-SXM") == ""
    assert "not NIM-verified" in sel.verified_note(llama, "A100-80GB-PCIe"), "NIM 2.0.13 verifies A100 SXM4, not PCIe"
    assert "check" in sel.verified_note({"nim_verified_gpus": "unknown"}, "L4")


def test_moe_reads_more_weights_as_batch_grows():
    moe = dict(LLAMA, params_total_b=121, params_active_b=12, moe_experts=512, moe_active_experts=22)
    b1 = sel.weights_read_gb(moe, "nvfp4", 1)
    b64 = sel.weights_read_gb(moe, "nvfp4", 64)
    assert b1 < 8 and b64 > 50, "MoE decode is cheap at batch 1 but approaches the full model at high batch"


def test_ssm_state_counts_against_memory_and_bandwidth():
    hybrid = dict(LLAMA, mamba_state_mb_per_seq=170.0)
    gpu = sel.load("gpus.yaml")["A100-80GB-PCIe"]
    plain = sel.evaluate(LLAMA, gpu, "bf16", 600, 200, 1000, 1000)
    with_state = sel.evaluate(hybrid, gpu, "bf16", 600, 200, 1000, 1000)
    assert with_state["max_batch_mem"] < plain["max_batch_mem"] / 1.5


def test_rank_returns_a_cheapest_option_per_slo():
    rows = sel.rank("llama-3.1-8b", 600, 200, 40, 1000, ["bf16", "fp8", "nvfp4"])
    priced = [r for r in rows if r["usd_per_1m_tok"] is not None]
    assert priced and priced == sorted(priced, key=lambda r: r["usd_per_1m_tok"])


def test_model_sizer_dense_and_legacy_hybrid():
    dense = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
    s = model_sizer.size(dense)
    assert s["kv_layers"] == 32 and s["head_dim"] == 128 and s["kv_bytes_per_token_bf16"] == 131072
    hybrid = {"text_config": {"num_hidden_layers": 8, "hybrid_override_pattern": "M-M*-M-M*", "num_attention_heads": 32,
                              "num_key_value_heads": 2, "head_dim": 128, "mamba_num_heads": 64, "mamba_head_dim": 64,
                              "ssm_state_size": 128, "n_routed_experts": 128, "num_experts_per_tok": 6}}
    h = model_sizer.size(hybrid)
    assert h["kv_layers"] == 2 and h["mamba_layers"] == 4 and h["moe_experts"] == 128


def test_model_sizer_real_nemotron35_configs():
    """Real config.json files (Hugging Face, 2026-10-15) - the add-on's sizing is derived, not typed."""
    lt = model_sizer.size(json.loads((FIX / "nemotron-3.5-lightning-30b-a3b.config.json").read_text()))
    assert (lt["kv_layers"], lt["mamba_layers"], lt["moe_experts"], lt["moe_active_experts"]) == (6, 23, 128, 6)
    assert lt["kv_bytes_per_token_bf16"] == 6144 and 45 < lt["mamba_state_mb_per_seq"] < 55
    sv = model_sizer.size(json.loads((FIX / "nemotron-3.5-super-vl-120b-a12b.config.json").read_text()))
    assert (sv["kv_layers"], sv["mamba_layers"], sv["moe_experts"], sv["moe_active_experts"]) == (8, 40, 512, 22)
    assert sv["kv_bytes_per_token_bf16"] == 8192 and 165 < sv["mamba_state_mb_per_seq"] < 175
    assert sv["max_position_embeddings"] == 1048576
    bolt = model_sizer.size(json.loads((FIX / "nemotron-3.5-bolt-74b-a9b-24gb.config.trimmed.json").read_text()))
    assert bolt["kv_layers"] == 8 and bolt["mamba_layers"] == 40
    assert 300 < bolt["moe_experts"] < 400 and bolt["moe_active_experts"] == 9, "Puzzle varies experts per layer"
