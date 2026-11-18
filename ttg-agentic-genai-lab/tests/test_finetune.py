"""finetune/ (FORK B): SFT data, train_lora --dry-run (no torch), tool-call scoring, accelerate configs, serving.

The SFT target for every step must be the oracle's step in canonical JSON - byte-identical to the mock engine -
so the data, the runtime agent and the bench all speak exactly the same protocol.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ttg import world  # noqa: E402
from ttg.agent.prompts import build_system_prompt  # noqa: E402
from ttg.agent.protocol import parse_step  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ttg_finetune_{name}", REPO / "finetune" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sft = _load("make_sft_data")
ev = _load("eval_toolcalls")
tl = _load("train_lora")


@pytest.fixture(scope="module")
def rows():
    return sft.build(25, 11, "short")


# ---------------------------------------------------------------------------------------------- SFT data
def test_one_example_per_plan_step(rows):
    scs = world.make_scenarios(25, seed=11)
    assert len(rows) == sum(len(world.expected_plan(s)) for s in scs)
    assert sum(1 for r in rows if r["meta"]["kind"] == "final") == 25


def test_canonical_conversation_shape(rows):
    short = build_system_prompt(variant="short")
    for r in rows:
        p = r["prompt"]
        assert p[0] == {"role": "system", "content": short}
        assert p[1]["role"] == "user" and p[1]["content"].startswith("Disrupted traveler: ")
        assert [m["role"] for m in p[2:]] == ["assistant", "user"] * ((len(p) - 2) // 2)
        assert all(m["content"].startswith("OBSERVATION ") for m in p[3::2])
        assert len(r["completion"]) == 1 and r["completion"][0]["role"] == "assistant"
        assert r["meta"]["step"] == (len(p) - 2) // 2 + 1


def test_targets_are_the_oracle_steps(rows):
    by_id = {s["scenario_id"]: s for s in world.make_scenarios(25, seed=11)}
    for r in rows:
        plan = world.expected_plan(by_id[r["meta"]["scenario_id"]])
        want = plan[r["meta"]["step"] - 1]
        step = parse_step(r["completion"][0]["content"])
        if "final" in want:
            assert step.kind == "final" and step.final == want["final"]
        else:
            assert step.kind == "action" and step.action == want["action"] and step.args == want["args"]


def test_targets_match_the_mock_engine_byte_for_byte(rows):
    from ttg.mock.fake_llm import agent_reply

    for r in rows:
        assert agent_reply(r["prompt"], "ttg-llm", set()) == r["completion"][0]["content"]


def test_short_prompt_is_the_point():
    short, full = sft.build(10, 11, "short"), sft.build(10, 11, "full")
    assert len(build_system_prompt(variant="short")) == 560 and len(build_system_prompt(variant="full")) == 7360
    ds, df = sft.describe(short), sft.describe(full)
    assert ds["examples"] == df["examples"]
    assert ds["approx_train_tokens"] < df["approx_train_tokens"] / 2          # baking the prompt halves (+) the cost
    assert [r["completion"] for r in short] == [r["completion"] for r in full]


def test_train_and_eval_never_overlap_and_never_use_the_golden_set():
    train = {s["scenario_id"] for s in world.make_scenarios(50, seed=sft.TRAIN_SEED)}
    evals = {s["scenario_id"] for s in world.make_scenarios(50, seed=sft.EVAL_SEED)}
    golden = {s["scenario_id"] for s in world.make_scenarios(50, seed=7)}
    assert not (train & evals) and not (train & golden) and not (evals & golden)
    with pytest.raises(SystemExit):
        sft.main(["--train-seed", "7", "--out", "/nonexistent"])


def test_cli_writes_files_and_stats(tmp_path):
    assert sft.main(["--n-train", "6", "--n-eval", "3", "--out", str(tmp_path)]) == 0
    tr = (tmp_path / "train.jsonl").read_text().splitlines()
    ev_lines = (tmp_path / "eval.jsonl").read_text().splitlines()
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["variant"] == "short" and stats["train"]["examples"] == len(tr) and stats["eval"]["examples"] == len(ev_lines)
    assert stats["train"]["tasks"] == 6 and stats["eval"]["seed"] == 13 and stats["teacher"] is None
    assert json.loads(tr[0])["meta"]["step"] == 1


def test_teacher_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        sft.main(["--teacher", "--teacher-model", "some/model", "--out", str(tmp_path)])


def test_rejection_sampling_rule(rows):
    act = next(r for r in rows if r["meta"]["action"] == "issue_voucher")
    fin = next(r for r in rows if r["meta"]["kind"] == "final")
    sc = next(s for s in world.make_scenarios(25, seed=11) if s["scenario_id"] == act["meta"]["scenario_id"])
    step = world.expected_plan(sc)[act["meta"]["step"] - 1]
    text = act["completion"][0]["content"]
    assert sft.accept_step(step, text)
    assert sft.accept_step(step, text.replace('"thought": "', '"thought": "My own words. '))   # wording is free
    bad = json.loads(text)
    bad["args"]["amount_usd"] = bad["args"]["amount_usd"] * 2
    assert not sft.accept_step(step, json.dumps(bad))
    assert not sft.accept_step(step, "Sure, I will issue the voucher now.")
    fsc = next(s for s in world.make_scenarios(25, seed=11) if s["scenario_id"] == fin["meta"]["scenario_id"])
    fstep = world.expected_plan(fsc)[-1]
    f = json.loads(fin["completion"][0]["content"])
    f["final"]["message"] = "A different, still friendly message."
    assert sft.accept_step(fstep, json.dumps(f))
    f["final"]["needs_human_approval"] = not f["final"]["needs_human_approval"]
    assert not sft.accept_step(fstep, json.dumps(f))


def test_teacher_examples_keep_oracle_history(rows):
    sc = world.make_scenarios(1, seed=11)[0]
    calls = []

    def teacher(_s, history):
        calls.append(len(history))
        return "not json"                                   # always rejected -> oracle text is used

    from collections import Counter
    stats = Counter()
    out = sft.examples_for(sc, build_system_prompt(variant="short"), teacher, stats)
    assert [r["completion"] for r in out] == [r["completion"] for r in rows[: len(out)]]
    assert stats["teacher_rejected"] == len(out) and calls == sorted(calls)


# ---------------------------------------------------------------------------------------------- train_lora
def _dry_run(tmp_path, *args, **env):
    """Run --dry-run with fake torch/trl/transformers/peft/datasets that explode on import."""
    fake = tmp_path / "fakelibs"
    fake.mkdir(exist_ok=True)
    for lib in ("torch", "trl", "transformers", "peft", "datasets", "accelerate"):
        (fake / f"{lib}.py").write_text(f"raise ImportError('{lib} must not be imported by --dry-run')\n")
    full = {**os.environ, "PYTHONPATH": f"{fake}{os.pathsep}{REPO}", **env}
    for k in ("WORLD_SIZE", "ACCELERATE_USE_FSDP"):
        if k not in env:
            full.pop(k, None)
    r = subprocess.run([sys.executable, "finetune/train_lora.py", "--dry-run", *args], cwd=REPO, env=full,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    sft.main(["--n-train", "40", "--n-eval", "10", "--out", str(d)])
    return d


def test_dry_run_plan_without_torch(tmp_path, data_dir):
    d = _dry_run(tmp_path, "--data-dir", str(data_dir))
    p = d["plan"]
    n = len((data_dir / "train.jsonl").read_text().splitlines())
    assert p["n_train"] == n and p["effective_batch"] == 16 and p["total_steps"] == math.ceil(n / 16)
    assert p["lora"]["trainable_params"] == 41_943_040 and p["lora"]["trainable_pct"] == pytest.approx(0.522, abs=1e-3)
    assert p["max_length"] == 4096 and p["data_variant"] == "short"
    assert p["memory_gb_per_gpu"]["base_weights_bf16"] == pytest.approx(16.06, abs=0.01)
    c = d["sft_config"]
    assert c["model_init_kwargs"] == {"dtype": "bfloat16"}            # TRL would load float32 otherwise
    assert c["completion_only_loss"] is True and c["max_length"] == 4096 and c["warmup_steps"] == 0.03
    assert c["learning_rate"] == 2e-4 and c["bf16"] is True and c["save_strategy"] == "no"
    lc = d["lora_config"]
    assert (lc["r"], lc["lora_alpha"], lc["lora_dropout"]) == (16, 32, 0.05)
    assert lc["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert d["run_name"] == "lora-r16-short-1gpu"


def test_dry_run_distributed(tmp_path, data_dir):
    ddp = _dry_run(tmp_path, "--data-dir", str(data_dir), WORLD_SIZE="4")
    assert ddp["plan"]["effective_batch"] == 64 and ddp["run_name"] == "lora-r16-short-ddp-4gpu"
    fsdp = _dry_run(tmp_path, "--data-dir", str(data_dir), WORLD_SIZE="2", ACCELERATE_USE_FSDP="true")
    assert fsdp["plan"]["fsdp"] and fsdp["plan"]["memory_gb_per_gpu"]["base_weights_bf16"] == pytest.approx(8.03, abs=0.01)
    assert fsdp["run_name"] == "lora-r16-short-fsdp2-2gpu"


def test_full_prompt_data_gets_a_longer_max_length(tmp_path):
    d = tmp_path / "full"
    sft.main(["--variant", "full", "--n-train", "5", "--n-eval", "2", "--out", str(d)])
    assert _dry_run(tmp_path, "--data-dir", str(d))["plan"]["max_length"] == 8192


def test_lora_param_math():
    cfg = json.loads((REPO / "archmath/configs/llama-3.1-8b.json").read_text())
    assert tl.lora_trainable_params(cfg, 16, ["q_proj", "v_proj"]) == 32 * 16 * ((4096 + 4096) + (4096 + 1024))
    assert tl.total_params(cfg) == 8_030_261_248
    with pytest.raises(ValueError):
        tl.lora_trainable_params(cfg, 16, ["lm_head"])


def test_no_data_message(tmp_path):
    d = _dry_run(tmp_path, "--data-dir", str(tmp_path / "empty"))
    assert d["plan"]["n_train"] == 0


# ----------------------------------------------------------------------------------------- tool-call scoring
def test_scoring_perfect_and_wrong(rows):
    perfect = [{"score": ev.score_prediction(r["completion"][0]["content"], r["completion"][0]["content"]),
                "usage": {"prompt_tokens": 100, "completion_tokens": 40}, "latency_s": 0.1} for r in rows]
    s = ev.aggregate(perfect)
    assert s["step_exact_pct"] == 100.0 and s["invalid_json_pct"] == 0.0 and s["final_exact_pct"] == 100.0
    assert s["mean_prompt_tokens"] == 100 and s["n_final_steps"] == 25

    act = next(r for r in rows if r["meta"]["action"] == "issue_voucher")["completion"][0]["content"]
    wrong = json.loads(act)
    wrong["args"]["amount_usd"] += 15
    x = ev.score_prediction(act, json.dumps(wrong))
    assert x["action_ok"] and not x["args_ok"] and not x["exact"]
    other = ev.score_prediction(act, json.dumps({"thought": "", "action": "policy_search", "args": {"query": "q"}}))
    assert not other["action_ok"]
    junk = ev.score_prediction(act, "I think we should issue a voucher.")
    assert not junk["valid"] and junk["error"]

    fin = next(r for r in rows if r["meta"]["kind"] == "final")["completion"][0]["content"]
    f = json.loads(fin)
    f["final"]["hotel_id"] = "H-XXX-9"
    y = ev.score_prediction(fin, json.dumps(f))
    assert y["action_ok"] and not y["exact"] and y["fields"]["hotel_id"] is False and y["fields"]["vouchers"] is True
    early = ev.score_prediction(fin, act)                       # an action when the plan is already done
    assert not early["action_ok"] and not any(early["fields"].values())
    agg = ev.aggregate([{"score": z} for z in (x, other, junk, y, early)])
    assert agg["invalid_json_pct"] == 20.0 and agg["final_field_pct"]["vouchers"] == 50.0


def test_system_prompt_swap(rows):
    p = rows[3]["prompt"]
    full = ev.with_system_prompt(p, "full")
    assert full[0]["content"] == build_system_prompt(variant="full") and full[1:] == p[1:]
    assert ev.with_system_prompt(p, "asis") is p


def test_score_only_cli(tmp_path, rows):
    preds = tmp_path / "eval-demo.jsonl"
    preds.write_text("".join(json.dumps({"meta": r["meta"], "expected": r["completion"][0]["content"],
                                         "text": r["completion"][0]["content"], "usage": {"prompt_tokens": 600},
                                         "latency_s": 0.05}) + "\n" for r in rows))
    out = tmp_path / "summary.json"
    assert ev.main(["--score-only", str(preds), "--out", str(out)]) == 0
    s = json.loads(out.read_text())
    assert s["label"] == "demo" and s["step_exact_pct"] == 100.0 and s["mean_prompt_tokens"] == 600
    assert ev.main(["--score-only", str(preds), "--out", str(out), "--variant", "full"]) == 0     # README spelling
    assert json.loads(out.read_text())["system_variant"] == "full"


def test_live_eval_against_the_mock(mock_engine, tmp_path, data_dir):
    good = tmp_path / "good.json"
    assert ev.main(["--base-url", f"{mock_engine}/v1", "--model", "ttg-short", "--data", str(data_dir / "eval.jsonl"),
                    "--limit", "40", "--out", str(good)]) == 0
    g = json.loads(good.read_text())
    n_eval = len((data_dir / "eval.jsonl").read_text().splitlines())
    assert g["n"] == min(40, n_eval) and g["step_exact_pct"] == 100.0 and g["mean_prompt_tokens"] > 0
    assert "issue_voucher" in g["per_tool_args_exact_pct"]
    assert (tmp_path / "good.jsonl").exists()
    bad = tmp_path / "bad.json"
    ev.main(["--base-url", f"{mock_engine}/v1", "--model", "mock-wrong", "--data", str(data_dir / "eval.jsonl"),
             "--system-variant", "full", "--limit", "40", "--out", str(bad)])
    b = json.loads(bad.read_text())
    assert b["per_tool_args_exact_pct"].get("issue_voucher") == 0.0 and b["step_exact_pct"] < 100
    assert b["mean_prompt_tokens"] > g["mean_prompt_tokens"] * 2          # the full prompt costs ~4x the tokens


# -------------------------------------------------------------------------------------- accelerate + serving
ACCELERATE_KEYS = {"compute_environment", "distributed_type", "mixed_precision", "use_cpu", "debug",
                   "num_processes", "machine_rank", "num_machines", "gpu_ids", "main_process_ip",
                   "main_process_port", "rdzv_backend", "same_network", "main_training_function",
                   "enable_cpu_affinity", "fsdp_config", "deepspeed_config"}


def test_accelerate_configs():
    ddp = yaml.safe_load((REPO / "finetune/accelerate_ddp.yaml").read_text())
    fsdp = yaml.safe_load((REPO / "finetune/accelerate_fsdp2.yaml").read_text())
    for c in (ddp, fsdp):
        assert set(c) <= ACCELERATE_KEYS and c["mixed_precision"] == "bf16" and c["num_processes"] >= 2
    assert ddp["distributed_type"] == "MULTI_GPU" and "fsdp_config" not in ddp
    assert fsdp["distributed_type"] == "FSDP"
    assert fsdp["fsdp_config"] == {"fsdp_version": 2, "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                                   "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                                   "fsdp_reshard_after_forward": True, "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                                   "fsdp_cpu_ram_efficient_loading": True, "fsdp_activation_checkpointing": False}


def test_train_lora_saves_a_full_state_dict_under_fsdp():
    src = (REPO / "finetune/train_lora.py").read_text()
    assert 'set_state_dict_type("FULL_STATE_DICT")' in src and "trainer.save_model(a.out)" in src


def _serve(*args, **env):
    full = {**os.environ, "DRY_RUN": "1", **env}
    full.pop("EXTRA_ARGS", None)
    return subprocess.run(["bash", "finetune/serve_lora.sh", *args], cwd=REPO, env=full, capture_output=True,
                          text=True, timeout=60)


def test_serve_lora_dry_run(tmp_path):
    r = _serve("vllm")
    assert r.returncode == 0, r.stderr
    assert "--lora-modules ttg-short=/adapters/ttg-short" in r.stdout and "--max-lora-rank 16" in r.stdout
    assert "eval_toolcalls.py --model ttg-short --system-variant short" in r.stdout
    assert "TTG_PROMPT_VARIANT=short TTG_MODEL=ttg-short" in r.stdout
    s = _serve("sglang")
    assert s.returncode == 0 and "--model ttg-llm:ttg-short" in s.stdout
    big = tmp_path / "ttg-short"
    big.mkdir()
    (big / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))
    (big / "adapter_model.safetensors").write_bytes(b"\0")
    r32 = _serve("vllm", str(big))
    assert r32.returncode == 0 and "--max-lora-rank 32" in r32.stdout and f"{tmp_path}:/adapters:ro" in r32.stdout
    wrong = tmp_path / "my-adapter"
    wrong.mkdir()
    (wrong / "adapter_config.json").write_text(json.dumps({"r": 16}))
    assert _serve("vllm", str(wrong)).returncode == 2
    assert _serve("trtllm").returncode == 2


# Field names checked against the trl 1.16.0 (SFTConfig) and transformers 5.20.1 (TrainingArguments) wheels.
VERIFIED_SFT_KEYS = {"output_dir", "num_train_epochs", "max_steps", "per_device_train_batch_size",
                     "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
                     "warmup_steps", "logging_steps", "save_strategy", "eval_strategy", "eval_steps", "bf16",
                     "gradient_checkpointing", "report_to", "seed", "max_length", "packing", "completion_only_loss",
                     "model_init_kwargs"}


def test_sft_config_uses_only_verified_field_names():
    a = tl.parse_args(["--eval-steps", "50"])
    keys = set(tl.sft_config_kwargs(a, {"max_length": 4096}))
    assert keys <= VERIFIED_SFT_KEYS, keys - VERIFIED_SFT_KEYS
    assert set(tl.lora_config_kwargs(a)) == {"r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"}
