"""M8 pipeline logic without a GPU: dataset filtering, loss masking, and a release gate that can actually fail."""
import json
import sys

import eval_gate
import make_dataset
import train_lora


def test_dataset_keeps_only_valid_teacher_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_dataset.py", "--endpoint", "mock", "--n", "20", "--out", str(tmp_path)])
    make_dataset.main()
    stats = json.loads((tmp_path / "dataset_stats.json").read_text())
    assert stats["kept"] == 20 and stats["train"] + stats["val"] == 20
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"], "student sees the SHORT instruction only"
    assert "Plover JSON" in row["messages"][0]["content"]
    assert json.loads(row["messages"][1]["content"])["recommendation"]


class FakeTok:
    """apply_chat_template stand-in: one token per character, assistant turn appended last."""
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        ids = [ord(c) % 250 for m in msgs for c in m["content"][:40]]
        return ids + ([1] if add_generation_prompt else [])


def test_loss_is_masked_to_the_answer():
    row = {"messages": [{"role": "user", "content": "short instruction + case"}, {"role": "assistant", "content": "{json answer}"}]}
    enc = train_lora.encode(FakeTok(), row)
    n_prompt = len(FakeTok().apply_chat_template(row["messages"][:-1], add_generation_prompt=True))
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert any(x != -100 for x in enc["labels"]) and len(enc["labels"]) == len(enc["input_ids"])


def test_release_gate_passes_good_adapter_and_blocks_missing_one(tmp_path, monkeypatch, capsys):
    # the mock answers in microseconds, so a latency RATIO is pure noise there - relax only that threshold
    gate = json.loads((eval_gate.ROOT / "mlops" / "gate.json").read_text())
    gate["max_latency_p95_ratio"] = 1e6
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--endpoint", "mock", "--candidate", "plover-style-v1", "--n", "10",
                                      "--gate", str(tmp_path / "gate.json")])
    assert eval_gate.main() == 0
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--endpoint", "mock", "--candidate", "plover-style-v9", "--n", "10",
                                      "--gate", str(tmp_path / "gate.json")])
    assert eval_gate.main() == 1, "an adapter NIM does not serve must fail the gate, not slip through"
    assert "GATE: FAIL" in capsys.readouterr().out
