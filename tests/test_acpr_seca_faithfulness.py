from pathlib import Path


def test_seca_faithfulness_is_eval_only():
    text = Path("fate_oia/engine/eval_acpr_seca_faithfulness.py").read_text(encoding="utf-8")
    assert "eval_only" in text
    assert "loss" not in text.lower()
