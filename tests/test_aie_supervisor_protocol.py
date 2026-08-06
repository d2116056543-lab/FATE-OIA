from pathlib import Path


def test_supervisor_is_attached_and_requires_review_and_pilot():
    text = Path("fate_oia/engine/supervise_aie_oia_foreground.py").read_text(encoding="utf-8")
    assert "subprocess.call" in text
    assert "AIE_IMPLEMENTATION_REVIEW.json" in text and "AIE_FULL_TRAIN_READY.json" in text
    for forbidden in ("Start-Process", "Start-Job", "nohup", "Popen"):
        assert forbidden not in text

