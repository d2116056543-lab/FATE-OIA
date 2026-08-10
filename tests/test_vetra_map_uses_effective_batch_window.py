from pathlib import Path


def test_training_computes_map_after_microbatch_concatenation():
    text=(Path(__file__).parents[1]/"fate_oia/engine/train_vetra_oia_probe.py").read_text()
    append=text.index("window_outputs.append(output)")
    merge=text.index("merged={key:torch.cat")
    loss=text.index("total,components=total_vetra_loss(merged")
    backward=text.index("total.backward()",loss)
    assert append < merge < loss < backward
    assert "total.backward(); micro_count" not in text
