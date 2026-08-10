import yaml


def test_source_head_is_exactly_locked():
    cfg=yaml.safe_load(open("configs/fate_oia_train_360x640_dice_oia_v1_probe.yaml",encoding="utf-8"))
    assert cfg["experiment"]["source_head"]=="8372dbb0bf0544ad0a3e3b741dc5d3abaab5a5cf"
