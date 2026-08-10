from pathlib import Path


def test_all_four_control_families_are_configured():
    text=Path("configs/fate_oia_train_360x640_dice_oia_v1_probe.yaml").read_text(encoding="utf-8")
    for name in ("same_region_1","same_region_2","wrong_probe_own_region","wrong_action_own_region"):
        assert name in text
