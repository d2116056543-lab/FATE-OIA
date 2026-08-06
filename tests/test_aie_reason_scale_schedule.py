from fate_oia.engine.train_aie_oia import load_config, schedule_values


def test_reason_residual_schedule_respects_metric_backed_safety_cap():
    config = load_config("configs/fate_oia_train_360x640_aie_oia_v1.yaml")
    assert config["reason_private"]["reason_scale_max"] == 0.60
    assert schedule_values(100, 100, config)["reason"] == 0.60
