from pathlib import Path

import yaml


def test_site_v20_config_enables_sparse_task_private_temporal_evidence():
    path = Path("configs/fate_oia_train_tida_site_v20.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment"]["name"] == "tida_site_v20"
    assert config["data"]["history_frames"] == 5
    assert config["model"]["context_chunk_size"] == 5
    assert config["model"]["reason_local_query_enabled"] is True
    assert config["model"]["reason_local_temporal_reason_indices"] == [
        1, 2, 5, 6, 7, 8, 10, 14, 16, 20
    ]
    for name in (
        "reason_local_aux", "reason_local_rank", "reason_local_utility",
        "reason_local_no_harm", "reason_local_delta",
        "reason_local_order", "reason_local_deletion",
    ):
        assert config["loss"][name] > 0
    for name in (
        "relational_reason_aux", "relational_reason_rank",
        "relational_reason_deletion", "relational_reason_no_harm",
    ):
        assert config["loss"][name] == 0
    assert config["backbone"]["freeze_backbone"] is True
    assert config["backbone"]["feature_cache_enabled"] is False
    assert config["backbone"]["token_compression"] == "none"
    assert config["deployment"]["calibration_source"] == "train_calib"
    assert config["deployment"]["test_oracle_writeback"] is False
