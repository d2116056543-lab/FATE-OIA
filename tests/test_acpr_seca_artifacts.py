import torch

from fate_oia.utils.acpr_seca_artifacts import seca_metrics_payload


def test_seca_metrics_payload_schema():
    payload = seca_metrics_payload({"seca_enabled": True, "seca_residual_scale": torch.zeros(4), "seca_null_attention": torch.ones(2,4)})
    assert payload["available"] is True
    assert "residual_scale" in payload
    assert "null_attention_mean" in payload



def test_seca_metrics_payload_per_action_deltas():
    metrics = {
        "metrics_legacy_base_fixed": {"Act_mF1": 0.5, "per_action_F1": [0.1, 0.2, 0.3, 0.4]},
        "metrics_base_fixed": {"Act_mF1": 0.6, "per_action_F1": [0.2, 0.1, 0.5, 0.4]},
        "metrics_raw_fixed": {"Act_mF1": 0.7},
    }
    payload = seca_metrics_payload({"seca_enabled": True, "seca_residual_scale": torch.zeros(4)}, metrics)
    assert payload["legacy_minus_seca_per_action"] == [-0.1, 0.1, -0.2, 0.0]
    assert payload["seca_minus_legacy_per_action"] == [0.1, -0.1, 0.2, 0.0]
