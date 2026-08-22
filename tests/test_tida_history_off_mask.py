import torch

from test_tida_model_forward import _ImageBase
from fate_oia.models.tida_oia_model import TIDAOIAModel


def test_history_off_rerun_marks_every_history_frame_invalid():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(_ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7).eval()
    output = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    changed = model.rerun_temporal_from_output(output, "history_off", temporal_action_scale=1.0, temporal_reason_scale=1.0)
    assert not changed["history_valid"].any()
    assert torch.equal(changed["innovation_reliability"], torch.zeros_like(changed["innovation_reliability"]))


def test_history_off_direct_forward_is_exact_image_fallback():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(_ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7).eval()
    output = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0, intervention="history_off",
    )
    assert not output["history_valid"].any()
    assert torch.equal(output["video_action_logits"], output["image_action_logits"])
    assert torch.equal(output["video_reason_logits"], output["image_reason_logits"])
    assert torch.equal(output["innovation_reliability"], torch.zeros_like(output["innovation_reliability"]))
