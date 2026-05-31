import torch
from fate_oia.models.trace_oia_model import TraceOIAModel

def test_trace_model_action_protected_and_reason_changes():
    model = TraceOIAModel(dim=32)
    tokens = torch.randn(2, 17, 32)
    out = model(tokens, batch={"file_name": ["a.jpg", "b.jpg"]}, return_cf=True, cf_targets=torch.zeros(2, 21), image_height=32, image_width=32, patch_size=8)
    assert torch.allclose(out["action_logits"], out["base_action_logits"], atol=1e-8)
    assert out["transport"]["T"].shape[:3] == (2, 21, 6)
    ev = dict(out["evidence"]); ev["evidence_tokens"] = ev["evidence_tokens"] + 1.0
    changed = model.transport(out["label_tokens"][:, 4:], out["base_reason_logits"], ev)["evidence_reason_logits"]
    assert (changed - out["transport"]["evidence_reason_logits"]).abs().mean() > 0
