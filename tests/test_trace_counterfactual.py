from pathlib import Path
import torch
from fate_oia.models.trace_oia_model import TraceOIAModel

def test_transport_counterfactual_not_proxy():
    model = TraceOIAModel(dim=32); labels = torch.zeros(2, 21); labels[:, 0] = 1
    out = model(torch.randn(2, 17, 32), batch={"file_name": ["a.jpg", "b.jpg"]}, return_cf=True, cf_targets=labels, image_height=32, image_width=32, patch_size=8)
    assert out["cf"]["cf_is_proxy"] is False and "cf_target_transport_mask" in out["cf"]
    assert "target_deleted_reason = reason_logits -" not in Path("fate_oia/models/transport_counterfactual.py").read_text()
