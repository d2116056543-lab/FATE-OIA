import torch
from fate_oia.models.diva_dense_adapter import ActionSpecificLayerMixer, DrivingDenseAdapter

def test_dense_adapter_preserves_action_dimension():
    maps = {3: torch.randn(2,384,45,80), 6: torch.randn(2,384,45,80), 9: torch.randn(2,384,45,80), 12: torch.randn(2,384,45,80)}
    mixer = ActionSpecificLayerMixer(dim=384, action_dim=4, layer_indices=(3,6,9,12))
    action_maps, gates = mixer(maps)
    assert action_maps.shape == (2,4,384,45,80)
    assert gates.shape == (4,4)
    adapter = DrivingDenseAdapter(dim=384, action_dim=4)
    out = adapter(action_maps)
    assert out['P1'].shape[:3] == (2,4,384)
    assert out['P1'].shape[-2:] == (45,80)
    assert out['P2'].shape[-2:] == (23,40)
    assert out['P3'].shape[-2:] == (12,20)
