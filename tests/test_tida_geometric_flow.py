import torch

from fate_oia.models.tida_geometric_flow import TIDAGeometricFlowEncoder


def _clip(frame: torch.Tensor, frames: int = 5) -> torch.Tensor:
    return torch.stack([frame.clone() for _ in range(frames)], dim=1)


def test_static_clip_has_negligible_geometric_motion():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    frame = torch.rand(2, 3, 32, 48)
    output = model(_clip(frame), torch.ones(2, 5, dtype=torch.bool))
    assert output["motion_energy"].abs().max() < 1e-5
    assert output["flow_field"].abs().max() < 1e-5


def test_horizontal_translation_has_direction_and_reversal_changes_sign():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    base = torch.zeros(1, 3, 32, 48)
    base[:, :, 8:24, 8:20] = 1.0
    frames = torch.stack([torch.roll(base, shifts=2 * step, dims=-1) for step in range(5)], dim=1)
    ordered = model(frames, torch.ones(1, 5, dtype=torch.bool))
    reversed_output = model(frames.flip(1), torch.ones(1, 5, dtype=torch.bool))
    assert ordered["global_horizontal"].mean() > 0
    assert reversed_output["global_horizontal"].mean() < 0


def test_outward_motion_has_positive_expansion():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(20, 28))
    frames = []
    for size in (4, 6, 8, 10, 12):
        frame = torch.zeros(1, 3, 40, 56)
        cy, cx = 20, 28
        frame[:, :, cy - size : cy + size, cx - size : cx + size] = 1.0
        frames.append(frame)
    output = model(torch.stack(frames, dim=1), torch.ones(1, 5, dtype=torch.bool))
    assert output["global_expansion"].mean() > 0


def test_history_off_is_exact_zero_and_gradients_are_finite():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    frames = torch.rand(2, 5, 3, 32, 48, requires_grad=True)
    off = model(frames, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.equal(off["flow_state"], torch.zeros_like(off["flow_state"]))
    on = model(frames, torch.ones(2, 5, dtype=torch.bool))
    on["flow_state"].square().mean().backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all()


def test_region_outputs_are_real_and_not_broadcast_copies():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    base = torch.zeros(1, 3, 32, 48)
    base[:, :, 10:24, :16] = 1.0
    frames = torch.stack([torch.roll(base, shifts=step, dims=-1) for step in range(5)], dim=1)
    output = model(frames, torch.ones(1, 5, dtype=torch.bool))
    assert output["region_motion"].shape == (1, 4, 5, 3)
    assert output["region_motion"].std(dim=2).mean() > 0
