from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel
from fate_oia.acpr_interactflow.timing import REQUIRED_TIMING_SECTIONS


def test_model_forward_surfaces_stage_timing_with_no_placeholder_schema() -> None:
    model = ACPRInteractFlowPPModel(
        use_mock_dino=True,
        action_dim=3,
        dino_input_height=64,
        dino_input_width=96,
        dino_chunk_size=2,
        anchor_frames=(0, 3, 6, 9, 12, 14),
        selected_layers=(3, 7, 11),
    )
    frames = torch.rand(2, 15, 3, 64, 96)

    out = model(frames, action_soft_target=torch.softmax(torch.randn(2, 3), dim=-1))

    timing = out.aux["model_timing"]
    for key in REQUIRED_TIMING_SECTIONS:
        assert f"{key}_time" in timing
    assert timing["visual_dino_time"] > 0.0
    assert timing["visual_motion_time"] > 0.0
    assert timing["predicate_time"] > 0.0
    assert timing["interaction_flow_time"] > 0.0
    assert timing["decision_ledger_time"] > 0.0
    assert timing["exp29_time"] > 0.0
    assert timing["total_profiled_time"] > 0.0
