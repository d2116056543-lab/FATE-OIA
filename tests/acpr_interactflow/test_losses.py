from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.types import PSIInteractFlowBatch
from fate_oia.losses.acpr_interactflow_losses import compute_interactflow_losses
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel


def test_losses_do_not_treat_unknown_exp29_as_hard_negative():
    model = ACPRInteractFlowPPModel(
        pretrained_weights="missing-is-ok-with-mock",
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path="configs/acpr_interactflow_state_grammar.yaml",
        action_dim=3,
        use_mock_dino=True,
    )
    frames = torch.randn(2, 15, 3, 32, 64)
    out = model(frames)
    batch = PSIInteractFlowBatch(
        input_frames=frames,
        action_soft=torch.nn.functional.one_hot(torch.tensor([0, 2]), 3).float(),
        action_majority=torch.tensor([0, 2]),
        exp29=torch.zeros(2, 29),
        exp29_mask=torch.zeros(2, 29),
        paper_effective_weight=torch.ones(2),
        video_id=["a", "b"],
        start_frame=torch.zeros(2, dtype=torch.long),
        target_frame_index=torch.zeros(2, dtype=torch.long),
        target_frame_path=["", ""],
        frame_paths=[[], []],
        explanation_text=["", ""],
        reasoning_text=["", ""],
        sample_id=["a", "b"],
    )
    loss, terms = compute_interactflow_losses(out, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(terms["exp29_masked_asl"])
    assert "action_final_soft_kl" in terms
    assert "contribution_alignment_js" in terms
