from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.interventions import evaluate_intervention_suite
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel


def test_interventions_run_real_forward_paths():
    model = ACPRInteractFlowPPModel(
        pretrained_weights="missing-is-ok-with-mock",
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path="configs/acpr_interactflow_state_grammar.yaml",
        action_dim=3,
        use_mock_dino=True,
    )
    frames = torch.randn(1, 15, 3, 32, 64)
    report = evaluate_intervention_suite(
        model,
        frames,
        names=["global_only", "predicate_off", "lag_disabled", "temporal_reverse", "last_frame_only"],
    )
    assert "results" in report
    assert report["results"]["temporal_reverse"]["recompute"] == "full_model_from_frames"
    assert report["results"]["predicate_off"]["recompute"] == "downstream_recompute_from_formal_hook"
    assert report["nonzero_delta_count"] > 0

