import torch
from fate_oia.utils.action_candidate_selector import evaluate_action_candidates, select_action_candidate


def test_selects_candidate_with_higher_action_mf1():
    labels = torch.tensor([[1,0,0,0, 1,0],[0,1,0,0, 0,1]], dtype=torch.float32)
    reason = torch.tensor([[4.0,-4.0],[-4.0,4.0]])
    candidates = {
        "bad": torch.tensor([[-4.0,4.0,0.0,0.0],[4.0,-4.0,0.0,0.0]]),
        "good": torch.tensor([[4.0,-4.0,-4.0,-4.0],[-4.0,4.0,-4.0,-4.0]]),
    }
    result = evaluate_action_candidates(candidates, reason, labels, action_dim=4)
    selected = select_action_candidate(result)
    assert selected["selected_action_mode"] == "good"
    assert selected["selected_action_metrics"]["Act_mF1"] > result["bad"]["metrics"]["Act_mF1"]
