from __future__ import annotations

import torch

INTERVENTION_NAMES = [
    "global_only",
    "regime_off",
    "phase_off",
    "source_off",
    "factor_off",
    "predicate_off",
    "evidence_tube_off",
    "equal_mass_random",
    "temporal_reverse",
    "temporal_shuffle",
    "lag_disabled",
    "last_frame_only",
    "prefix_5",
    "prefix_10",
    "prefix_15",
]


def zero_predicate_intervention(predicate_probs: torch.Tensor, indices: list[int]) -> torch.Tensor:
    out = predicate_probs.clone()
    if indices:
        out[:, indices] = 0
    return out


def selected_vs_random_influence(action_logits: torch.Tensor, intervened_logits: torch.Tensor) -> torch.Tensor:
    return (action_logits.softmax(-1).max(-1).values - intervened_logits.softmax(-1).max(-1).values).detach()


def apply_temporal_intervention(frames: torch.Tensor, name: str) -> torch.Tensor:
    if name == "temporal_reverse":
        return frames.flip(1)
    if name == "temporal_shuffle":
        idx = torch.randperm(frames.shape[1], device=frames.device)
        return frames[:, idx]
    if name == "last_frame_only":
        return frames[:, -1:].expand_as(frames)
    if name == "prefix_5":
        out = frames.clone()
        out[:, 5:] = frames[:, 4:5]
        return out
    if name == "prefix_10":
        out = frames.clone()
        out[:, 10:] = frames[:, 9:10]
        return out
    if name == "prefix_15":
        return frames
    raise ValueError(f"unsupported temporal intervention: {name}")


def intervention_suite() -> list[str]:
    return list(INTERVENTION_NAMES)


@torch.no_grad()
def evaluate_intervention_suite(model, frames: torch.Tensor, epoch: int = 0, names: list[str] | None = None) -> dict:
    """Run real intervention forwards and report action probability deltas.

    Temporal interventions modify input frames and rerun the whole model. The
    structural interventions use the model's formal intervention hook, which
    recomputes downstream state/flow/ledger tensors instead of changing display
    tensors after the fact.
    """
    model.eval()
    names = names or intervention_suite()
    full = model(frames, epoch=epoch)
    full_action = full.action_logits.softmax(-1)
    full_exp = full.exp29_logits.sigmoid()
    results: dict[str, dict] = {}
    for name in names:
        if name in {"temporal_reverse", "temporal_shuffle", "last_frame_only", "prefix_5", "prefix_10", "prefix_15"}:
            out = model(apply_temporal_intervention(frames, name), epoch=epoch)
            recompute = "full_model_from_frames"
        elif name == "equal_mass_random":
            shuffled = frames.clone()
            b, t = shuffled.shape[:2]
            flat = shuffled.flatten(0, 1)
            perm = torch.randperm(flat.shape[0], device=frames.device)
            shuffled = flat[perm].view_as(shuffled)
            out = model(shuffled, epoch=epoch)
            recompute = "full_model_equal_mass_frame_random"
        else:
            out = model(frames, epoch=epoch, intervention=name)
            recompute = "downstream_recompute_from_formal_hook"
        act = out.action_logits.softmax(-1)
        exp = out.exp29_logits.sigmoid()
        action_delta = (full_action - act).abs().mean().item()
        exp_delta = (full_exp - exp).abs().mean().item()
        results[name] = {
            "action_prob_l1_delta": action_delta,
            "exp29_prob_l1_delta": exp_delta,
            "recompute": recompute,
        }
    nonzero = [v for v in results.values() if v["action_prob_l1_delta"] > 1e-7 or v["exp29_prob_l1_delta"] > 1e-7]
    return {
        "pass": len(nonzero) >= max(3, len(results) // 2),
        "full_action_prob_mean": full_action.mean(0),
        "results": results,
        "nonzero_delta_count": len(nonzero),
    }
