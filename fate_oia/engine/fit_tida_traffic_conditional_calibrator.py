from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _macro_f1(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = logits >= 0
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())


def _apply_policy(rows: dict[str, Any], policy: dict[str, Any], cap: float = 0.08):
    candidate = rows["object_intent_action_candidate"].float()
    directional = rows["object_intent_action_directional_utility_gate"].float()
    risk = rows["object_intent_action_risk_utility_gate"].float()
    source = torch.as_tensor(policy["utility_source"], dtype=torch.long)
    utility = torch.where(source[None] == 1, risk, directional)
    gate = torch.as_tensor(policy["gate"], dtype=candidate.dtype)
    scale = torch.as_tensor(policy["scale"], dtype=candidate.dtype)
    cutoff = torch.as_tensor(policy["cutoff"], dtype=candidate.dtype)
    selected = (gate[None] > 0) & (utility >= cutoff[None])
    delta = gate[None] * selected.to(candidate.dtype) * (
        candidate * scale[None]
    ).clamp(-float(cap), float(cap))
    return rows["pre_object_intent_action"].float() + delta


def _align_traffic_rows(
    policy_rows: dict[str, Any], traffic_rows: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, torch.Tensor]]:
    traffic_index = {name: index for index, name in enumerate(traffic_rows["file_names"])}
    policy_index = {name: index for index, name in enumerate(policy_rows["file_names"])}
    common = [name for name in traffic_rows["file_names"] if name in policy_index]
    if len(common) == len(traffic_rows["file_names"]):
        ti = torch.tensor([traffic_index[name] for name in common], dtype=torch.long)
        pi = torch.tensor([policy_index[name] for name in common], dtype=torch.long)
    else:
        sizes = policy_rows.get("_policy_cohort_sizes", {})
        calib_count = int(sizes.get("train_calib", 0))
        core_count = int(sizes.get("train_core", 0))
        if calib_count + core_count != len(traffic_rows["action_target"]):
            raise ValueError("traffic rows cannot be aligned to train calib/core cohorts")
        eligible = list(range(calib_count)) + list(
            range(len(policy_rows["action_target"]) - core_count, len(policy_rows["action_target"]))
        )

        def fingerprint(source, logits, action, reason):
            return (
                str(source), logits.contiguous().numpy().tobytes(),
                action.contiguous().numpy().tobytes(), reason.contiguous().numpy().tobytes(),
            )

        buckets: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
        for index in eligible:
            buckets[fingerprint(
                policy_rows["source_batches"][index],
                policy_rows["pre_object_intent_action"][index],
                policy_rows["action_target"][index], policy_rows["reason_target"][index],
            )].append(index)
        matched = []
        for index in range(len(traffic_rows["action_target"])):
            key = fingerprint(
                traffic_rows["source_batches"][index],
                traffic_rows["video_action_logits_base"][index],
                traffic_rows["action_target"][index], traffic_rows["reason_target"][index],
            )
            if not buckets[key]:
                raise ValueError("traffic row fingerprint has no policy-row match")
            matched.append(buckets[key].popleft())
        if any(bucket for bucket in buckets.values()):
            raise ValueError("policy-row fingerprints were not consumed one-to-one")
        ti = torch.arange(len(matched), dtype=torch.long)
        pi = torch.tensor(matched, dtype=torch.long)
    policy_count = len(policy_rows["action_target"])

    def align_policy_value(value):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == policy_count:
            return value.index_select(0, pi)
        if isinstance(value, list) and len(value) == policy_count:
            return [value[index] for index in pi.tolist()]
        return value

    aligned_policy = {
        key: align_policy_value(value) for key, value in policy_rows.items()
    }
    # Legacy policy artifacts can have repaired tensors but stale core-only names.
    # Downstream cohort slicing must use the one-to-one aligned traffic identities.
    aligned_policy["file_names"] = list(common)
    aligned_traffic = {
        key: value.index_select(0, ti) if torch.is_tensor(value) and value.shape[0] == len(traffic_rows["file_names"]) else value
        for key, value in traffic_rows.items()
    }
    return pi, ti, [traffic_rows["source_batches"][index] for index in ti], {
        **aligned_traffic, "_policy": aligned_policy,
    }


def _traffic_features(
    rows: dict[str, torch.Tensor], margin: torch.Tensor
) -> torch.Tensor:
    risk = rows["trajectory_interaction_risk"].float()
    return torch.cat((
        margin[..., None], margin.abs()[..., None],
        rows["traffic_trajectory_order_delta"].float()[..., None],
        rows["traffic_trajectory_support"].float()[..., None],
        rows["trajectory_state_strength"].float()[..., None],
        risk.mean(-1, keepdim=True), risk.amax(-1, keepdim=True),
        rows["traffic_trajectory_state_features"].float(),
    ), dim=-1)


class TrafficConditionalCalibrator(nn.Module):
    def __init__(self, actions: int, feature_dim: int, cap: float, bandwidth: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(actions, feature_dim))
        self.bias = nn.Parameter(torch.zeros(actions))
        self.cap = float(cap)
        self.bandwidth = float(bandwidth)

    def forward(
        self, margin: torch.Tensor, features: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = (features * self.weight[None]).sum(-1) + self.bias[None]
        uncertainty = torch.exp(-margin.detach().abs() / self.bandwidth)
        support_gate = 0.15 + 0.85 * support.detach().clamp(0.0, 1.0)
        residual = self.cap * torch.tanh(raw) * uncertainty * support_gate
        return margin + residual, residual


def _source_balanced_weight(source_ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sample_weight = torch.zeros_like(source_ids, dtype=torch.float32)
    for source in source_ids.unique():
        mask = source_ids == source
        sample_weight[mask] = 1.0 / mask.sum().clamp_min(1)
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-8)
    positive_rate = target.mean(0).clamp(0.05, 0.95)
    class_weight = torch.where(
        target > 0.5, 0.5 / positive_rate[None], 0.5 / (1.0 - positive_rate)[None]
    )
    return sample_weight[:, None] * class_weight


def _soft_f1_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    tp = (probability * target).sum(0)
    fp = (probability * (1.0 - target)).sum(0)
    fn = ((1.0 - probability) * target).sum(0)
    return 1.0 - (2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-6)).mean()


def _fit_model(
    margin: torch.Tensor, features: torch.Tensor, support: torch.Tensor,
    target: torch.Tensor, source_ids: torch.Tensor, index: torch.Tensor,
    *, cap: float, bandwidth: float, l2: float, steps: int, lr: float,
    feature_mask: torch.Tensor,
) -> TrafficConditionalCalibrator:
    device = margin.device
    model = TrafficConditionalCalibrator(
        margin.shape[1], features.shape[2], cap, bandwidth
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    selected_features = features * feature_mask[None, None]
    weights = _source_balanced_weight(source_ids[index], target[index])
    near = torch.exp(-margin[index].abs() / bandwidth)
    for _ in range(steps):
        output, residual = model(
            margin[index], selected_features[index], support[index]
        )
        bce = F.binary_cross_entropy_with_logits(
            output, target[index], reduction="none"
        )
        loss = ((0.25 + 0.75 * near) * weights * bce).mean()
        loss = loss + 0.25 * _soft_f1_loss(output, target[index])
        loss = loss + float(l2) * model.weight.square().mean()
        loss = loss + 0.02 * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _fold_assignment(source_ids: torch.Tensor, folds: int) -> torch.Tensor:
    assignment = torch.empty_like(source_ids)
    for source in source_ids.unique():
        index = torch.where(source_ids == source)[0]
        assignment[index] = torch.arange(len(index), device=source_ids.device) % folds
    return assignment


def _cross_fit(
    margin: torch.Tensor, features: torch.Tensor, support: torch.Tensor,
    target: torch.Tensor, source_ids: torch.Tensor, *, folds: int,
    cap: float, bandwidth: float, l2: float, steps: int, lr: float,
    feature_mask: torch.Tensor,
) -> tuple[list[TrafficConditionalCalibrator], torch.Tensor, dict[str, Any]]:
    assignment = _fold_assignment(source_ids, folds)
    output = margin.clone()
    models = []
    fold_rows = []
    for fold in range(folds):
        fit_index = torch.where(assignment != fold)[0]
        hold_index = torch.where(assignment == fold)[0]
        model = _fit_model(
            margin, features, support, target, source_ids, fit_index,
            cap=cap, bandwidth=bandwidth, l2=l2, steps=steps, lr=lr,
            feature_mask=feature_mask,
        )
        with torch.no_grad():
            output[hold_index] = model(
                margin[hold_index], features[hold_index] * feature_mask[None, None],
                support[hold_index],
            )[0]
        models.append(model.cpu())
        fold_rows.append({
            "fold": fold,
            "base_mf1": _macro_f1(margin[hold_index], target[hold_index]),
            "calibrated_mf1": _macro_f1(output[hold_index], target[hold_index]),
        })
    return models, output, {
        "base_oof_mf1": _macro_f1(margin, target),
        "calibrated_oof_mf1": _macro_f1(output, target),
        "folds": fold_rows,
    }


def _effectiveness(base: torch.Tensor, final: torch.Tensor, target: torch.Tensor):
    truth = target > 0.5
    base_correct = (base >= 0) == truth
    final_correct = (final >= 0) == truth
    recovered = (~base_correct) & final_correct
    damaged = base_correct & (~final_correct)
    return {
        "errors_recovered": int(recovered.sum()),
        "correct_damaged": int(damaged.sum()),
        "net_corrected": int(recovered.sum() - damaged.sum()),
        "net_corrected_by_action": (recovered.sum(0) - damaged.sum(0)).tolist(),
        "nll_improvement": float(
            F.binary_cross_entropy_with_logits(base, target)
            - F.binary_cross_entropy_with_logits(final, target)
        ),
        "brier_improvement": float(
            (base.sigmoid() - target).square().mean()
            - (final.sigmoid() - target).square().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", required=True)
    parser.add_argument("--traffic-rows", required=True)
    parser.add_argument("--policy-metrics", required=True)
    parser.add_argument("--test-epoch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_rows = torch.load(args.policy_rows, map_location="cpu")
    traffic_artifact = torch.load(args.traffic_rows, map_location="cpu")
    _, _, sources, train = _align_traffic_rows(policy_rows, traffic_artifact["rows"])
    metrics = json.loads(Path(args.policy_metrics).read_text(encoding="utf-8"))
    policy = metrics["object_intent_deployment_gate_fit"]["online"]["action"]
    train_logits = _apply_policy(train["_policy"], policy)
    threshold = torch.tensor(metrics["online"]["thresholds"]["image"][:4])
    threshold_logit = torch.logit(threshold.clamp(1e-5, 1 - 1e-5))
    train_margin = train_logits - threshold_logit[None]
    train_target = train["action_target"].float()
    train_features = _traffic_features(train, train_margin)
    support = train["traffic_trajectory_support"].float()
    source_names = sorted(set(sources))
    source_map = {name: index for index, name in enumerate(source_names)}
    source_ids = torch.tensor([source_map[name] for name in sources])
    mean = train_features.mean((0, 1), keepdim=True)
    std = train_features.std((0, 1), keepdim=True).clamp_min(1e-4)
    train_features = (train_features - mean) / std
    train_margin = train_margin.to(device)
    train_target = train_target.to(device)
    train_features = train_features.to(device)
    support = support.to(device)
    source_ids = source_ids.to(device)
    traffic_mask = torch.ones(train_features.shape[-1], device=device)
    margin_mask = torch.zeros_like(traffic_mask); margin_mask[:2] = 1.0
    candidates = []
    for cap in (0.02, 0.04, 0.06):
        for bandwidth in (0.15, 0.25, 0.40):
            models, oof, report = _cross_fit(
                train_margin, train_features, support, train_target, source_ids,
                folds=args.folds, cap=cap, bandwidth=bandwidth, l2=0.05,
                steps=args.steps, lr=args.lr, feature_mask=traffic_mask,
            )
            report.update({"cap": cap, "bandwidth": bandwidth, "models": models})
            candidates.append(report)
    selected = max(candidates, key=lambda item: item["calibrated_oof_mf1"])
    _, _, margin_only = _cross_fit(
        train_margin, train_features, support, train_target, source_ids,
        folds=args.folds, cap=selected["cap"], bandwidth=selected["bandwidth"],
        l2=0.05, steps=args.steps, lr=args.lr, feature_mask=margin_mask,
    )
    test_dir = Path(args.test_epoch_dir)
    test_logits = torch.load(test_dir / "video_action_test.pt", map_location="cpu").float()
    test_target = torch.load(test_dir / "action_target_test.pt", map_location="cpu").float()
    test_margin = test_logits - threshold_logit[None]
    test_rows = {
        "traffic_trajectory_order_delta": torch.load(test_dir / "traffic_trajectory_order_delta_test.pt", map_location="cpu"),
        "traffic_trajectory_support": torch.load(test_dir / "traffic_trajectory_support_test.pt", map_location="cpu"),
        "trajectory_state_strength": torch.load(test_dir / "trajectory_state_strength_test.pt", map_location="cpu"),
        "trajectory_interaction_risk": torch.load(test_dir / "trajectory_interaction_risk_test.pt", map_location="cpu"),
        "traffic_trajectory_state_features": torch.load(test_dir / "traffic_trajectory_state_features_test.pt", map_location="cpu"),
    }
    test_features = (_traffic_features(test_rows, test_margin) - mean) / std
    ensemble = []
    residuals = []
    for model in selected["models"]:
        model = model.to(device)
        with torch.no_grad():
            value, residual = model(
                test_margin.to(device), test_features.to(device),
                test_rows["traffic_trajectory_support"].to(device),
            )
        ensemble.append(value.cpu()); residuals.append(residual.cpu())
    final = torch.stack(ensemble).mean(0)
    residual = torch.stack(residuals).mean(0)
    generator = torch.Generator().manual_seed(20260827)
    shuffle = torch.randperm(len(test_features), generator=generator)
    shuffled_outputs = []
    for model in selected["models"]:
        model = model.to(device)
        with torch.no_grad():
            shuffled_outputs.append(model(
                test_margin.to(device), test_features[shuffle].to(device),
                test_rows["traffic_trajectory_support"][shuffle].to(device),
            )[0].cpu())
    shuffled = torch.stack(shuffled_outputs).mean(0)
    payload = {
        "test_labels_used_for_fit_or_selection": False,
        "selected": {key: value for key, value in selected.items() if key != "models"},
        "candidate_oof": [{key: value for key, value in row.items() if key != "models"} for row in candidates],
        "margin_only_oof": margin_only,
        "traffic_incremental_oof_mf1": selected["calibrated_oof_mf1"] - margin_only["calibrated_oof_mf1"],
        "test": {
            "base_Act_mF1": _macro_f1(test_margin, test_target),
            "calibrated_Act_mF1": _macro_f1(final, test_target),
            "traffic_shuffle_Act_mF1": _macro_f1(shuffled, test_target),
            "traffic_shuffle_drop": _macro_f1(final, test_target) - _macro_f1(shuffled, test_target),
            "residual_rms": float(residual.square().mean().sqrt()),
            "residual_active_rate_gt_0p002": float((residual.abs() > 0.002).float().mean()),
            **_effectiveness(test_margin, final, test_target),
        },
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "traffic_conditional_calibrator_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({
        "models": [model.state_dict() for model in selected["models"]],
        "feature_mean": mean, "feature_std": std,
        "cap": selected["cap"], "bandwidth": selected["bandwidth"],
        "threshold": threshold,
    }, output / "traffic_conditional_calibrator.pt")
    torch.save({
        "base_margin": test_margin, "calibrated_margin": final,
        "shuffled_margin": shuffled, "residual": residual,
        "action_target": test_target,
    }, output / "traffic_conditional_calibrator_test.pt")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
