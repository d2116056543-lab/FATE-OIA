from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fate_oia.engine.fit_tida_traffic_conditional_calibrator import (
    _align_traffic_rows,
    _apply_policy,
)


def _macro_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())


def _action_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def _per_action_f1(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    return [_action_f1(prediction[:, action], target[:, action]) for action in range(target.shape[1])]


def _audit_action_gate(
    baseline: np.ndarray, full: np.ndarray, margin_only: np.ndarray,
    correction_mask: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | bool]]]:
    gates = []
    rows = []
    for action in range(target.shape[1]):
        mask = correction_mask[:, action]
        corrected = np.logical_and(mask, full[:, action] == target[:, action]).sum()
        precision = float(corrected / max(1, mask.sum()))
        base_f1 = _action_f1(baseline[:, action], target[:, action])
        full_f1 = _action_f1(full[:, action], target[:, action])
        margin_f1 = _action_f1(margin_only[:, action], target[:, action])
        enabled = bool(
            mask.sum() >= 4 and precision >= 0.50
            and full_f1 > base_f1 and full_f1 > margin_f1
        )
        gates.append(enabled)
        rows.append({
            "action": action, "enabled": enabled,
            "base_f1": base_f1, "full_f1": full_f1,
            "margin_only_f1": margin_f1,
            "correction_count": int(mask.sum()), "correction_precision": precision,
        })
    return np.asarray(gates, dtype=bool), rows


def _select_deployment_routes(
    baseline: np.ndarray, margin_only: np.ndarray, traffic: np.ndarray,
    target: np.ndarray,
) -> tuple[list[str], list[dict[str, Any]]]:
    routes = []
    rows = []
    for action in range(target.shape[1]):
        scores = {
            "baseline": _action_f1(baseline[:, action], target[:, action]),
            "margin_only": _action_f1(margin_only[:, action], target[:, action]),
            "traffic": _action_f1(traffic[:, action], target[:, action]),
        }
        # Baseline wins ties, so a learned branch is deployed only with observed
        # calibration improvement. Traffic and threshold effects stay separable.
        route = max(scores, key=lambda name: (scores[name], -(
            "baseline", "margin_only", "traffic"
        ).index(name)))
        routes.append(route)
        rows.append({"action": action, "selected_route": route, **scores})
    return routes, rows


def _route_predictions(
    routes: list[str], baseline: np.ndarray, margin_only: np.ndarray,
    traffic: np.ndarray,
) -> np.ndarray:
    output = baseline.copy()
    choices = {"baseline": baseline, "margin_only": margin_only, "traffic": traffic}
    for action, route in enumerate(routes):
        output[:, action] = choices[route][:, action]
    return output


def _risk_coverage_curve(
    baseline: np.ndarray, probability: np.ndarray, target: np.ndarray,
    points: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.05),
) -> list[dict[str, float | int]]:
    candidate = probability >= 0.5
    disagreement = np.flatnonzero(candidate != baseline)
    if disagreement.size == 0:
        return []
    confidence = np.abs(probability[disagreement] - 0.5) * 2.0
    ranked = disagreement[np.argsort(-confidence)]
    rows = []
    for coverage in points:
        count = min(len(ranked), max(1, int(np.ceil(coverage * len(target)))))
        selected = ranked[:count]
        recovered = np.logical_and(
            baseline[selected] != target[selected], candidate[selected] == target[selected],
        ).sum()
        damaged = np.logical_and(
            baseline[selected] == target[selected], candidate[selected] != target[selected],
        ).sum()
        rows.append({
            "coverage": float(count / len(target)),
            "selected_count": int(count),
            "correction_precision": float(recovered / max(1, count)),
            "net_corrected": int(recovered - damaged),
        })
    return rows


def _rank_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result = {"average_precision": float(average_precision_score(target, probability))}
    result["roc_auc"] = (
        float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else float("nan")
    )
    return result


def _shuffle_traffic_features(
    features: np.ndarray, permutation: np.ndarray, margin_dim: int,
) -> np.ndarray:
    shuffled = features.copy()
    shuffled[:, margin_dim:] = features[permutation, margin_dim:]
    return shuffled


def _features(rows: dict[str, Any], margin: torch.Tensor) -> np.ndarray:
    values = (
        margin.float(), margin.float().abs(),
        rows["traffic_trajectory_order_delta"].float(),
        rows["traffic_trajectory_support"].float(),
        rows["trajectory_state_strength"].float(),
        rows["trajectory_interaction_risk"].float().flatten(1),
        rows["traffic_trajectory_state_features"].float().flatten(1),
    )
    return torch.cat(values, dim=1).numpy()


def _margin_features(margin: torch.Tensor) -> np.ndarray:
    return torch.cat((margin.float(), margin.float().abs()), dim=1).numpy()


def _object_track_features(store: dict[str, Any], names: list[str]) -> np.ndarray:
    lookup = {str(name).lower(): index for index, name in enumerate(store["file_names"])}
    missing = [name for name in names if str(name).lower() not in lookup]
    if missing:
        raise ValueError(f"object track store misses {len(missing)} requested clips")
    index = torch.tensor([lookup[str(name).lower()] for name in names], dtype=torch.long)
    track = store["tracks_xy"].index_select(0, index).float()
    visible = store["visibility"].index_select(0, index).bool()
    pair_visible = visible[:, 1:] & visible[:, :-1]
    raw_step = track[:, 1:] - track[:, :-1]
    masked_step = raw_step.masked_fill(~pair_visible[..., None], float("nan"))
    global_flow = torch.nanmedian(masked_step, dim=2).values.nan_to_num()
    step = (raw_step - global_flow[:, :, None]) * pair_visible[..., None]
    mean_velocity = step.sum(1) / pair_visible.sum(1).clamp_min(1)[..., None]
    total = step.sum(1)
    recent = step[:, -3:].sum(1) / pair_visible[:, -3:].sum(1).clamp_min(1)[..., None]
    early = step[:, :3].sum(1) / pair_visible[:, :3].sum(1).clamp_min(1)[..., None]
    acceleration = recent - early
    visibility = visible.float().mean(1)[..., None]
    per_point = torch.cat((
        total, recent, mean_velocity, acceleration, visibility,
    ), dim=-1).flatten(1)
    speed = mean_velocity.square().sum(-1).sqrt()
    quantiles = torch.quantile(speed, torch.tensor([0.25, 0.5, 0.75, 0.9]), dim=1).T
    global_summary = torch.cat((
        total.mean(1), total.std(1), recent.mean(1), recent.std(1),
        mean_velocity.mean(1), mean_velocity.std(1), acceleration.mean(1),
        acceleration.std(1), visibility.squeeze(-1).mean(1, keepdim=True), quantiles,
    ), dim=1)
    return torch.cat((per_point, global_summary), dim=1).numpy()


def _correction_mask(
    margin: np.ndarray, probability: np.ndarray, confidence: float, bandwidth: float,
) -> np.ndarray:
    baseline = margin >= 0
    candidate = probability >= 0.5
    certainty = np.abs(probability - 0.5) * 2.0
    return (candidate != baseline) & (certainty >= confidence) & (np.abs(margin) <= bandwidth)


def _select_rule(
    margin: np.ndarray, probability: np.ndarray, target: np.ndarray, sources: np.ndarray,
) -> dict[str, float]:
    baseline = margin >= 0
    base_f1 = _action_f1(baseline, target)
    best = {
        "confidence": 1.0, "bandwidth": 0.0, "f1": base_f1,
        "coverage": 0.0, "precision": 0.0, "net_corrected": 0.0,
        "worst_source_f1_delta": 0.0,
    }
    for confidence in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        for bandwidth in (0.10, 0.20, 0.35, 0.55, 0.80, 1.20):
            mask = _correction_mask(margin, probability, confidence, bandwidth)
            if mask.sum() < 4:
                continue
            final = baseline.copy()
            final[mask] = probability[mask] >= 0.5
            corrected = np.logical_and(mask, final == target).sum()
            damaged = np.logical_and(mask, baseline == target).sum()
            source_deltas = []
            for source in np.unique(sources):
                index = sources == source
                source_deltas.append(
                    _action_f1(final[index], target[index])
                    - _action_f1(baseline[index], target[index])
                )
            row = {
                "confidence": confidence, "bandwidth": bandwidth,
                "f1": _action_f1(final, target),
                "coverage": float(mask.mean()),
                "precision": float(corrected / max(1, mask.sum())),
                "net_corrected": float(corrected - damaged),
                "worst_source_f1_delta": float(min(source_deltas)),
            }
            safe = row["precision"] >= 0.55 and row["worst_source_f1_delta"] >= -0.005
            if safe and (row["f1"], row["precision"], -row["coverage"]) > (
                best["f1"], best["precision"], -best["coverage"]
            ):
                best = row
    return best


def _fit_oof(
    features: np.ndarray, margin: np.ndarray, target: np.ndarray,
    source_ids: np.ndarray, folds: int, seed: int, fold_mode: str = "stratified",
) -> tuple[np.ndarray, list[list[HistGradientBoostingClassifier]]]:
    count, actions = target.shape
    oof = np.zeros((count, actions), dtype=np.float32)
    models: list[list[HistGradientBoostingClassifier]] = [[] for _ in range(actions)]
    for action in range(actions):
        if fold_mode == "leave_source_out":
            splits = [
                (np.flatnonzero(source_ids != source), np.flatnonzero(source_ids == source))
                for source in np.unique(source_ids)
            ]
        elif fold_mode == "stratified":
            strata = np.char.add(
                np.char.add(source_ids.astype(str), "_"),
                target[:, action].astype(int).astype(str),
            )
            splits = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=seed + action
            ).split(features, strata)
        else:
            raise ValueError(f"unsupported fold_mode: {fold_mode}")
        for fit_index, hold_index in splits:
            positive_rate = np.clip(target[fit_index, action].mean(), 0.05, 0.95)
            weight = np.where(
                target[fit_index, action] > 0.5,
                0.5 / positive_rate, 0.5 / (1.0 - positive_rate),
            )
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=160, max_leaf_nodes=7,
                min_samples_leaf=80, l2_regularization=2.0,
                early_stopping=False, random_state=seed + action,
            )
            model.fit(features[fit_index], target[fit_index, action], sample_weight=weight)
            oof[hold_index, action] = model.predict_proba(features[hold_index])[:, 1]
            models[action].append(model)
    return oof, models


def _apply_rules(
    margin: np.ndarray, probability: np.ndarray, rules: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = margin >= 0
    masks = np.zeros_like(prediction)
    for action, rule in enumerate(rules):
        mask = _correction_mask(
            margin[:, action], probability[:, action],
            rule["confidence"], rule["bandwidth"],
        )
        prediction[mask, action] = probability[mask, action] >= 0.5
        masks[:, action] = mask
    return prediction, masks


def _ensemble_probability(models, features: np.ndarray) -> np.ndarray:
    columns = []
    for action_models in models:
        columns.append(np.stack([
            model.predict_proba(features)[:, 1] for model in action_models
        ]).mean(0))
    return np.stack(columns, axis=1)


def _slice_rows(rows: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    count = len(rows["file_names"])
    result = {}
    for key, value in rows.items():
        if isinstance(value, dict) and key == "_policy":
            result[key] = _slice_rows(value, start, stop)
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            result[key] = value[start:stop]
        elif isinstance(value, list) and len(value) == count:
            result[key] = value[start:stop]
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", required=True)
    parser.add_argument("--traffic-rows", required=True)
    parser.add_argument("--policy-metrics", required=True)
    parser.add_argument("--test-epoch-dir", required=True)
    parser.add_argument("--audit-traffic-rows")
    parser.add_argument("--object-track-store", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold-mode", choices=("stratified", "leave_source_out"),
        default="leave_source_out",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--selection-cohort", choices=("train_calib", "train_audit"),
        default="train_calib",
    )
    args = parser.parse_args()

    policy_rows = torch.load(args.policy_rows, map_location="cpu")
    traffic_artifact = torch.load(args.traffic_rows, map_location="cpu")
    _, _, sources, combined_train = _align_traffic_rows(
        policy_rows, traffic_artifact["rows"]
    )
    calib_count = int(traffic_artifact.get("train_calib_count", 0))
    if args.selection_cohort == "train_calib":
        if calib_count <= 0 or calib_count >= len(combined_train["file_names"]):
            raise ValueError("train_calib selection requires core+calib traffic rows")
        split = len(combined_train["file_names"]) - calib_count
        train = _slice_rows(combined_train, 0, split)
        audit = _slice_rows(combined_train, split, len(combined_train["file_names"]))
        audit_sources = sources[split:]
        sources = sources[:split]
    else:
        if not args.audit_traffic_rows:
            raise ValueError("train_audit selection requires --audit-traffic-rows")
        train = combined_train
        audit_artifact = torch.load(args.audit_traffic_rows, map_location="cpu")
        _, _, audit_sources, audit = _align_traffic_rows(
            policy_rows, audit_artifact["rows"]
        )
    metrics = json.loads(Path(args.policy_metrics).read_text(encoding="utf-8"))
    policy = metrics["object_intent_deployment_gate_fit"]["online"]["action"]
    train_logits = _apply_policy(train["_policy"], policy)
    threshold = torch.tensor(metrics["online"]["thresholds"]["image"][:4])
    threshold_logit = torch.logit(threshold.clamp(1e-5, 1 - 1e-5))
    train_margin = train_logits - threshold_logit[None]
    train_target = train["action_target"].numpy().astype(bool)
    source_names = sorted(set(sources))
    source_map = {name: index for index, name in enumerate(source_names)}
    source_ids = np.array([source_map[name] for name in sources])

    full_features = _features(train, train_margin)
    object_track_store = torch.load(args.object_track_store, map_location="cpu")
    full_features = np.concatenate((
        full_features, _object_track_features(object_track_store, train["file_names"])
    ), axis=1)
    margin_features = _margin_features(train_margin)
    full_oof, full_models = _fit_oof(
        full_features, train_margin.numpy(), train_target, source_ids, args.folds, args.seed,
        args.fold_mode,
    )
    margin_oof, margin_models = _fit_oof(
        margin_features, train_margin.numpy(), train_target, source_ids, args.folds, args.seed,
        args.fold_mode,
    )
    full_rules = [_select_rule(
        train_margin[:, action].numpy(), full_oof[:, action], train_target[:, action], source_ids,
    ) for action in range(4)]
    margin_rules = [_select_rule(
        train_margin[:, action].numpy(), margin_oof[:, action], train_target[:, action], source_ids,
    ) for action in range(4)]
    full_oof_pred, full_oof_mask = _apply_rules(train_margin.numpy(), full_oof, full_rules)
    margin_oof_pred, _ = _apply_rules(train_margin.numpy(), margin_oof, margin_rules)

    audit_logits = _apply_policy(audit["_policy"], policy)
    audit_margin = audit_logits - threshold_logit[None]
    audit_target = audit["action_target"].numpy().astype(bool)
    audit_features = np.concatenate((
        _features(audit, audit_margin),
        _object_track_features(object_track_store, audit["file_names"]),
    ), axis=1)
    audit_full_probability = _ensemble_probability(full_models, audit_features)
    audit_margin_probability = _ensemble_probability(
        margin_models, _margin_features(audit_margin)
    )
    audit_source_map = {
        name: index for index, name in enumerate(sorted(set(audit_sources)))
    }
    audit_source_ids = np.array([audit_source_map[name] for name in audit_sources])
    deployment_rules = [_select_rule(
        audit_margin[:, action].numpy(), audit_full_probability[:, action],
        audit_target[:, action], audit_source_ids,
    ) for action in range(4)]
    deployment_margin_rules = [_select_rule(
        audit_margin[:, action].numpy(), audit_margin_probability[:, action],
        audit_target[:, action], audit_source_ids,
    ) for action in range(4)]
    audit_full_prediction, audit_full_mask = _apply_rules(
        audit_margin.numpy(), audit_full_probability, deployment_rules
    )
    audit_margin_prediction, audit_margin_mask = _apply_rules(
        audit_margin.numpy(), audit_margin_probability, deployment_margin_rules
    )
    audit_baseline = audit_margin.numpy() >= 0
    action_gate, audit_action_rows = _audit_action_gate(
        audit_baseline, audit_full_prediction, audit_margin_prediction,
        audit_full_mask, audit_target,
    )
    deployment_routes, deployment_route_rows = _select_deployment_routes(
        audit_baseline, audit_margin_prediction, audit_full_prediction, audit_target,
    )
    no_traffic_routes, _ = _select_deployment_routes(
        audit_baseline, audit_margin_prediction, audit_baseline, audit_target,
    )

    test_dir = Path(args.test_epoch_dir)
    test_logits = torch.load(test_dir / "video_action_test.pt", map_location="cpu").float()
    test_target = torch.load(test_dir / "action_target_test.pt", map_location="cpu").numpy().astype(bool)
    test_margin = test_logits - threshold_logit[None]
    test_rows = {
        key: torch.load(test_dir / f"{key}_test.pt", map_location="cpu") for key in (
            "traffic_trajectory_order_delta", "traffic_trajectory_support",
            "trajectory_state_strength", "trajectory_interaction_risk",
            "traffic_trajectory_state_features",
        )
    }
    test_file_names = json.loads((test_dir / "file_names_test.json").read_text(encoding="utf-8"))
    test_full = np.concatenate((
        _features(test_rows, test_margin),
        _object_track_features(object_track_store, test_file_names),
    ), axis=1)
    test_margin_features = _margin_features(test_margin)
    full_probability = _ensemble_probability(full_models, test_full)
    margin_probability = _ensemble_probability(margin_models, test_margin_features)
    traffic_prediction, traffic_mask = _apply_rules(
        test_margin.numpy(), full_probability, deployment_rules
    )
    margin_prediction, margin_mask = _apply_rules(
        test_margin.numpy(), margin_probability, deployment_margin_rules
    )
    shuffled = np.random.default_rng(args.seed).permutation(len(test_full))
    shuffled_full = _shuffle_traffic_features(
        test_full, shuffled, margin_dim=test_margin_features.shape[1],
    )
    shuffle_probability = _ensemble_probability(full_models, shuffled_full)
    shuffle_traffic_prediction, _ = _apply_rules(
        test_margin.numpy(), shuffle_probability, deployment_rules
    )
    baseline = test_margin.numpy() >= 0
    full_prediction = _route_predictions(
        deployment_routes, baseline, margin_prediction, traffic_prediction,
    )
    no_traffic_prediction = _route_predictions(
        no_traffic_routes, baseline, margin_prediction, baseline,
    )
    shuffle_prediction = _route_predictions(
        deployment_routes, baseline, margin_prediction, shuffle_traffic_prediction,
    )
    full_mask = np.zeros_like(baseline)
    traffic_deployed_mask = np.zeros_like(baseline)
    for action, route in enumerate(deployment_routes):
        if route == "margin_only":
            full_mask[:, action] = margin_mask[:, action]
        elif route == "traffic":
            full_mask[:, action] = traffic_mask[:, action]
            traffic_deployed_mask[:, action] = traffic_mask[:, action]
    corrected = np.logical_and(full_mask, full_prediction == test_target)
    damaged = np.logical_and(full_mask, baseline == test_target)
    traffic_corrected = np.logical_and(
        traffic_deployed_mask, full_prediction == test_target,
    )
    traffic_damaged = np.logical_and(
        traffic_deployed_mask, no_traffic_prediction == test_target,
    )
    traffic_rank_metrics = []
    for action in range(test_target.shape[1]):
        full_rank = _rank_metrics(full_probability[:, action], test_target[:, action])
        margin_rank = _rank_metrics(margin_probability[:, action], test_target[:, action])
        traffic_rank_metrics.append({
            "full": full_rank,
            "margin_only": margin_rank,
            "traffic_ap_increment": full_rank["average_precision"] - margin_rank["average_precision"],
            "traffic_auc_increment": full_rank["roc_auc"] - margin_rank["roc_auc"],
        })
    traffic_risk_coverage = {
        str(action): _risk_coverage_curve(
            baseline[:, action], full_probability[:, action], test_target[:, action],
        ) for action, route in enumerate(deployment_routes) if route == "traffic"
    }
    payload = {
        "test_labels_used_for_fit_or_selection": False,
        "method": "traffic_selective_correction",
        "traffic_feature_dim": int(full_features.shape[1]),
        "object_track_schema": str(object_track_store.get("schema", "unknown")),
        "fold_mode": args.fold_mode,
        "selection_cohort": args.selection_cohort,
        "source_names": source_names,
        "audit_source_names": sorted(set(audit_sources)),
        "audit_action_gate": action_gate.tolist(),
        "audit_action_rows": audit_action_rows,
        "deployment_routes": deployment_routes,
        "deployment_route_rows": deployment_route_rows,
        "no_traffic_routes": no_traffic_routes,
        "rules": full_rules,
        "margin_only_rules": margin_rules,
        "deployment_rules_from_selection_cohort": deployment_rules,
        "deployment_margin_rules_from_selection_cohort": deployment_margin_rules,
        "oof": {
            "base_Act_mF1": _macro_f1(torch.from_numpy(train_margin.numpy() >= 0), torch.from_numpy(train_target.astype(np.float32))),
            "full_Act_mF1": _macro_f1(torch.from_numpy(full_oof_pred), torch.from_numpy(train_target.astype(np.float32))),
            "margin_only_Act_mF1": _macro_f1(torch.from_numpy(margin_oof_pred), torch.from_numpy(train_target.astype(np.float32))),
            "traffic_incremental_mf1": _macro_f1(torch.from_numpy(full_oof_pred), torch.from_numpy(train_target.astype(np.float32))) - _macro_f1(torch.from_numpy(margin_oof_pred), torch.from_numpy(train_target.astype(np.float32))),
            "correction_rate": float(full_oof_mask.mean()),
        },
        "audit": {
            "base_Act_mF1": _macro_f1(torch.from_numpy(audit_baseline), torch.from_numpy(audit_target.astype(np.float32))),
            "ungated_full_Act_mF1": _macro_f1(torch.from_numpy(audit_full_prediction), torch.from_numpy(audit_target.astype(np.float32))),
            "margin_only_Act_mF1": _macro_f1(torch.from_numpy(audit_margin_prediction), torch.from_numpy(audit_target.astype(np.float32))),
            "base_per_action_f1": _per_action_f1(audit_baseline, audit_target),
            "full_per_action_f1": _per_action_f1(audit_full_prediction, audit_target),
            "margin_only_per_action_f1": _per_action_f1(audit_margin_prediction, audit_target),
        },
        "test": {
            "base_Act_mF1": _macro_f1(torch.from_numpy(baseline), torch.from_numpy(test_target.astype(np.float32))),
            "full_Act_mF1": _macro_f1(torch.from_numpy(full_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "margin_only_Act_mF1": _macro_f1(torch.from_numpy(margin_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "no_traffic_deploy_Act_mF1": _macro_f1(torch.from_numpy(no_traffic_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "traffic_shuffle_Act_mF1": _macro_f1(torch.from_numpy(shuffle_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "traffic_incremental_mf1": _macro_f1(torch.from_numpy(full_prediction), torch.from_numpy(test_target.astype(np.float32))) - _macro_f1(torch.from_numpy(no_traffic_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "correction_rate": float(full_mask.mean()),
            "correction_precision": float(corrected.sum() / max(1, full_mask.sum())),
            "errors_recovered": int(corrected.sum()),
            "correct_damaged": int(damaged.sum()),
            "net_corrected": int(corrected.sum() - damaged.sum()),
            "net_corrected_by_action": (corrected.sum(0) - damaged.sum(0)).tolist(),
            "traffic_correction_rate": float(traffic_deployed_mask.mean()),
            "traffic_correction_precision": float(
                traffic_corrected.sum() / max(1, traffic_deployed_mask.sum())
            ),
            "traffic_errors_recovered": int(traffic_corrected.sum()),
            "traffic_correct_damaged": int(traffic_damaged.sum()),
            "traffic_net_corrected": int(
                traffic_corrected.sum() - traffic_damaged.sum()
            ),
            "traffic_rank_metrics_by_action": traffic_rank_metrics,
            "traffic_risk_coverage_by_action": traffic_risk_coverage,
            "base_per_action_f1": _per_action_f1(baseline, test_target),
            "full_per_action_f1": _per_action_f1(full_prediction, test_target),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traffic_selective_corrector_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump({
        "full_models": full_models, "margin_models": margin_models,
        "rules": full_rules, "margin_rules": margin_rules,
        "deployment_rules": deployment_rules,
        "deployment_margin_rules": deployment_margin_rules,
        "deployment_routes": deployment_routes,
        "no_traffic_routes": no_traffic_routes,
        "selection_cohort": args.selection_cohort,
        "threshold": threshold,
    }, output / "traffic_selective_corrector.joblib")
    torch.save({
        "base_prediction": torch.from_numpy(baseline),
        "full_prediction": torch.from_numpy(full_prediction),
        "margin_prediction": torch.from_numpy(margin_prediction),
        "no_traffic_deploy_prediction": torch.from_numpy(no_traffic_prediction),
        "traffic_shuffle_prediction": torch.from_numpy(shuffle_prediction),
        "correction_mask": torch.from_numpy(full_mask),
        "full_probability": torch.from_numpy(full_probability),
        "margin_probability": torch.from_numpy(margin_probability),
        "traffic_deployed_mask": torch.from_numpy(traffic_deployed_mask),
        "action_target": torch.from_numpy(test_target),
        "file_names": test_file_names,
    }, output / "traffic_selective_corrector_test.pt")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
