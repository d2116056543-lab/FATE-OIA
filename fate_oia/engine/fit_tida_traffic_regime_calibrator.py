from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fate_oia.engine.fit_tida_selective_traffic_corrector import (
    _action_f1,
    _align_traffic_rows,
    _apply_policy,
    _features,
    _macro_f1,
    _object_track_features,
    _per_action_f1,
)


def _compact_traffic_features(
    rows: dict[str, Any], names: list[str], track_store: dict[str, Any],
) -> np.ndarray:
    zero_margin = torch.zeros(len(names), 4)
    summary = _features(rows, zero_margin)[:, 8:]
    tracks = _object_track_features(track_store, names)
    return np.concatenate((summary, tracks), axis=1)


def _best_threshold(margin: np.ndarray, target: np.ndarray) -> float:
    if len(margin) < 32 or target.min() == target.max():
        return 0.0
    candidates = np.unique(np.concatenate((
        np.linspace(-0.6, 0.6, 49),
        np.quantile(margin, np.linspace(0.05, 0.95, 19)),
    )))
    scores = np.array([_action_f1(margin >= value, target) for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def _cluster_thresholds(
    margin: np.ndarray, target: np.ndarray, cluster: np.ndarray, cluster_count: int,
) -> np.ndarray:
    result = np.zeros((cluster_count, target.shape[1]), dtype=np.float32)
    for group in range(cluster_count):
        index = cluster == group
        for action in range(target.shape[1]):
            result[group, action] = _best_threshold(margin[index, action], target[index, action])
    return result


def _predict(
    margin: np.ndarray, cluster: np.ndarray, thresholds: np.ndarray,
    shrink: float, cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.clip(thresholds[cluster] * shrink, -cap, cap)
    return margin >= boundary, boundary


def _per_source_worst_delta(
    baseline: np.ndarray, final: np.ndarray, target: np.ndarray, sources: list[str],
) -> float:
    values = []
    source_array = np.asarray(sources)
    for source in sorted(set(sources)):
        index = source_array == source
        values.append(
            _macro_f1(torch.from_numpy(final[index]), torch.from_numpy(target[index].astype(np.float32)))
            - _macro_f1(torch.from_numpy(baseline[index]), torch.from_numpy(target[index].astype(np.float32)))
        )
    return float(min(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", required=True)
    parser.add_argument("--traffic-rows", required=True)
    parser.add_argument("--audit-traffic-rows", required=True)
    parser.add_argument("--object-track-store", required=True)
    parser.add_argument("--policy-metrics", required=True)
    parser.add_argument("--test-epoch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    policy_rows = torch.load(args.policy_rows, map_location="cpu")
    traffic = torch.load(args.traffic_rows, map_location="cpu")["rows"]
    audit_traffic = torch.load(args.audit_traffic_rows, map_location="cpu")["rows"]
    track_store = torch.load(args.object_track_store, map_location="cpu")
    _, _, train_sources, train = _align_traffic_rows(policy_rows, traffic)
    _, _, audit_sources, audit = _align_traffic_rows(policy_rows, audit_traffic)
    metrics = json.loads(Path(args.policy_metrics).read_text(encoding="utf-8"))
    policy = metrics["object_intent_deployment_gate_fit"]["online"]["action"]
    threshold = torch.tensor(metrics["online"]["thresholds"]["image"][:4])
    threshold_logit = torch.logit(threshold.clamp(1e-5, 1 - 1e-5))
    train_margin = (_apply_policy(train["_policy"], policy) - threshold_logit[None]).numpy()
    audit_margin = (_apply_policy(audit["_policy"], policy) - threshold_logit[None]).numpy()
    train_target = train["action_target"].numpy().astype(bool)
    audit_target = audit["action_target"].numpy().astype(bool)
    train_feature = _compact_traffic_features(train, train["file_names"], track_store)
    audit_feature = _compact_traffic_features(audit, audit["file_names"], track_store)
    scaler = StandardScaler().fit(train_feature)
    pca = PCA(n_components=24, whiten=True, random_state=args.seed).fit(
        scaler.transform(train_feature)
    )
    train_embedding = pca.transform(scaler.transform(train_feature))
    audit_embedding = pca.transform(scaler.transform(audit_feature))
    audit_baseline = audit_margin >= 0
    candidates = []
    fitted = {}
    for clusters in (2, 4, 8, 16):
        model = MiniBatchKMeans(
            n_clusters=clusters, batch_size=1024, n_init=10,
            random_state=args.seed,
        ).fit(train_embedding)
        train_cluster = model.predict(train_embedding)
        audit_cluster = model.predict(audit_embedding)
        boundaries = _cluster_thresholds(train_margin, train_target, train_cluster, clusters)
        fitted[clusters] = (model, boundaries)
        for shrink in (0.25, 0.50, 0.75, 1.0):
            for cap in (0.10, 0.20, 0.35):
                prediction, used = _predict(audit_margin, audit_cluster, boundaries, shrink, cap)
                score = _macro_f1(
                    torch.from_numpy(prediction), torch.from_numpy(audit_target.astype(np.float32))
                )
                candidates.append({
                    "clusters": clusters, "shrink": shrink, "cap": cap,
                    "audit_Act_mF1": score,
                    "audit_per_action_f1": _per_action_f1(prediction, audit_target),
                    "audit_worst_source_delta": _per_source_worst_delta(
                        audit_baseline, prediction, audit_target, audit_sources
                    ),
                    "boundary_rms": float(np.sqrt(np.mean(used ** 2))),
                })
    safe = [row for row in candidates if row["audit_worst_source_delta"] >= -0.005]
    selected = max(safe or candidates, key=lambda row: row["audit_Act_mF1"])
    cluster_model, cluster_boundaries = fitted[selected["clusters"]]

    test_dir = Path(args.test_epoch_dir)
    test_logits = torch.load(test_dir / "video_action_test.pt", map_location="cpu").float()
    test_target = torch.load(test_dir / "action_target_test.pt", map_location="cpu").numpy().astype(bool)
    test_margin = (test_logits - threshold_logit[None]).numpy()
    test_names = json.loads((test_dir / "file_names_test.json").read_text(encoding="utf-8"))
    test_rows = {
        key: torch.load(test_dir / f"{key}_test.pt", map_location="cpu") for key in (
            "traffic_trajectory_order_delta", "traffic_trajectory_support",
            "trajectory_state_strength", "trajectory_interaction_risk",
            "traffic_trajectory_state_features",
        )
    }
    test_feature = _compact_traffic_features(test_rows, test_names, track_store)
    test_embedding = pca.transform(scaler.transform(test_feature))
    test_cluster = cluster_model.predict(test_embedding)
    test_prediction, test_boundary = _predict(
        test_margin, test_cluster, cluster_boundaries,
        selected["shrink"], selected["cap"],
    )
    audit_cluster = cluster_model.predict(audit_embedding)
    audit_prediction, _ = _predict(
        audit_margin, audit_cluster, cluster_boundaries,
        selected["shrink"], selected["cap"],
    )
    action_gate = np.array([
        _action_f1(audit_prediction[:, action], audit_target[:, action])
        > _action_f1(audit_baseline[:, action], audit_target[:, action])
        for action in range(4)
    ])
    test_baseline = test_margin >= 0
    test_prediction = np.where(action_gate[None], test_prediction, test_baseline)
    shuffled = np.random.default_rng(args.seed).permutation(len(test_embedding))
    shuffled_cluster = cluster_model.predict(test_embedding[shuffled])
    shuffled_prediction, _ = _predict(
        test_margin, shuffled_cluster, cluster_boundaries,
        selected["shrink"], selected["cap"],
    )
    shuffled_prediction = np.where(action_gate[None], shuffled_prediction, test_baseline)
    payload = {
        "test_labels_used_for_fit_or_selection": False,
        "method": "ego_compensated_traffic_regime_calibration",
        "selected": selected,
        "action_gate_from_train_audit": action_gate.tolist(),
        "candidate_count": len(candidates),
        "audit": {
            "base_Act_mF1": _macro_f1(torch.from_numpy(audit_baseline), torch.from_numpy(audit_target.astype(np.float32))),
            "selected_Act_mF1": _macro_f1(torch.from_numpy(audit_prediction), torch.from_numpy(audit_target.astype(np.float32))),
        },
        "test": {
            "base_Act_mF1": _macro_f1(torch.from_numpy(test_baseline), torch.from_numpy(test_target.astype(np.float32))),
            "selected_Act_mF1": _macro_f1(torch.from_numpy(test_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "traffic_shuffle_Act_mF1": _macro_f1(torch.from_numpy(shuffled_prediction), torch.from_numpy(test_target.astype(np.float32))),
            "base_per_action_f1": _per_action_f1(test_baseline, test_target),
            "selected_per_action_f1": _per_action_f1(test_prediction, test_target),
            "boundary_rms": float(np.sqrt(np.mean(test_boundary ** 2))),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traffic_regime_calibrator_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump({
        "scaler": scaler, "pca": pca, "cluster_model": cluster_model,
        "cluster_boundaries": cluster_boundaries, "selected": selected,
        "action_gate": action_gate, "threshold": threshold,
    }, output / "traffic_regime_calibrator.joblib")
    torch.save({
        "base_prediction": torch.from_numpy(test_baseline),
        "selected_prediction": torch.from_numpy(test_prediction),
        "shuffle_prediction": torch.from_numpy(shuffled_prediction),
        "traffic_cluster": torch.from_numpy(test_cluster),
        "traffic_boundary": torch.from_numpy(test_boundary),
        "action_target": torch.from_numpy(test_target),
    }, output / "traffic_regime_calibrator_test.pt")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
