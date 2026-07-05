from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fate_oia.acpr_interactflow.artifacts import append_jsonl, save_epoch_tensors, write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel
from fate_oia.acpr_interactflow.psi_damo_dataset import PSIDAMO11902Dataset, psi_interactflow_collate
from fate_oia.acpr_interactflow.psi_metrics import compute_psi_action_metrics, compute_psi_exp29_metrics


def _mean_dict(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float, bool))})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
        if values:
            out[key] = sum(values) / len(values)
    return out


def _threshold_sweep(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict:
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    best = {"threshold": 0.50, "Exp_mF1": -1.0, "Exp_oF1": -1.0}
    for threshold in thresholds:
        metrics = compute_psi_exp29_metrics(logits, labels, mask, threshold=threshold)
        row = {"threshold": threshold, "Exp_mF1": metrics["Exp_mF1"], "Exp_oF1": metrics["Exp_oF1"]}
        rows.append(row)
        if row["Exp_mF1"] > best["Exp_mF1"]:
            best = row
    return {"diagnostic_only": True, "thresholds": rows, "best_global": best}


@torch.no_grad()
def evaluate(model: ACPRInteractFlowPPModel, loader: DataLoader, device: torch.device, output_dir: str | Path | None = None, epoch: int = 0) -> dict:
    model.eval()
    action_logits, exp_logits, exp_logits_calibrated, action_labels, action_soft, exp_labels, exp_mask = [], [], [], [], [], [], []
    global_logits, flow_delta_logits, calibration_delta, visual_logits, motion_logits, predicate_logits = [], [], [], [], [], []
    gated_state_contrib, benefit_gate, ledger_gate = [], [], []
    innovation_rows: list[dict] = []
    file_names: list[str] = []
    progress_path = Path(output_dir) / "eval_progress.jsonl" if output_dir is not None else None
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    for batch_idx, batch in enumerate(loader, start=1):
        frames = batch.input_frames.to(device, non_blocking=True)
        out = model(frames, epoch=epoch)
        action_logits.append(out.action_logits.detach().cpu())
        exp_logits.append(out.exp29_logits.detach().cpu())
        exp_logits_calibrated.append(out.aux.get("exp29_logits_calibrated", out.exp29_logits).detach().cpu())
        global_logits.append(out.ledger.global_logits.detach().cpu())
        flow_delta_logits.append(out.ledger.flow_delta_logits.detach().cpu())
        calibration_delta.append(out.ledger.calibration_delta.detach().cpu())
        visual_logits.append(out.ledger.visual_logits.detach().cpu())
        motion_logits.append(out.ledger.motion_logits.detach().cpu())
        predicate_logits.append(out.ledger.predicate_logits.detach().cpu())
        gated_state_contrib.append(out.ledger.gated_state_contributions.detach().cpu())
        benefit_gate.append(out.ledger.benefit_gate.detach().cpu())
        ledger_gate.append(out.ledger.gate.detach().cpu())
        exp_prob_raw = torch.sigmoid(out.exp29_logits)
        exp_prob_cal = torch.sigmoid(out.aux.get("exp29_logits_calibrated", out.exp29_logits))
        state_group_logits = out.aux.get("state_group_logits")
        state_layer_weights = out.aux.get("state_layer_weights")
        innovation_rows.append(
            {
                "epoch": epoch,
                "batch": batch_idx,
                "predicate_positive_rate": float(out.predicates.temporal_stats.get("predicate_positive_rate", 0.0)),
                "predicate_confidence_mean": float(out.predicates.predicate_confidence.detach().mean().cpu()),
                "predicate_confidence_max": float(out.predicates.predicate_confidence.detach().max().cpu()),
                "predicate_attention_entropy": float(out.predicates.temporal_stats.get("attention_entropy", 0.0)),
                "predicate_transfer_gate_mean": float(out.predicates.transfer_gate.detach().mean().cpu()),
                "predicate_corridor_mass_mean": float(out.predicates.predicate_corridor_mass.detach().mean().cpu()),
                "predicate_centroid_x_mean": float(out.predicates.predicate_centroids[..., 0].detach().mean().cpu()),
                "predicate_centroid_y_mean": float(out.predicates.predicate_centroids[..., 1].detach().mean().cpu()),
                "state_group_logits_mean": float(state_group_logits.detach().mean().cpu()) if state_group_logits is not None else 0.0,
                "state_group_logits_std": float(state_group_logits.detach().std().cpu()) if state_group_logits is not None else 0.0,
                "state_layer_weights_max": float(state_layer_weights.detach().max().cpu()) if state_layer_weights is not None else 0.0,
                "flow_edge_abs_mean": float(out.flow.flow_edges.detach().abs().mean().cpu()),
                "flow_edge_abs_max": float(out.flow.flow_edges.detach().abs().max().cpu()),
                "factor_attention_entropy": float(out.flow.stats.get("factor_attention_entropy", 0.0)),
                "lag_argmax_mean": float(out.flow.stats.get("lag_argmax_mean", 0.0)),
                "ledger_gate_mean": float(out.ledger.gate.detach().mean().cpu()),
                "ledger_benefit_gate_mean": float(out.ledger.benefit_gate.detach().mean().cpu()),
                "ledger_flow_delta_abs_mean": float(out.ledger.flow_delta_logits.detach().abs().mean().cpu()),
                "ledger_calibration_delta_abs_mean": float(out.ledger.calibration_delta.detach().abs().mean().cpu()),
                "ledger_identity_error": float(out.ledger.identity_error.detach().cpu()),
                "exp29_raw_prob_mean": float(exp_prob_raw.detach().mean().cpu()),
                "exp29_raw_prob_max": float(exp_prob_raw.detach().max().cpu()),
                "exp29_calibrated_prob_mean": float(exp_prob_cal.detach().mean().cpu()),
                "exp29_calibrated_prob_max": float(exp_prob_cal.detach().max().cpu()),
                "exp29_raw_pred_positive_rate_0p5": float((exp_prob_raw >= 0.5).float().mean().detach().cpu()),
                "exp29_calibrated_pred_positive_rate_0p5": float((exp_prob_cal >= 0.5).float().mean().detach().cpu()),
            }
        )
        action_labels.append(batch.action_majority.detach().cpu())
        action_soft.append(batch.action_soft.detach().cpu())
        exp_labels.append(batch.exp29.detach().cpu())
        exp_mask.append(batch.exp29_mask.detach().cpu())
        file_names.extend(batch.sample_id)
        if progress_path is not None and (batch_idx == 1 or batch_idx % 50 == 0 or (total_batches is not None and batch_idx == total_batches)):
            append_jsonl(
                progress_path,
                {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "phase": "eval_test",
                },
            )
            print(f"interactflow_eval epoch={epoch} batch={batch_idx}/{total_batches}", flush=True)
    al = torch.cat(action_logits)
    el = torch.cat(exp_logits)
    ecal = torch.cat(exp_logits_calibrated)
    ay = torch.cat(action_labels)
    asoft = torch.cat(action_soft)
    ey = torch.cat(exp_labels)
    em = torch.cat(exp_mask)
    action = compute_psi_action_metrics(al, ay, asoft)
    exp_raw = compute_psi_exp29_metrics(el, ey, em)
    exp_calibrated = compute_psi_exp29_metrics(ecal, ey, em)
    exp_diag = _threshold_sweep(ecal, ey, em)
    exp = exp_calibrated
    joint = 0.60 * action["Act_mAcc"] + 0.25 * action["Act_stopF1"] + 0.15 * exp["Exp_mF1"]
    innovation = _mean_dict(innovation_rows)
    metrics = {
        "epoch": epoch,
        "joint": joint,
        "Act_mAcc": action["Act_mAcc"],
        "Act_oAcc": action["Act_oAcc"],
        "Exp_mF1": exp["Exp_mF1"],
        "Exp_oF1": exp["Exp_oF1"],
        "Exp_mAP": exp["Exp_mAP"],
        "ExpRaw_mF1": exp_raw["Exp_mF1"],
        "ExpRaw_oF1": exp_raw["Exp_oF1"],
        "ExpCal_mF1": exp_calibrated["Exp_mF1"],
        "ExpCal_oF1": exp_calibrated["Exp_oF1"],
        "action": action,
        "exp29": exp,
        "exp29_raw_fixed": exp_raw,
        "exp29_calibrated_fixed": exp_calibrated,
        "exp29_diagnostic_threshold_sweep": exp_diag,
        "exp29_primary": "calibrated_fixed",
        "innovation": innovation,
    }
    if output_dir is not None:
        save_epoch_tensors(
            output_dir,
            "test",
            al,
            el,
            ay,
            ey,
            file_names,
            extra_tensors={
                "logits_exp29_calibrated": ecal,
                "logits_action_global": torch.cat(global_logits),
                "logits_action_visual": torch.cat(visual_logits),
                "logits_action_motion": torch.cat(motion_logits),
                "logits_action_predicate": torch.cat(predicate_logits),
                "logits_action_flow_delta": torch.cat(flow_delta_logits),
                "logits_action_calibration_delta": torch.cat(calibration_delta),
                "ledger_gated_state_contributions": torch.cat(gated_state_contrib),
                "ledger_benefit_gate": torch.cat(benefit_gate),
                "ledger_gate": torch.cat(ledger_gate),
            },
        )
        append_jsonl(Path(output_dir) / "innovation_intermediate_metrics.jsonl", {"epoch": epoch, **innovation})
        write_json(Path(output_dir) / "innovation_intermediate_latest.json", {"epoch": epoch, **innovation})
        write_json(Path(output_dir) / "exp29_diagnostic_threshold_sweep.json", exp_diag)
        write_json(Path(output_dir) / "metrics_latest.json", metrics)
    return metrics


def build_eval_loader(cfg: dict, max_test_samples: int | None = None) -> DataLoader:
    data = cfg["data"]
    paths = cfg["paths"]
    protocol_index_cfg = data.get("protocol_index", {})
    protocol_index_enabled = bool(protocol_index_cfg.get("enabled", False))
    ds = PSIDAMO11902Dataset(
        paths["psi_package_root"],
        "test",
        frames_root=paths.get("psi2_root_reference_only"),
        image_size=(int(data["image_height"]), int(data["image_width"])),
        action_dim=int(data["action_dim"]),
        strict_counts=bool(data.get("strict_counts", True)) and max_test_samples is None,
        max_samples=max_test_samples,
        max_sample_strategy=str(data.get("eval_max_sample_strategy", data.get("max_sample_strategy", "head"))),
        max_sample_seed=int(data.get("max_sample_seed", 7)) + 1000,
        frame_protocol=str(data.get("eval_frame_protocol", data.get("frame_protocol", "recorded_observed"))),
        allow_target_frame_in_input=bool(data.get("allow_target_frame_in_input", False)),
        exp_supervision_policy=str(data.get("eval_exp_supervision_policy", data.get("exp_supervision_policy", "record_mask"))),
        exp_near_keyframe_max_gap=int(data.get("exp_near_keyframe_max_gap", 30)),
        use_decision_group_weight=False,
        protocol_index_dir=protocol_index_cfg.get("dir") if protocol_index_enabled else None,
        protocol_name=protocol_index_cfg.get("name") if protocol_index_enabled else None,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=psi_interactflow_collate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_test_samples", type=int)
    args = parser.parse_args()
    cfg = load_interactflow_config(args.config)
    pred_cfg = cfg["model"].get("predicates", {})
    visual_cfg = cfg["model"].get("visual_encoder", {})
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
        oia_acpr_checkpoint=cfg["paths"].get("oia_acpr_checkpoint"),
        text_encoder_model=cfg["paths"].get("text_encoder_model"),
        require_oia_transfer_source=bool(pred_cfg.get("require_oia_transfer_source", False)),
        require_transformer_text=bool(pred_cfg.get("require_transformer_text", False)),
        action_dim=int(cfg["data"]["action_dim"]),
        dino_chunk_size=int(visual_cfg.get("dino_chunk_size", 2)),
        anchor_frames=tuple(int(x) for x in visual_cfg.get("anchor_frames", [0, 3, 6, 9, 12, 14])),
        selected_layers=tuple(int(x) for x in visual_cfg.get("selected_layers", [3, 7, 11])),
        dino_input_height=int(visual_cfg.get("dino_input_height", cfg["data"].get("image_height", 320))),
        dino_input_width=int(visual_cfg.get("dino_input_width", cfg["data"].get("image_width", 576))),
        patch_size=int(cfg["data"].get("patch_size", 8)),
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
    loader = build_eval_loader(cfg, max_test_samples=args.max_test_samples)
    metrics = evaluate(model, loader, device, args.output_dir)
    write_json(Path(args.output_dir) / "eval_metrics.json", metrics)


if __name__ == "__main__":
    main()
