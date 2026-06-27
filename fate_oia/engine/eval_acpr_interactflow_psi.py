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


@torch.no_grad()
def evaluate(model: ACPRInteractFlowPPModel, loader: DataLoader, device: torch.device, output_dir: str | Path | None = None, epoch: int = 0) -> dict:
    model.eval()
    action_logits, exp_logits, exp_logits_calibrated, action_labels, action_soft, exp_labels, exp_mask = [], [], [], [], [], [], []
    global_logits, flow_delta_logits, calibration_delta, visual_logits, motion_logits, predicate_logits = [], [], [], [], [], []
    gated_state_contrib, benefit_gate, ledger_gate = [], [], []
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
    exp = exp_calibrated
    joint = 0.60 * action["Act_mAcc"] + 0.25 * action["Act_stopF1"] + 0.15 * exp["Exp_mF1"]
    metrics = {
        "epoch": epoch,
        "joint": joint,
        "action": action,
        "exp29": exp,
        "exp29_raw_fixed": exp_raw,
        "exp29_calibrated_fixed": exp_calibrated,
        "exp29_primary": "calibrated_fixed",
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
        write_json(Path(output_dir) / "metrics_latest.json", metrics)
    return metrics


def build_eval_loader(cfg: dict, max_test_samples: int | None = None) -> DataLoader:
    data = cfg["data"]
    paths = cfg["paths"]
    ds = PSIDAMO11902Dataset(
        paths["psi_package_root"],
        "test",
        frames_root=paths.get("psi2_root_reference_only"),
        image_size=(int(data["image_height"]), int(data["image_width"])),
        action_dim=int(data["action_dim"]),
        strict_counts=max_test_samples is None,
        max_samples=max_test_samples,
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
        dino_chunk_size=int(cfg["model"]["visual_encoder"].get("dino_chunk_size", 2)),
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
    loader = build_eval_loader(cfg, max_test_samples=args.max_test_samples)
    metrics = evaluate(model, loader, device, args.output_dir)
    write_json(Path(args.output_dir) / "eval_metrics.json", metrics)


if __name__ == "__main__":
    main()
