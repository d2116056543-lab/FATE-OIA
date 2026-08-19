from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import Subset

from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.engine.train_aie_oia import make_dataset, make_loader
from fate_oia.engine.train_vetra_strong_refine import (
    base_forward,
    build_refiner,
    collect,
    load_base,
    run_refiner,
    select_gains,
)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.vetra_strong_rank_losses import (
    action_pairwise_ap_loss,
    action_smooth_ap_loss,
    base_margin_trust_loss,
    residual_energy_loss,
)
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.utils.acpr_threshold_search import search_best_thresholds_for_f1
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.vetra_stage_contracts import (
    atomic_write_json,
    sha256_file,
    validate_stage_checkpoint,
)


def freeze_base_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def verify_reason_identity(source: Tensor, candidate: Tensor) -> None:
    if source.shape != candidate.shape or not torch.equal(
        source.detach(), candidate.detach()
    ):
        raise RuntimeError("Stage B reason identity was violated")


def choose_refiner_candidate(
    rows: list[dict],
    *,
    min_mf1_gain: float,
    max_map_drop: float,
) -> dict:
    if not rows or rows[0].get("name") != "base":
        raise ValueError("candidate rows must begin with the untouched base")
    base = rows[0]
    eligible = [
        row
        for row in rows[1:]
        if float(row["audit_mf1"]) >= float(base["audit_mf1"]) + min_mf1_gain
        and float(row["audit_map"]) >= float(base["audit_map"]) - max_map_drop
    ]
    if not eligible:
        return {**base, "refiner_selected": False}
    selected = max(
        eligible,
        key=lambda row: (float(row["audit_mf1"]), float(row["audit_map"])),
    )
    return {**selected, "refiner_selected": True}


def make_stage_b_checkpoint(
    *,
    parent_path: str | Path,
    identity: dict,
    refiner_selected: bool,
    refiner_state: dict | None,
    deployment_gain: Tensor,
    selection: dict,
) -> dict:
    if refiner_selected and refiner_state is None:
        raise ValueError("selected refiner requires a state dictionary")
    if not refiner_selected:
        refiner_state = None
    return {
        "stage": "action_refined",
        "run_identity": dict(identity),
        "parent_checkpoint": str(Path(parent_path).resolve()),
        "parent_checkpoint_sha256": sha256_file(parent_path),
        "refiner_selected": bool(refiner_selected),
        "refiner": refiner_state,
        "deployment_gain": deployment_gain.detach().cpu(),
        "selection": dict(selection),
        "manifest": {"external_task_checkpoint": None},
    }


def _write_json(path: Path, payload) -> None:
    atomic_write_json(path, payload)


def _append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _audit_row(name: str, calib: dict, audit: dict, gain: Tensor) -> dict:
    gain = gain.float().view(1, 4)
    calib_logits = calib["base_action"] + calib["delta"] * gain
    audit_logits = audit["base_action"] + audit["delta"] * gain
    grid = torch.arange(0.01, 0.9501, 0.005)
    thresholds = search_best_thresholds_for_f1(
        calib_logits, calib["action_target"], grid=grid
    )["threshold_prob"]
    shifted = apply_posthoc_threshold(audit_logits, thresholds)
    metrics = multilabel_metrics_from_logits(
        shifted, audit["action_target"], prefix="Act_"
    )
    return {
        "name": name,
        "audit_mf1": float(metrics["Act_mF1"]),
        "audit_of1": float(metrics["Act_oF1"]),
        "audit_map": float(metrics["Act_mAP"]),
        "thresholds": thresholds.tolist(),
        "deployment_gain": gain.flatten().tolist(),
        "delta_rms": float(
            (audit["delta"] * gain).float().square().mean().sqrt()
        ),
    }


def _collect_candidate(base, refiner, loaders, device, cfg):
    calib = collect(base, refiner, loaders["calib"], device, cfg)
    audit = collect(base, refiner, loaders["audit"], device, cfg)
    verify_reason_identity(calib["base_reason"], calib["reason"])
    verify_reason_identity(audit["base_reason"], audit["reason"])
    gain, gain_rows = select_gains(
        calib, audit, cfg["stage_b"]["gain_candidates"]
    )
    return calib, audit, gain, gain_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-calib-samples", type=int)
    parser.add_argument("--max-audit-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    identity = json.loads(Path(args.run_identity).read_text(encoding="utf-8"))
    stage_a = validate_stage_checkpoint(
        args.stage_a_checkpoint, identity, expected_stage="base_selected"
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["data"]["split_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["runtime"].get("cpu_threads", 6)))
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    base = freeze_base_model(load_base(cfg, args.stage_a_checkpoint, device))
    refiner = build_refiner(base, cfg).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(cfg["stage_b"]["lr"]),
        weight_decay=float(cfg["stage_b"]["weight_decay"]),
    )

    train = make_dataset(cfg, "train")
    names = [sample.file_name for sample in train.samples]
    split = stable_split_ids(
        names,
        seed,
        float(cfg["data"]["train_calib_fraction"]),
        int(cfg["data"]["train_audit_count"]),
    )
    held_out = set(split["train_calib"]) | set(split["train_audit"])
    train_names = [name for name in names if name not in held_out]
    if args.max_train_samples:
        train_names = train_names[: args.max_train_samples]
    calib_names = split["train_calib"][: args.max_calib_samples or None]
    audit_names = split["train_audit"][: args.max_audit_samples or None]
    index = {sample.file_name: offset for offset, sample in enumerate(train.samples)}
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    accumulation = args.gradient_accumulation_steps or int(
        cfg["training"]["gradient_accumulation_steps"]
    )
    workers = args.num_workers if args.num_workers is not None else int(
        cfg["data"]["num_workers"]
    )
    loaders = {
        "train": make_loader(
            Subset(train, [index[name] for name in train_names]),
            batch_size, True, workers, cfg,
        ),
        "calib": make_loader(
            Subset(train, [index[name] for name in calib_names]),
            batch_size, False, workers, cfg,
        ),
        "audit": make_loader(
            Subset(train, [index[name] for name in audit_names]),
            batch_size, False, workers, cfg,
        ),
    }
    manifest = {
        "run_identity": identity,
        "stage": "action_refinement",
        "command_line": [sys.executable, *sys.argv],
        "stage_a_checkpoint": str(Path(args.stage_a_checkpoint).resolve()),
        "stage_a_checkpoint_sha256": sha256_file(args.stage_a_checkpoint),
        "external_task_checkpoint": None,
        "base_frozen": True,
        "reason_identity": True,
        "test_evaluated": False,
        "train_count": len(train_names),
        "train_calib_count": len(calib_names),
        "train_audit_count": len(audit_names),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "num_workers": workers,
    }
    _write_json(output / "run_manifest.json", manifest)

    initial_calib, initial_audit, _, _ = _collect_candidate(
        base, refiner, loaders, device, cfg
    )
    base_row = _audit_row(
        "base", initial_calib, initial_audit, torch.zeros(4)
    )
    candidate_rows = [base_row]
    candidate_states: dict[str, dict] = {}
    update = 0
    epochs = args.epochs or int(cfg["stage_b"]["epochs"])
    total_updates = max(1, math.ceil(len(loaders["train"]) / accumulation) * epochs)
    for epoch in range(epochs):
        refiner.train(); optimizer.zero_grad(set_to_none=True); window = []
        for micro, batch in enumerate(loaders["train"], 1):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                source = base_forward(base, images, cfg)
            refined = run_refiner(
                refiner,
                source,
                action_scale=float(cfg["evidence"]["action_scale"]),
                gain=torch.ones(4, device=device),
            )
            verify_reason_identity(source["reason_logits_final"], refined["reason_logits_final"])
            window.append((refined, target))
            if micro % accumulation and micro != len(loaders["train"]):
                continue
            logits = torch.cat([item[0]["action_logits_final"] for item in window])
            base_logits = torch.cat([item[0]["action_logits_base"] for item in window])
            delta = torch.cat([item[0]["action_delta"] for item in window])
            targets = torch.cat([item[1] for item in window])
            losses = {
                "action_asl": asymmetric_loss_with_logits(
                    logits, targets,
                    gamma_pos=float(cfg["stage_b_loss"]["gamma_pos"]),
                    gamma_neg=float(cfg["stage_b_loss"]["gamma_neg"]),
                    clip=float(cfg["stage_b_loss"]["clip"]),
                ),
                "action_pairwise_ap": action_pairwise_ap_loss(
                    logits, targets,
                    temperature=float(cfg["stage_b_loss"]["pair_temperature"]),
                ),
                "action_smooth_ap": action_smooth_ap_loss(
                    logits, targets,
                    temperature=float(cfg["stage_b_loss"]["smooth_ap_temperature"]),
                ),
                "base_margin_trust": base_margin_trust_loss(
                    logits, base_logits, targets,
                    tolerance=float(cfg["stage_b_loss"]["trust_tolerance"]),
                ),
                "residual_energy": residual_energy_loss(delta),
            }
            total = sum(
                float(cfg["stage_b_loss_weights"][name]) * value
                for name, value in losses.items()
            )
            total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                refiner.parameters(), float(cfg["stage_b"]["grad_clip"])
            ))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            update += 1; window = []
            if update == 1 or update % int(cfg["stage_b"]["print_every_updates"]) == 0:
                row = {
                    "event": "vetra_stage_b_batch", "epoch": epoch,
                    "update": update, "total_updates": total_updates,
                    "loss_total": float(total.detach()), "grad_norm": grad_norm,
                    "delta_rms": float(delta.detach().float().square().mean().sqrt()),
                    **{f"loss_{key}": float(value.detach()) for key, value in losses.items()},
                }
                print(json.dumps(row), flush=True)
                _append_jsonl(output / "loss_components.jsonl", row)
        calib, audit, gain, gain_rows = _collect_candidate(
            base, refiner, loaders, device, cfg
        )
        name = f"epoch_{epoch}"
        row = _audit_row(name, calib, audit, gain)
        candidate_rows.append(row)
        candidate_states[name] = {
            key: value.detach().cpu() for key, value in refiner.state_dict().items()
        }
        _append_jsonl(output / "candidate_metrics.jsonl", row)
        _write_json(output / f"gain_audit_epoch_{epoch:03d}.json", {
            "selected_gain": gain.tolist(), "per_action_candidates": gain_rows,
            "metrics": row,
        })
        print(json.dumps({"event": "vetra_stage_b_epoch", **row}), flush=True)

    selection = choose_refiner_candidate(
        candidate_rows,
        min_mf1_gain=float(cfg["stage_b"]["min_audit_mf1_gain"]),
        max_map_drop=float(cfg["stage_b"]["max_audit_map_drop"]),
    )
    selected_state = candidate_states.get(selection["name"])
    selected_gain = torch.tensor(selection["deployment_gain"])
    payload = make_stage_b_checkpoint(
        parent_path=args.stage_a_checkpoint,
        identity=identity,
        refiner_selected=bool(selection["refiner_selected"]),
        refiner_state=selected_state,
        deployment_gain=selected_gain,
        selection=selection,
    )
    payload["candidate_rows"] = candidate_rows
    payload["manifest"] = manifest
    checkpoint_path = output / "checkpoint_stage_b_selected.pth"
    torch.save(payload, checkpoint_path)
    _write_json(output / "stage_b_gain_audit.json", {
        "selection": selection, "candidates": candidate_rows,
        "refiner_selected": bool(selection["refiner_selected"]),
    })
    atomic_write_json(output / "STAGE_B_COMPLETE.json", {
        "complete": True,
        "stage": "stage_b",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parent_checkpoint_sha256": sha256_file(args.stage_a_checkpoint),
        "refiner_selected": bool(selection["refiner_selected"]),
        "selection": selection,
    })


if __name__ == "__main__":
    main()
