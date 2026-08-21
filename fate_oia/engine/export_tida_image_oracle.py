from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import yaml
from PIL import Image


ORACLE_TENSOR_KEYS = (
    "cls_tokens_by_layer", "patch_tokens_by_layer",
    "action_logits_primary", "reason_logits_primary",
    "predicate_logits", "predicate_probs", "predicate_tokens", "predicate_attention", "predicate_layer_weights",
    "action_nodes_primary", "reason_nodes_primary",
    "evidence_token", "evidence_map", "reference_point", "sampling_offsets", "sampling_weights", "layer_mixture",
    "bounded_contribution", "action_logits_final",
    "reason_private_attention", "reason_private_token", "reason_delta", "reason_logits_final",
    "action_visual_logits_primary", "action_reason_logits_primary", "action_fusion_gate_primary",
)


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_test_rows(manifest: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sorted((row for row in rows if row["partition"] == "test"), key=lambda row: row["file_name"])
    if len(selected) < count:
        raise ValueError(f"oracle requires {count} test rows, found {len(selected)}")
    return selected[:count]


def _checkpoint_scales(checkpoint: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
    scales = checkpoint.get("inference_scales") or {}
    return float(scales.get("action", 1.0)), float(scales.get("reason", config["reason_private"]["reason_scale_max"]))


def _load_source_model(source_root: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    source = str(source_root.resolve())
    sys.path = [source, *[entry for entry in sys.path if str(Path(entry or ".").resolve()) != source_root.resolve()]]
    from fate_oia.engine.train_aie_oia import build_model, canonical_model_state_dict
    from fate_oia.engine.train_vetra_strong_refine import build_refiner

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stage_b = stage if stage.get("stage") == "action_refined" else None
    parent_path = Path(stage_b["parent_checkpoint"]) if stage_b is not None else checkpoint_path
    if not parent_path.is_file():
        raise FileNotFoundError(f"VETRA parent checkpoint does not exist: {parent_path}")
    if stage_b is not None and _file_sha256(parent_path) != stage_b["parent_checkpoint_sha256"]:
        raise RuntimeError("VETRA Stage-B parent hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = build_model(config, device)
    model.load_state_dict(canonical_model_state_dict(parent["model"]), strict=True)
    refiner = None
    if stage_b is not None and bool(stage_b.get("refiner_selected")):
        refiner = build_refiner(model, config).to(device)
        refiner.load_state_dict(stage_b["refiner"], strict=True)
        refiner.set_deployment_gain(stage_b["deployment_gain"].to(device))
        refiner.eval()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, refiner, config, _checkpoint_scales(parent, config), parent_path


@torch.no_grad()
def _forward_source(model, refiner, images: torch.Tensor, scales: tuple[float, float]) -> dict[str, Any]:
    source = model(images, action_scale=scales[0], reason_scale=scales[1])
    if refiner is None:
        return source
    from fate_oia.models.vetra_strong_refiner import SelectiveActionPathRefiner, SelectiveVisualActionRankRefiner

    if isinstance(refiner, SelectiveVisualActionRankRefiner):
        refined = refiner(
            source["action_logits_final"], source["reason_logits_final"],
            source["action_nodes_primary"], source["evidence_token"],
        )
    elif isinstance(refiner, SelectiveActionPathRefiner):
        refined = refiner(source, action_scale=scales[0])
    else:
        raise TypeError(f"unsupported frozen VETRA refiner: {type(refiner).__name__}")
    if not torch.equal(refined["reason_logits_final"], source["reason_logits_final"]):
        raise RuntimeError("VETRA Stage-B refiner changed reason logits")
    return {**source, "action_logits_final": refined["action_logits_final"], "stage_b_action_delta": refined["action_delta"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.image_checkpoint).resolve()
    manifest_path = Path(args.clip_manifest).resolve()
    device = torch.device(args.device)
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("source worktree has tracked modifications")
    model, refiner, config, scales, parent_path = _load_source_model(
        source_root, config_path, checkpoint_path, device
    )
    from fate_oia.transforms import AspectRatioLetterboxTransform

    transform = AspectRatioLetterboxTransform(360, 640, patch_size=8)
    rows = _load_test_rows(manifest_path, int(args.count))
    inputs = torch.stack([transform(Image.open(row["target_image_path"]).convert("RGB")) for row in rows])
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in ORACLE_TENSOR_KEYS}
    for image in inputs.split(1):
        output = _forward_source(model, refiner, image.to(device), scales)
        missing = [key for key in ORACLE_TENSOR_KEYS if key not in output or not torch.is_tensor(output[key])]
        if missing:
            raise RuntimeError(f"source image output misses oracle tensors: {missing}")
        for key in ORACLE_TENSOR_KEYS:
            collected[key].append(output[key].detach().float().cpu())
    payload = {
        "schema": "tida_image_oracle_v1",
        "source_root": str(source_root),
        "source_head": _git(source_root, "rev-parse", "HEAD"),
        "source_tree": _git(source_root, "rev-parse", "HEAD^{tree}"),
        "source_tracked_clean": True,
        "image_checkpoint": str(checkpoint_path),
        "image_checkpoint_sha256": _file_sha256(checkpoint_path),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": _file_sha256(parent_path),
        "clip_manifest_sha256": _file_sha256(manifest_path),
        "action_scale": scales[0],
        "reason_scale": scales[1],
        "file_names": [row["file_name"] for row in rows],
        "target_image_paths": [row["target_image_path"] for row in rows],
        "input_tensor": inputs,
        "tensors": {key: torch.cat(values, dim=0) for key, values in collected.items()},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(json.dumps({
        "event": "tida_image_oracle", "output": str(output_path.resolve()),
        "count": len(rows), "source_head": payload["source_head"], "source_tree": payload["source_tree"],
    }), flush=True)


if __name__ == "__main__":
    main()
