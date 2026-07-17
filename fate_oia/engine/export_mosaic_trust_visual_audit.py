from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


class ICDORVisualExportError(RuntimeError):
    """Raised when a visual audit would omit or fabricate model evidence."""


_REQUIRED_EVIDENCE = (
    "action_final_logits",
    "reason_observed_logits",
    "factor_soft_masks",
    "action_support_mask",
    "action_veto_mask",
    "support_weights",
    "veto_weights",
)


def _tensor(output: Mapping[str, Any], name: str, batch_size: int) -> torch.Tensor:
    value = output.get(name)
    if not isinstance(value, torch.Tensor):
        raise ICDORVisualExportError(f"visual export requires real {name}")
    if value.ndim == 0 or value.shape[0] != batch_size:
        raise ICDORVisualExportError(f"visual export received invalid {name} batch shape")
    if not torch.isfinite(value).all():
        raise ICDORVisualExportError(f"visual export received non-finite {name}")
    return value.detach().float().cpu()


@torch.no_grad()
def export_visual_audit(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    device: torch.device,
    max_samples: int = 16,
    source_split: str = "train_audit",
) -> dict[str, Any]:
    """Persist only tensors returned by an actual IC-DOR forward with masks enabled."""
    if type(max_samples) is not int or max_samples <= 0:
        raise ICDORVisualExportError("visual export max_samples must be positive")
    if source_split not in {"train_audit", "audit_visual"}:
        raise ICDORVisualExportError("visual export source must be a visual audit split")
    root = Path(output_dir)
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    samples: list[dict[str, Any]] = []
    try:
        for batch in loader:
            images = batch.get("image")
            names = batch.get("file_name")
            splits = batch.get("split")
            if not isinstance(images, torch.Tensor) or not isinstance(names, list) or not isinstance(splits, list):
                raise ICDORVisualExportError("visual export batch must contain image, file_name, and split lists")
            if len(names) != images.shape[0] or len(splits) != images.shape[0]:
                raise ICDORVisualExportError("visual export metadata does not match image batch")
            if any(split != source_split for split in splits):
                raise ICDORVisualExportError(f"visual export only accepts {source_split} batches")
            output = model(
                images.to(device, non_blocking=True),
                route_mode="admitted",
                latent_enabled=True,
                reason_route_mode="full",
                return_masks=True,
            )
            if not isinstance(output, Mapping):
                raise ICDORVisualExportError("visual export forward must return a mapping")
            evidence = {name: _tensor(output, name, images.shape[0]) for name in _REQUIRED_EVIDENCE}
            for index, file_name in enumerate(names):
                if len(samples) >= max_samples:
                    break
                sample_id = len(samples)
                factor_masks = evidence["factor_soft_masks"][index]
                for factor_index, factor_mask in enumerate(factor_masks):
                    torch.save(factor_mask, mask_dir / f"{sample_id:04d}_factor_{factor_index:02d}.pt")
                    matched_random = torch.roll(
                        factor_mask,
                        shifts=(factor_mask.shape[-2] // 3, factor_mask.shape[-1] // 3),
                        dims=(-2, -1),
                    )
                    torch.save(
                        matched_random,
                        mask_dir / f"{sample_id:04d}_random_factor_{factor_index:02d}.pt",
                    )
                torch.save(evidence["action_support_mask"][index], mask_dir / f"{sample_id:04d}_action_support.pt")
                torch.save(evidence["action_veto_mask"][index], mask_dir / f"{sample_id:04d}_action_veto.pt")
                samples.append({
                    "sample_id": sample_id,
                    "file_name": str(file_name),
                    "split": source_split,
                    "action_final_logits": evidence["action_final_logits"][index].tolist(),
                    "reason_observed_logits": evidence["reason_observed_logits"][index].tolist(),
                    "support_weights": evidence["support_weights"][index].tolist(),
                    "veto_weights": evidence["veto_weights"][index].tolist(),
                    "factor_mask_files": [
                        str((mask_dir / f"{sample_id:04d}_factor_{factor_index:02d}.pt").relative_to(root))
                        for factor_index in range(factor_masks.shape[0])
                    ],
                    "matched_random_factor_mask_files": [
                        str((mask_dir / f"{sample_id:04d}_random_factor_{factor_index:02d}.pt").relative_to(root))
                        for factor_index in range(factor_masks.shape[0])
                    ],
                    "action_support_mask_file": str((mask_dir / f"{sample_id:04d}_action_support.pt").relative_to(root)),
                    "action_veto_mask_file": str((mask_dir / f"{sample_id:04d}_action_veto.pt").relative_to(root)),
                })
            if len(samples) >= max_samples:
                break
    finally:
        model.train(was_training)
    if not samples:
        raise ICDORVisualExportError(f"visual export received no {source_split} samples")
    manifest = {
        "schema": "icdor_visual_audit_v1",
        "source_split": source_split,
        "selection": f"deterministic_first_rows_from_frozen_{source_split}_split",
        "fixed_sample_ids": [sample["file_name"] for sample in samples],
        "sample_count": len(samples),
        "matched_random_control": "same_factor_equal_mass_spatial_roll",
        "samples": samples,
    }
    (root / "visual_audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pass": True, "sample_count": len(samples), "manifest": str(root / "visual_audit_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export already-computed IC-DOR visual-audit evidence")
    parser.add_argument("--manifest_only", action="store_true")
    parser.parse_args()
    raise ICDORVisualExportError(
        "CLI export requires an explicit model and visual-audit loader from the formal trainer; it never fabricates fallback logits or masks."
    )


if __name__ == "__main__":
    main()

